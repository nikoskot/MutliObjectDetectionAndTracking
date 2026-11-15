from pathlib import Path

import cv2 as cv
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
import torchvision

from .base_detector import BaseDetector

_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
}


class TRTDetector(BaseDetector):
    """
    Native TensorRT detector for YOLOv8-style engines (static shapes, batch 1,
    raw output of shape (1, 4 + num_classes, num_anchors), i.e. NO NMS baked in).

    All GPU buffers are torch tensors and all work runs on one torch CUDA stream,
    so there is a single CUDA context (PyTorch's) and no pycuda.

    Per-frame flow:
        host frame -> pinned buffer -> GPU (uint8)
        -> [GPU] resize/pad/BGR->RGB/CHW/normalize -> engine input tensor
        -> TensorRT execute_async_v3 on the same stream
        -> [GPU] class max, conf threshold, xywh->xyxy, per-class NMS
        -> one device-to-host copy of the final (N, 6) array

    Options:
        preprocess:  "gpu" -> upload the raw frame, letterbox on the GPU
                     "cpu" -> letterbox with OpenCV, upload the small uint8 canvas
        nms_backend: "gpu" -> torchvision batched_nms
                     "cpu" -> NumPy NMS on the thresholded candidates (for comparison)
    """

    def __init__(
        self,
        logger,
        monitor,
        engine_path: str,
        conf: float = 0.1,    # keep low so ByteTrack's second association pass has boxes
        iou: float = 0.7,
        max_det: int = 300,
        preprocess: str = "gpu",
        nms_backend: str = "gpu",
        class_names: dict[int, str] | list[str] | None = None,
        device_index: int = 0,
    ) -> None:
        super().__init__(logger, monitor, conf, iou, max_det)

        if preprocess not in ("gpu", "cpu"):
            raise ValueError(f"TRTDetector: preprocess must be 'gpu' or 'cpu', got {preprocess!r}")
        if nms_backend not in ("gpu", "cpu"):
            raise ValueError(f"TRTDetector: nms_backend must be 'gpu' or 'cpu', got {nms_backend!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("TRTDetector: CUDA is not available")

        self._engine_path = Path(engine_path)
        self._preprocess_mode = preprocess
        self._nms_backend = nms_backend
        self._device = torch.device("cuda", device_index)
        self._given_class_names = class_names

        # Filled in by _load()
        self._runtime = None
        self._engine = None
        self._context = None
        self._stream = None
        self._events = None
        self._input = None
        self._output = None
        self._in_h = self._in_w = 0
        self._num_classes = 0
        self._class_names: dict[int, str] = {}

        # Staging buffers for the uploaded frame (allocated lazily / when the frame shape changes)
        self._frame_shape = None
        self._pinned = None
        self._pinned_np = None
        self._frame_gpu = None

    # ======================================================================
    # Load / release
    # ======================================================================
    def _load(self) -> None:
        if not self._engine_path.is_file():
            raise FileNotFoundError(f"TRTDetector: engine not found: {self._engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(self._engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TRTDetector: failed to deserialize {self._engine_path}")
        self._context = self._engine.create_execution_context()

        self._stream = torch.cuda.Stream(device=self._device)
        self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        # ---- discover bindings ----
        inputs, outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            trt_dtype = self._engine.get_tensor_dtype(name)
            if any(d < 0 for d in shape):
                raise RuntimeError(f"TRTDetector: dynamic shape {shape} on '{name}' is not supported")
            if trt_dtype not in _TRT_TO_TORCH_DTYPE:
                raise RuntimeError(f"TRTDetector: unsupported dtype {trt_dtype} on '{name}'")
            entry = (name, shape, _TRT_TO_TORCH_DTYPE[trt_dtype])
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(entry)
            else:
                outputs.append(entry)

        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"TRTDetector: expected 1 input and 1 output, got {len(inputs)} and {len(outputs)}. "
                "Engines with NMS or extra outputs baked in are not supported."
            )
        (in_name, in_shape, in_dtype), (out_name, out_shape, out_dtype) = inputs[0], outputs[0]

        if len(in_shape) != 4 or in_shape[0] != 1 or in_shape[1] != 3:
            raise RuntimeError(f"TRTDetector: expected input (1, 3, H, W), got {in_shape}")
        if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] <= 4:
            raise RuntimeError(f"TRTDetector: expected output (1, 4+nc, anchors), got {out_shape}")

        self._in_h, self._in_w = in_shape[2], in_shape[3]
        self._num_classes = out_shape[1] - 4
        self._class_names = self._build_class_names(self._given_class_names, self._num_classes)

        # ---- allocate persistent GPU buffers on the stream; addresses are bound once ----
        with torch.cuda.stream(self._stream):
            self._input = torch.empty(in_shape, dtype=in_dtype, device=self._device)
            self._output = torch.empty(out_shape, dtype=out_dtype, device=self._device)
        self._context.set_tensor_address(in_name, self._input.data_ptr())
        self._context.set_tensor_address(out_name, self._output.data_ptr())

        self._logger.info(
            "TRTDetector: loaded %s | input %s %s | output %s %s | classes=%d | "
            "preprocess=%s | nms=%s | conf=%.2f iou=%.2f max_det=%d",
            self._engine_path.name, in_shape, in_dtype, out_shape, out_dtype, self._num_classes,
            self._preprocess_mode, self._nms_backend, self._conf, self._iou, self._max_det,
        )

    def _release(self) -> None:
        self._stream.synchronize()
        # Order matters: execution context first, then engine, then runtime, then tensors/stream.
        self._context = None
        self._engine = None
        self._runtime = None
        self._input = None
        self._output = None
        self._pinned = None
        self._pinned_np = None
        self._frame_gpu = None
        self._frame_shape = None
        self._events = None
        self._stream = None
        self._logger.info("TRTDetector: released")

    def _build_class_names(self, given, num_classes: int) -> dict[int, str]:
        if given is None:
            return {i: str(i) for i in range(num_classes)}
        names = dict(enumerate(given)) if isinstance(given, (list, tuple)) else dict(given)
        if len(names) != num_classes:
            self._logger.warning(
                "TRTDetector: %d class names given but the engine has %d classes", len(names), num_classes
            )
        return names

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    # ======================================================================
    # Per-frame work
    # ======================================================================
    def _detect_impl(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"TRTDetector: expected HxWx3 uint8 BGR image, got {image.shape} {image.dtype}")
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        ev = self._events

        with torch.cuda.stream(self._stream):
            ev[0].record(self._stream)

            if self._preprocess_mode == "gpu":
                gain, pad_left, pad_top = self._preprocess_gpu(image)
            else:
                gain, pad_left, pad_top = self._preprocess_cpu(image)
            ev[1].record(self._stream)

            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TRTDetector: execute_async_v3 failed")
            ev[2].record(self._stream)

            # The final .cpu() inside _postprocess is the only host sync of the frame.
            out = self._postprocess(gain, pad_left, pad_top, h, w)
            ev[3].record(self._stream)

        ev[3].synchronize()
        stages = {
            "preprocess": ev[0].elapsed_time(ev[1]),
            "inference": ev[1].elapsed_time(ev[2]),
            "postprocess": ev[2].elapsed_time(ev[3]),
        }
        return out, stages

    # ---- preprocessing ---------------------------------------------------
    def _letterbox_params(self, h: int, w: int):
        """Same geometry as Ultralytics' LetterBox (centered, pad value 114)."""
        gain = min(self._in_h / h, self._in_w / w)
        new_w, new_h = int(round(w * gain)), int(round(h * gain))
        dw, dh = (self._in_w - new_w) / 2, (self._in_h - new_h) / 2
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        return gain, new_w, new_h, left, right, top, bottom

    def _upload(self, image: np.ndarray) -> None:
        """Host uint8 HxWx3 -> pinned buffer -> GPU (async). Buffers are reused between frames.

        Reusing one pinned buffer is safe because every detect() call ends with a
        stream sync, so the previous copy has always finished.
        """
        if self._frame_shape != image.shape:
            self._pinned = torch.empty(image.shape, dtype=torch.uint8, pin_memory=True)
            self._pinned_np = self._pinned.numpy()
            self._frame_gpu = torch.empty(image.shape, dtype=torch.uint8, device=self._device)
            self._frame_shape = image.shape
        np.copyto(self._pinned_np, image)
        self._frame_gpu.copy_(self._pinned, non_blocking=True)

    def _preprocess_gpu(self, image: np.ndarray):
        """Upload the raw frame; resize, pad, BGR->RGB, normalize on the GPU."""
        h, w = image.shape[:2]
        gain, new_w, new_h, left, right, top, bottom = self._letterbox_params(h, w)

        self._upload(image)
        x = self._frame_gpu.permute(2, 0, 1).unsqueeze(0).float()          # (1, 3, h, w), BGR, 0..255
        if (new_h, new_w) != (h, w):
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False).round_()
        x = F.pad(x, (left, right, top, bottom), value=114.0)
        x = x.flip(1)                                                       # BGR -> RGB (on the small tensor)
        self._input.copy_(x.div_(255.0))                                    # copy_ casts to the engine dtype
        return gain, left, top

    def _preprocess_cpu(self, image: np.ndarray):
        """Letterbox with OpenCV (identical to Ultralytics), upload the small canvas."""
        h, w = image.shape[:2]
        gain, new_w, new_h, left, right, top, bottom = self._letterbox_params(h, w)

        if (new_w, new_h) != (w, h):
            image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))

        self._upload(image)
        x = self._frame_gpu.permute(2, 0, 1).unsqueeze(0).float().flip(1)   # (1, 3, H, W), RGB
        self._input.copy_(x.div_(255.0))
        return gain, left, top

    # ---- postprocessing --------------------------------------------------
    def _postprocess(self, gain: float, pad_left: int, pad_top: int, h: int, w: int) -> np.ndarray:
        pred = self._output[0].transpose(0, 1)                              # (anchors, 4 + nc)
        conf, cls = pred[:, 4:].max(dim=1)

        idx = torch.nonzero(conf > self._conf).squeeze(1)                   # dynamic size -> implicit sync
        if idx.numel() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        box = pred[idx, :4].float()                                         # cx, cy, w, h (letterbox space)
        conf = conf[idx].float()
        cls = cls[idx]
        xyxy = torch.cat((box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2), dim=1)

        # NMS in letterbox space, per class (matches Ultralytics' default agnostic=False)
        if self._nms_backend == "gpu":
            keep = torchvision.ops.batched_nms(xyxy, conf, cls, self._iou)
        else:
            xyxy, conf, cls = xyxy.cpu(), conf.cpu(), cls.cpu()
            keep = torch.from_numpy(self._nms_cpu(xyxy.numpy(), conf.numpy(), cls.numpy(), self._iou))
        keep = keep[: self._max_det]

        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        # Undo the letterbox and clip to the original image
        xyxy[:, 0::2] -= pad_left
        xyxy[:, 1::2] -= pad_top
        xyxy /= gain
        xyxy[:, 0::2].clamp_(0, w)
        xyxy[:, 1::2].clamp_(0, h)

        out = torch.cat((xyxy, conf[:, None], cls[:, None].float()), dim=1)
        return out.cpu().numpy()                                            # single D2H copy + sync

    @staticmethod
    def _nms_cpu(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thresh: float) -> np.ndarray:
        """Greedy per-class NMS in NumPy (class offset trick, same as torchvision's batched_nms)."""
        if len(boxes) == 0:
            return np.empty(0, dtype=np.int64)

        boxes = boxes + (class_ids.astype(np.float32) * (boxes.max() + 1.0))[:, None]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            rest = order[1:]
            if rest.size == 0:
                break
            inter_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
            inter_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thresh]

        return np.asarray(keep, dtype=np.int64)