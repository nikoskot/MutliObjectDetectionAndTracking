from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector


class UltralyticsDetector(BaseDetector):
    """
    Detector backed by the Ultralytics YOLO implementation.

    Works with both PyTorch weights (.pt) and exported TensorRT engines (.engine):
    Ultralytics picks the backend from the file extension. Preprocessing (letterbox),
    inference and NMS are all done by Ultralytics, so this is the reference
    implementation to compare TRTDetector against.

    Notes:
      - For .engine files the input size, batch size and precision are fixed at export
        time, so `imgsz` must match the engine and `half` has no effect on it.
      - For .pt files `half=True` runs FP16.
    """

    def __init__(
        self,
        logger,
        monitor,
        model_path: str,
        conf: float = 0.1,    # keep low so ByteTrack's second association pass has boxes
        iou: float = 0.7,     # Ultralytics predict default
        max_det: int = 300,   # Ultralytics default
        imgsz: int = 640,
        half: bool = False,
        class_names: dict[int, str] | list[str] | None = None,
        device_index: int | str = 0,
    ) -> None:
        super().__init__(logger, monitor, conf, iou, max_det)
        self._model_path = Path(model_path)
        self._imgsz = imgsz
        self._half = half
        self._device = device_index
        self._model = None
        self._class_names: dict[int, str] = {}
        self._given_class_names = class_names

    # ---- lifecycle -------------------------------------------------------
    def _load(self) -> None:
        if not self._model_path.is_file():
            raise FileNotFoundError(f"UltralyticsDetector: model not found: {self._model_path}")

        self._model = YOLO(str(self._model_path), task="detect", verbose=True)
        
        self._model_class_names = dict(self._model.names)
        self._class_names = self._build_class_names(self._given_class_names, self._model_class_names)
        

        self._logger.info(
            "UltralyticsDetector: loaded %s | classes=%d | imgsz=%d half=%s device=%s | "
            "conf=%.2f iou=%.2f max_det=%d",
            self._model_path.name, len(self._class_names), self._imgsz, self._half, self._device,
            self._conf, self._iou, self._max_det,
        )

    def _build_class_names(self, given, model_classes) -> dict[int, str]:
        if given is None:
            return {i: cls for i, cls in model_classes.items()}
        names = dict(enumerate(given)) if isinstance(given, (list, tuple)) else dict(given)
        if len(names) != len(list(model_classes.key())):
            self._logger.warning(
                "UltralyticsDetector: %d class names given but the model has %d classes", len(names), len(list(model_classes.key()))
            )
        return names
        
    def _release(self) -> None:
        self._model = None
        self._logger.info("UltralyticsDetector: released")

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    # ---- per-frame work --------------------------------------------------
    def _detect_impl(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        # A numpy image is interpreted by Ultralytics as BGR (OpenCV convention).
        results = self._model.predict(
            source=image,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            half=self._half,
            device=self._device,
            verbose=False,
        )
        res = results[0]

        # Ultralytics reports its own per-stage times in ms
        stages = {name: ms for name, ms in res.speed.items()}

        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32), stages

        # boxes.data is (N, 6): x1, y1, x2, y2, conf, cls in original-image pixel coordinates.
        # The .cpu() copy is the host sync point.
        out = res.boxes.data.cpu().numpy().astype(np.float32, copy=False)
        return out, stages