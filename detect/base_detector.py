from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from capture.base_capturer import Frame
from monitor import now


@dataclass
class Detections:
    boxes: np.ndarray        # (N, 4) float32, xyxy, original-image pixel coordinates
    scores: np.ndarray       # (N,)   float32
    class_ids: np.ndarray    # (N,)   int32
    frame_index: int
    t_capture_ns: int

    def __len__(self) -> int:
        return len(self.scores)


class BaseDetector(ABC):
    """
    Template-method base class.

    Lifecycle:
        detector = SomeDetector(...)   # cheap, stores config only
        detector.load()                # allocates model / GPU resources (call in the thread that will use it)
        detector.warmup(image)
        detector.detect(frame)         # many times
        detector.release()             # idempotent, call it in a try/finally
    """

    def __init__(
        self, 
        logger, 
        monitor, 
        conf: float, 
        iou: float, 
        max_det: int
    ) -> None:
        self._logger = logger
        self._monitor = monitor
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._loaded = False

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def release(self) -> None:
        if not self._loaded:
            return
        self._release()
        self._loaded = False

    def warmup(
        self, 
        image: np.ndarray, 
        n: int = 10
    ) -> None:
        """Run n inferences that are NOT recorded in the monitor (first calls are slow)."""
        self._require_loaded()
        for _ in range(n):
            self._detect_impl(image)
        self._logger.info("%s: warmup done (%d runs)", type(self).__name__, n)

    # ---- public API ------------------------------------------------------
    def detect(self, frame: Frame) -> Detections:
        self._require_loaded()

        t0 = now()
        out, stage_durations_ns = self._detect_impl(frame.image)
        t1 = now()

        self._monitor.record("detect", t1 - t0)
        for name, duration_ns in stage_durations_ns.items():
            self._monitor.record(name, duration_ns)

        detections = Detections(
            boxes=np.ascontiguousarray(out[:, :4], dtype=np.float32),
            scores=np.ascontiguousarray(out[:, 4], dtype=np.float32),
            class_ids=out[:, 5].astype(np.int32),
            frame_index=frame.index,
            t_capture_ns=frame.t_capture_ns,
        )
        self._logger.debug("Detector: frame=%d detections=%d", frame.index, len(detections))
        return detections

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(f"{type(self).__name__}: call load() before use")

    # ---- subclass contract ----------------------------------------------
    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _release(self) -> None: ...

    @abstractmethod
    def _detect_impl(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """
        Returns:
            out: (N, 6) float32 array [x1, y1, x2, y2, score, class_id] in original-image
                 pixel coordinates (shape (0, 6) when nothing is detected, never None).
            stage_durations_ns: e.g. {"preprocess": ..., "inference": ..., "postprocess": ...}
        """

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]: ...