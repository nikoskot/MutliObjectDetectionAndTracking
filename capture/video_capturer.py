import cv2 as cv
import numpy as np

from .base_capturer import BaseCapturer


class VideoCapturer(BaseCapturer):
    def __init__(
        self, 
        logger, 
        monitor,
        video_source
    ) -> None:
        super().__init__(logger, monitor)
        self._video_source = video_source
        self._cap = None
        
        if self._video_source is None or self._video_source == "":
            self._logger.error("VideoCapturer: Invalid video source received (%s)", self._video_source)
            raise FileNotFoundError("VideoCapturer: Invalid video source received (%s)", self._video_source)
        
    def _open(self):
        self._cap = cv.VideoCapture(self._video_source, cv.CAP_ANY)
        if not self._cap.isOpened():
            self._logger.error("VideoCapturer: Cannot open video source")
            raise RuntimeError("VideoCapturer: Cannot open video source")
        else: 
            self._logger.info("VideoCapturer: Capturer initialized with video_source=%s", self._video_source)
    
    def _read_impl(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            self._logger.debug("VideoCapturer: Can't receive frame from video source (stream end?).")
            return None
        return frame
    
    def _release(self):
        if self._cap is not None: self._cap.release()
        self._logger.debug("VideoCapturer: Capturer released")

    @property
    def frame_size(self):
        raise NotImplementedError()
    
    @property
    def fps(self):
        raise NotImplementedError()
    
    @property
    def total_frames(self):
        raise NotImplementedError()
        