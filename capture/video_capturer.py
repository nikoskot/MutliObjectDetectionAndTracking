import cv2 as cv
import numpy as np
from base_capturer import Capturer


class VideoCapturer(Capturer):
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
            raise ValueError(f"VideoCapturer: Invalid video source received ({self._video_source})")
        
    def _open(self):
        self._cap = cv.VideoCapture(self._video_source, cv.CAP_ANY)
        if not self._cap.isOpened():
            raise RuntimeError("VideoCapturer: Cannot open video source")
        else: 
            self._logger.info(f"VideoCapturer: Capturer initialized with video_source={self._video_source}")
    
    def _read_impl(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            self._logger.debug("VideoCapturer: Can't receive frame from video source (stream end?).")
            return None
        return frame
    
    def _release(self):
        self._cap.release()

    @property
    def frame_size(self):
        raise NotImplementedError()
    
    @property
    def fps(self):
        raise NotImplementedError()
    
    @property
    def total_frames(self):
        raise NotImplementedError()
        