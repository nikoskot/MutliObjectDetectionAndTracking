import cv2 as cv
import numpy as np

from .base_capturer import Capturer


class ImagesCapturer(Capturer):
    def __init__(
        self, 
        logger, 
        monitor,
        images_abs_path
    ) -> None:
        super().__init__(logger, monitor)
        self._images_abs_path = images_abs_path
        self._cap = None
        
        if self._images_abs_path is None or self._images_abs_path == "":
            self._logger.error(f"ImagesCapturer: Invalid images absolute path received ({self._images_abs_path})")
            return
        
    def _open(self):
        self._cap = cv.VideoCapture(self._images_abs_path, cv.CAP_IMAGES)
        if not self._cap.isOpened():
            self._logger.error("ImagesCapturer: Cannot open images source")
            return
        else: 
            self._logger.info(f"ImagesCapturer: Capturer initialized with images_source={self._images_abs_path}")
    
    def _read_impl(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            self._logger.debug("ImagesCapturer: Can't receive frame from images source (stream end?).")
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
        