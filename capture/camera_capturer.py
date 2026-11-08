import cv2 as cv
import numpy as np

from .base_capturer import Capturer


class CameraCapturer(Capturer):
    def __init__(
        self, 
        logger, 
        monitor,
        camera_index,
        frame_width,
        frame_height,
        fps
    ) -> None:
        super().__init__(logger, monitor)
        self._camera_index = camera_index
        self._cap = None
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._fps = fps
        
        if self._camera_index < 0:
            self._logger.error(f"CameraCapturer: Invalid camera index received ({self._camera_index})")
            return
        
    def _open(self):
        self._cap = cv.VideoCapture(self._camera_index)
        self._cap.set(cv.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv.CAP_PROP_FPS, self._fps)
        if not self._cap.isOpened():
            self._logger.error("CameraCapturer: Cannot open camera source")
            return
        else: 
            self._logger.info(f"CameraCapturer: Capturer initialized with camera_source={self._camera_index}, frame_width={self._cap.get(cv.CAP_PROP_FRAME_WIDTH)}, frame_height={self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)}, fps={self._cap.get(cv.CAP_PROP_FPS)}")
    
    def _read_impl(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            self._logger.debug("CameraCapturer: Can't receive frame from camera source (stream end?).")
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
        