from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import monitor


@dataclass
class Frame:
    image: np.ndarray
    index: int
    t_capture_ns: int


class Capturer(ABC):
    def __init__(
        self, 
        logger, 
        monitor
    ) -> None:
        self._logger = logger
        self._monitor = monitor
        self._index = 0

    # For context manager: __enter__ opens, __exit__ releases
    def __enter__(self): self._open(); return self
    def __exit__(self, *exc): self.release()

    def read(self) -> Frame | None:
        self._before_read()
        
        t0 = monitor.now()
        image = self._read_impl()
        t1 = monitor.now()
        
        if image is None:
            self._logger.debug("Capturer: Captured None frame")
            return None                             # end-of-stream contract
        
        self._monitor.record("capture", t1 - t0)
        
        frame = Frame(image, self._index, t1)
        self._logger.debug(f"Capturer: Captured frame with shape={frame.image.shape}, index={frame.index}")
        self._index += 1
        return frame

    def release(self):
        self._release()

    # subclass contract
    @abstractmethod
    def _open(self): ...
    
    @abstractmethod
    def _read_impl(self) -> np.ndarray | None: ...
    
    @abstractmethod
    def _release(self): ...
    
    def _before_read(self): pass                    # hook, default no-op

    @property
    @abstractmethod
    def frame_size(self): ...
    
    @property
    @abstractmethod
    def fps(self): ...
    
    @property
    @abstractmethod
    def total_frames(self): ...