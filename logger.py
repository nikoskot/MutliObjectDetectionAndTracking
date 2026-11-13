import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from types import TracebackType


class _RelativePathFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.relative_path = os.path.relpath(record.pathname, os.getcwd())
        record.location = f"{record.relative_path}:{record.lineno}"
        return super().format(record)

    def formatTime(
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        timestamp = datetime.fromtimestamp(record.created)
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S.%f")


class Logger(logging.Logger):
    """Log to the console and to a timestamped file in the given directory."""

    def __init__(self, log_directory: str | os.PathLike[str]) -> None:
        timestamp = datetime.now()
        super().__init__(f"application_logger.{id(self)}", level=logging.DEBUG)
        self.propagate = False

        output_directory = Path(log_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        # log_file = output_directory / timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f_log.txt")
        log_file = output_directory / "multi_object_detection_tracking.txt"

        formatter = _RelativePathFormatter(
            "%(asctime)s | %(levelname)-8s | %(location)-30s | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    #     sys.excepthook = self._handle_uncaught_exception

    # def _handle_uncaught_exception(
    #     self,
    #     exception_type: type[BaseException],
    #     exception: BaseException,
    #     traceback: TracebackType | None,
    # ) -> None:
    #     if issubclass(exception_type, (KeyboardInterrupt, SystemExit)):
    #         sys.__excepthook__(exception_type, exception, traceback)
    #         return

    #     self.critical(
    #         "Uncaught exception",
    #         exc_info=(exception_type, exception, traceback),
    #     )
