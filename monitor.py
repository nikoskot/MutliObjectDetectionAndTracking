import time


def now() -> float:
    return time.perf_counter()


class Monitor:
    def __init__(self, logger):
        self._logger = logger
        self._stages_durations = {}
    
    def reset_durations(self):
        self._stages_durations = {}
        
    def record(self, stage_name, duration):
        self._stages_durations[stage_name] = duration
    
    def log_stages_durations(self):
        self._logger.debug("Monitor: Stages durations " + ", ".join(f"{name}={dur:.4f}ms" for name, dur in self._stages_durations.items()))