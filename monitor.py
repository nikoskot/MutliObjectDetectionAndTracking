import time


def now() -> int:
    return time.perf_counter_ns()