import contextlib
import time


class Timing(contextlib.ContextDecorator):
    """
    taken from https://github.com/tinygrad/tinygrad/blob/master/tinygrad/helpers.py#L86 and modified slightly
    """
    def __init__(self, prefix="", on_exit=None,
                 enabled=True):
        self.prefix = prefix
        self.on_exit = on_exit
        self.enabled = enabled

    def __enter__(self): self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(
            f"{self.prefix}{self.et * 1e-6:.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))
