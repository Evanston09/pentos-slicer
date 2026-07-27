from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator


class SlicingBusyError(RuntimeError):
    pass


class SlicingCoordinator:
    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("MAX_CONCURRENT_SLICES must be at least 1")
        self._semaphore = BoundedSemaphore(max_concurrent)

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._semaphore.acquire(blocking=False):
            raise SlicingBusyError("Server is busy slicing other models")
        try:
            yield
        finally:
            self._semaphore.release()
