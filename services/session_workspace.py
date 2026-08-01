from contextlib import contextmanager
from pathlib import Path
import tempfile
from threading import Lock
from typing import Iterator


class SessionWorkspace:
    def __init__(self, client_id: int) -> None:
        runtime_dir = Path(tempfile.gettempdir()) / "pentos-slicer"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f"{client_id}-",
            dir=runtime_dir,
        )
        self.path = Path(self._temporary_directory.name)
        self._lock = Lock()
        self._job_active = False
        self._closed = False

    @contextmanager
    def active_job(self) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            if self._job_active:
                raise RuntimeError("A slice is already running for this session")
            self._job_active = True

        try:
            yield
        finally:
            with self._lock:
                self._job_active = False
                cleanup = self._closed
            if cleanup:
                self._temporary_directory.cleanup()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            cleanup = not self._job_active
        if cleanup:
            self._temporary_directory.cleanup()
