from threading import Event, Thread

from services.session_workspace import SessionWorkspace


def test_close_waits_for_active_job_before_removing_workspace() -> None:
    workspace = SessionWorkspace(1)
    path = workspace.path
    started = Event()
    release = Event()

    def run_job() -> None:
        with workspace.active_job():
            started.set()
            release.wait(timeout=2.0)

    thread = Thread(target=run_job)
    thread.start()
    assert started.wait(timeout=2.0)

    workspace.close()
    assert path.exists()

    release.set()
    thread.join(timeout=2.0)
    assert not path.exists()
