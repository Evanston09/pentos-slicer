from typing import Protocol


class SceneView(Protocol):
    def mount(self) -> None: ...

    def unmount(self) -> None: ...
