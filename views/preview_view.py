from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf
from viser.transforms import SO3

from machine import BUILD_PLATE_CENTER
from models import GcodePreview, MachinePose
from views.theming import PENTOS_ORANGE

if TYPE_CHECKING:
    from controllers.preview_controller import PreviewController

SETUP_COLOR = PENTOS_ORANGE
URDF_SCALE = 1000.0
URDF_XYZ_HOME_OFFSET_MM = np.array([-2.3, 13.8, 0.0])
URDF_B_AXIS = np.array([0.00321170129672137, 0.0, -0.99999484247409])
URDF_BED_NORMAL = -URDF_B_AXIS / np.linalg.norm(URDF_B_AXIS)
URDF_BED_SURFACE_OFFSET = URDF_BED_NORMAL * 0.05349142183056172
URDF_BED_SURFACE_ROTATION = np.column_stack(
    (
        np.cross([0.0, 1.0, 0.0], URDF_BED_NORMAL),
        [0.0, 1.0, 0.0],
        URDF_BED_NORMAL,
    )
)
URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "Pentos_URDF"
    / "urdf"
    / "Pentos_URDF.urdf"
)


def bed_surface_pose(
    bed_link_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bed_rotation = bed_link_transform[:3, :3]
    return (
        bed_rotation @ URDF_BED_SURFACE_ROTATION,
        bed_link_transform[:3, 3] + bed_rotation @ URDF_BED_SURFACE_OFFSET,
    )


def machine_root_pose(
    bed_link_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bed_surface_rotation, bed_surface_position = bed_surface_pose(bed_link_transform)
    root_rotation = bed_surface_rotation.T
    root_position = (
        np.asarray(BUILD_PLATE_CENTER)
        - root_rotation @ bed_surface_position * URDF_SCALE
    )
    return SO3.from_matrix(root_rotation).wxyz, root_position


def build_plate_pose(
    machine_root_wxyz: np.ndarray,
    machine_root_position: np.ndarray,
    bed_link_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root_rotation = SO3(machine_root_wxyz).as_matrix()
    bed_surface_rotation, bed_surface_position = bed_surface_pose(bed_link_transform)
    scene_rotation = root_rotation @ bed_surface_rotation
    scene_position = (
        machine_root_position + root_rotation @ bed_surface_position * URDF_SCALE
    )
    plate_position = scene_position - scene_rotation @ np.asarray(BUILD_PLATE_CENTER)
    return SO3.from_matrix(scene_rotation).wxyz, plate_position


def machine_configuration(
    joint_names: tuple[str, ...],
    pose: MachinePose,
) -> np.ndarray:
    x, y, z = pose.xyz - URDF_XYZ_HOME_OFFSET_MM
    a_degrees, b_degrees = pose.ab
    values = {
        "x_joint": x / 1000.0,
        "y_joint": y / 1000.0,
        "z_joint": z / 1000.0,
        "a_joint": np.radians(-a_degrees),
        "b_joint": np.radians(b_degrees),
    }
    return np.array([values[name] for name in joint_names])


@dataclass(frozen=True)
class PreviewControls:
    status: viser.GuiTextHandle
    output_path: viser.GuiTextHandle
    show_travel: viser.GuiCheckboxHandle
    line_width: viser.GuiNumberHandle[float]
    show_machine: viser.GuiCheckboxHandle
    machine_move: viser.GuiSliderHandle[int]
    back_button: viser.GuiButtonHandle
    download_button: viser.GuiButtonHandle


class PreviewView:
    def __init__(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.controller: PreviewController
        self.controls: PreviewControls | None = None
        self.machine_root: viser.FrameHandle | None = None
        self.build_plate: viser.FrameHandle | None = None
        self.machine_root_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.machine_root_position = np.zeros(3)
        self.machine_urdf: yourdfpy.URDF | None = None
        self.machine: ViserUrdf | None = None
        self.machine_poses: list[MachinePose] = []
        self.travel_handles: list[viser.LineSegmentsHandle] = []
        self.extrusion_handles: list[viser.LineSegmentsHandle] = []

    def bind_controller(self, controller: PreviewController) -> None:
        self.controller = controller

    def mount(self, gcode_path: Path | None) -> None:
        controls = PreviewControls(
            status=self.client.gui.add_text(
                "Status",
                "Saved G-code",
                disabled=True,
            ),
            output_path=self.client.gui.add_text(
                "Output G-code",
                "" if gcode_path is None else gcode_path.name,
                disabled=True,
            ),
            show_travel=self.client.gui.add_checkbox("Travel", True),
            line_width=self.client.gui.add_number(
                "Line width",
                2.0,
                min=1.0,
                max=10.0,
            ),
            show_machine=self.client.gui.add_checkbox("Show machine", True),
            machine_move=self.client.gui.add_slider(
                "Machine move",
                min=0,
                max=1,
                step=1,
                initial_value=0,
                disabled=True,
            ),
            download_button=self.client.gui.add_button(
                "Download G-code",
                icon=viser.Icon.DOWNLOAD,
            ),
            back_button=self.client.gui.add_button("Back to Setup"),
        )
        self.controls = controls

        @controls.show_travel.on_update
        def _(_) -> None:
            visible = controls.show_travel.value
            for travel_handle in self.travel_handles:
                travel_handle.visible = visible

        @controls.line_width.on_update
        def _(_) -> None:
            width = controls.line_width.value
            for extrusion_handle in self.extrusion_handles:
                extrusion_handle.line_width = width
            for travel_handle in self.travel_handles:
                travel_handle.line_width = max(1.0, width * 0.5)

        @controls.show_machine.on_update
        def _(_) -> None:
            if self.machine is not None:
                self.machine.show_visual = controls.show_machine.value

        @controls.machine_move.on_update
        def _(_) -> None:
            self._set_machine_pose(int(controls.machine_move.value))

        @controls.back_button.on_click
        def _(_) -> None:
            self.controller.show_setup()

        @controls.download_button.on_click
        def _(event) -> None:
            download = self.controller.download_gcode()
            if download is None:
                return
            filename, content = download
            assert event.client is not None
            event.client.send_file_download(
                filename,
                content,
                save_immediately=True,
            )

    def set_status(self, message: str) -> None:
        self._mounted().status.value = message

    def show_preview(self, preview: GcodePreview) -> None:
        controls = self._mounted()
        self.machine_poses = preview.machine_poses
        self.machine_root = self.client.scene.add_frame(
            "/preview/machine",
            show_axes=False,
        )
        self.build_plate = self.client.scene.add_frame(
            "/shared/build_plate",
            show_axes=False,
        )
        self.machine_urdf = yourdfpy.URDF.load(
            URDF_PATH,
            build_scene_graph=True,
            load_meshes=True,
            filename_handler=partial(
                yourdfpy.filename_handler_magic,
                dir=URDF_PATH.parent,
            ),
        )
        self.machine = ViserUrdf(
            self.client,
            self.machine_urdf,
            scale=URDF_SCALE,
            root_node_name="/preview/machine",
        )
        self.machine.show_visual = controls.show_machine.value
        self.machine.update_cfg(np.zeros(len(self.machine.get_actuated_joint_names())))
        self._align_machine()
        if self.machine_poses:
            controls.machine_move.max = max(1, len(self.machine_poses) - 1)
            controls.machine_move.disabled = len(self.machine_poses) == 1
            self._set_machine_pose(0)
        else:
            self._move_virtual_bed()

        line_width = controls.line_width.value
        travel_visible = controls.show_travel.value

        if len(preview.setup):
            self.travel_handles.append(
                self.client.scene.add_line_segments(
                    "/shared/build_plate/preview/setup",
                    points=preview.setup,
                    colors=SETUP_COLOR,
                    line_width=max(1.0, line_width * 0.5),
                    visible=travel_visible,
                )
            )

        for index, part in enumerate(preview.parts):
            if len(part.extrusion):
                self.extrusion_handles.append(
                    self.client.scene.add_line_segments(
                        f"/shared/build_plate/preview/part_{index}/extrusion",
                        points=part.extrusion,
                        colors=part.color,
                        line_width=line_width,
                    )
                )

            if len(part.travel):
                self.travel_handles.append(
                    self.client.scene.add_line_segments(
                        f"/shared/build_plate/preview/part_{index}/travel",
                        points=part.travel,
                        colors=part.color,
                        line_width=max(1.0, line_width * 0.5),
                        visible=travel_visible,
                    )
                )

    def _set_machine_pose(self, index: int) -> None:
        if self.machine is None or not self.machine_poses:
            return
        with self.client.atomic():
            self.machine.update_cfg(
                machine_configuration(
                    self.machine.get_actuated_joint_names(),
                    self.machine_poses[index],
                )
            )
            self._move_virtual_bed()

    def _bed_transform(self) -> np.ndarray | None:
        if self.machine_urdf is None:
            return None
        return self.machine_urdf.get_transform(
            "b_rotary_link",
            self.machine_urdf.base_link,
        )

    def _align_machine(self) -> None:
        bed_transform = self._bed_transform()
        if self.machine_root is None or bed_transform is None:
            return
        self.machine_root_wxyz, self.machine_root_position = machine_root_pose(
            bed_transform
        )
        self.machine_root.wxyz = self.machine_root_wxyz
        self.machine_root.position = self.machine_root_position

    def _move_virtual_bed(self) -> None:
        bed_transform = self._bed_transform()
        if self.build_plate is None or bed_transform is None:
            return
        self.build_plate.wxyz, self.build_plate.position = build_plate_pose(
            self.machine_root_wxyz,
            self.machine_root_position,
            bed_transform,
        )

    def unmount(self) -> None:
        if self.controls is None:
            return

        if self.build_plate is not None:
            self.build_plate.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
            self.build_plate.position = np.zeros(3)
        if self.machine_root is not None:
            self.machine_root.remove()

        controls = self.controls
        for handle in (
            *self.extrusion_handles,
            *self.travel_handles,
            controls.back_button,
            controls.line_width,
            controls.show_travel,
            controls.machine_move,
            controls.show_machine,
            controls.download_button,
            controls.output_path,
            controls.status,
        ):
            handle.remove()

        self.controls = None
        self.machine_root = None
        self.build_plate = None
        self.machine_root_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.machine_root_position = np.zeros(3)
        self.machine_urdf = None
        self.machine = None
        self.machine_poses = []
        self.travel_handles = []
        self.extrusion_handles = []

    def _mounted(self) -> PreviewControls:
        if self.controls is None:
            raise RuntimeError("Preview view is not mounted")
        return self.controls
