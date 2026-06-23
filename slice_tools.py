import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

import gcode_tools
from machine import BUILD_PLATE_CENTER, ROTATION_CENTER, rotation_matrix


class SlicePlane(Protocol):
    position: np.ndarray
    wxyz: np.ndarray


@dataclass
class Chunk:
    path: Path
    print_up_normal: np.ndarray | None
    z_offset: float
    flat_xy_offset: np.ndarray
    a_degrees: float
    b_degrees: float


class Slicer:
    def __init__(
        self,
        out_dir: Path = Path("output"),
        temp_dir: Path = Path("temp"),
        rotation_center: np.ndarray | tuple[float, float, float] = ROTATION_CENTER,
    ) -> None:
        self.out_dir = out_dir
        self.temp_dir = temp_dir
        self.rotation_center = np.array(rotation_center)
        self.out_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def slice(
        self,
        mesh: trimesh.Trimesh,
        planes: list[SlicePlane],
        source_name: str = "model",
    ) -> Path:
        chunks = self.export_stl_chunks(mesh, planes, source_name)
        if not chunks:
            raise ValueError("No chunks were generated")

        gcode_paths = self.run_prusa_slicer(chunks)
        return self.merge_gcode_files(gcode_paths, chunks, source_name)

    def debug_transition_check(
        self,
        mesh: trimesh.Trimesh,
        planes: list[SlicePlane],
        source_name: str = "model",
    ) -> Path:
        chunks = self.export_stl_chunks(mesh, planes, source_name)
        if not chunks:
            raise ValueError("No chunks were generated")

        gcode_paths = self.run_prusa_slicer(chunks)
        output_path = self.out_dir / f"{source_name}_debug.gcode"
        return gcode_tools.generate_debug_transition_check(
            gcode_paths,
            chunks,
            output_path,
        )

    def run_prusa_slicer(
        self,
        chunks: list[Chunk],
        config_path: Path = Path("pentos_config.ini"),
        slicer_cmd: str = "prusa-slicer",
    ) -> list[Path]:
        gcode_paths = []

        for index, chunk in enumerate(chunks):
            gcode_path = chunk.path.with_suffix(".gcode")
            command = [
                slicer_cmd,
                "--export-gcode",
                str(chunk.path),
                "--load",
                str(config_path),
                "--dont-arrange",
                "--output",
                str(gcode_path),
            ]

            if index > 0:
                command.extend(
                    [
                        "--skirts",
                        "0",
                        "--skirt-height",
                        "0",
                        "--min-skirt-length",
                        "0",
                        "--brim-type",
                        "no_brim",
                    ]
                )

            subprocess.run(command, check=True)
            gcode_paths.append(gcode_path)

        return gcode_paths

    def merge_gcode_files(
        self,
        gcode_paths: list[Path],
        chunks: list[Chunk],
        source_name: str,
    ) -> Path:
        output_path = self.out_dir / f"{source_name}.gcode"
        return gcode_tools.merge_gcode_files(gcode_paths, chunks, output_path)

    def export_stl_chunks(
        self, mesh: trimesh.Trimesh, planes: list[SlicePlane], source_name: str
    ) -> list[Chunk]:
        pieces = [(mesh, None)]

        for plane in planes:
            plane_position = plane.position
            normal = self._plane_normal(plane)
            next_pieces = []

            for piece, print_up_normal in pieces:
                part_a = piece.slice_plane(plane_position, normal, cap=True)
                part_b = piece.slice_plane(plane_position, -normal, cap=True)

                if self._is_valid(part_b):
                    next_pieces.append((part_b, print_up_normal))
                if self._is_valid(part_a):
                    next_pieces.append((part_a, normal))

            pieces = next_pieces

        chunks = []
        for index, (piece, print_up_normal) in enumerate(pieces):
            path = self.temp_dir / f"{source_name}_{mesh.identifier_hash}_{index}.stl"
            piece, z_offset, flat_xy_offset, a_degrees, b_degrees = self._orient(
                piece,
                print_up_normal,
            )
            piece.export(path)
            chunks.append(
                Chunk(
                    path=path,
                    print_up_normal=print_up_normal,
                    z_offset=z_offset,
                    flat_xy_offset=flat_xy_offset,
                    a_degrees=a_degrees,
                    b_degrees=b_degrees,
                )
            )

        return chunks

    @staticmethod
    def _plane_normal(plane: SlicePlane) -> np.ndarray:
        # Only care about rotation and we start pointing up
        rotation_matrix = trimesh.transformations.quaternion_matrix(plane.wxyz)[:3, :3]
        normal = rotation_matrix @ np.array([0.0, 0.0, 1.0])
        return normal

    def _orient(
        self, mesh: trimesh.Trimesh, print_up_normal: np.ndarray | None
    ) -> tuple[trimesh.Trimesh, float, np.ndarray, float, float]:
        a_degrees, b_degrees = self.ab_angles(print_up_normal)
        flat_xy_offset = np.zeros(2)

        if print_up_normal is not None:
            transform = np.eye(4)
            transform[:3, :3] = self.rotation_matrix(a_degrees, b_degrees)
            transform = trimesh.transformations.transform_around(
                transform,
                self.rotation_center,
            )
            mesh.apply_transform(transform)

        z_offset = float(mesh.bounds[0][2])
        mesh.apply_translation([0, 0, -z_offset])

        if print_up_normal is not None:
            flat_center = mesh.bounds.mean(axis=0)[:2]
            flat_xy_offset = BUILD_PLATE_CENTER[:2] - flat_center
            mesh.apply_translation([flat_xy_offset[0], flat_xy_offset[1], 0.0])

        return mesh, z_offset, flat_xy_offset, a_degrees, b_degrees

    @staticmethod
    def _is_valid(mesh: trimesh.Trimesh | None) -> bool:
        return mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0

    @staticmethod
    def rotation_matrix(a_degrees: float, b_degrees: float) -> np.ndarray:
        return rotation_matrix(a_degrees, b_degrees)

    @staticmethod
    def ab_angles(print_up_normal: np.ndarray | None) -> tuple[float, float]:
        if print_up_normal is None:
            return 0.0, 0.0

        normal = print_up_normal
        norm = np.linalg.norm(normal)
        if np.isclose(norm, 0.0):
            raise ValueError("print_up_normal must be non-zero")
        normal = normal / norm
        nx, ny, nz = normal

        horizontal = np.hypot(nx, ny)
        if np.isclose(horizontal, 0.0):
            return (0.0 if nz >= 0.0 else 180.0), 0.0

        a = np.degrees(np.arctan2(horizontal, nz))
        b = np.degrees(np.arctan2(-ny, -nx))

        b = ((b + 180.0) % 360.0) - 180.0

        return a, b
