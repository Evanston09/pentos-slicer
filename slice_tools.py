import subprocess
from pathlib import Path

import numpy as np
import trimesh
from viser import MeshHandle


class Slicer:
    def __init__(self, out_dir: Path = Path("temp")):
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)

    def slice_parts(
        self,
        paths: list[Path],
        config_path: Path = Path("pentos_config.ini"),
        slicer_cmd: str = "prusa-slicer",
    ) -> list[Path]:
        gcode_paths = []

        for path in paths:
            gcode_path = path.with_suffix(".gcode")
            subprocess.run(
                [
                    slicer_cmd,
                    "--export-gcode",
                    str(path),
                    "--load",
                    str(config_path),
                    "--output",
                    str(gcode_path),
                ],
                check=True,
            )
            gcode_paths.append(gcode_path)

        return gcode_paths

    def cut(self, mesh: trimesh.Trimesh, planes: list[MeshHandle]) -> list[Path]:
        pieces = [(mesh, None)]

        for plane in planes:
            next_pieces = []
            normal = self._plane_normal(plane)

            for piece, base_normal in pieces:
                part_a = piece.slice_plane(plane.position, normal, cap=True)
                part_b = piece.slice_plane(plane.position, -normal, cap=True)

                if self._is_valid(part_a):
                    next_pieces.append((part_a, -normal))
                if self._is_valid(part_b):
                    next_pieces.append((part_b, base_normal))

            pieces = next_pieces

        paths = []
        for index, (piece, base_normal) in enumerate(pieces):
            path = self.out_dir / f"{mesh.identifier_hash}_{index}.stl"
            if base_normal is not None:
                piece = self._orient_base_to_bed(piece, base_normal)

            if piece.extents.max() < 1.0:
                piece.units = "m"
                piece.convert_units("mm")
            piece.export(path)
            paths.append(path)

        return paths

    @staticmethod
    def _plane_normal(plane: MeshHandle) -> np.ndarray:
        # Only care about rotation and we start pointing up
        rotation_matrix = trimesh.transformations.quaternion_matrix(plane.wxyz)[:3, :3]
        normal = rotation_matrix @ np.array([0.0, 0.0, 1.0])
        return normal

    @staticmethod
    def _orient_base_to_bed(
        mesh: trimesh.Trimesh, base_normal: np.ndarray
    ) -> trimesh.Trimesh:
        transform = trimesh.geometry.align_vectors(base_normal, [0, 0, -1])
        mesh.apply_transform(transform)

        # Prevent floating
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
        return mesh

    @staticmethod
    def _is_valid(mesh: trimesh.Trimesh | None) -> bool:
        return mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0
