import subprocess
from pathlib import Path

import numpy as np
import trimesh
from viser import MeshHandle


class Slicer:
    def __init__(
        self, out_dir: Path = Path("output"), temp_dir: Path = Path("temp")
    ) -> None:
        self.out_dir = out_dir
        self.temp_dir = temp_dir
        self.out_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def slice(
        self, mesh: trimesh.Trimesh, planes: list[MeshHandle], source_name: str = "model"
    ) -> Path:
        chunks = self.export_stl_chunks(mesh, planes, source_name)
        if not chunks:
            raise ValueError("No chunks were generated")

        stl_paths, base_normals = map(list, zip(*chunks))
        gcode_paths = self.run_prusa_slicer(stl_paths)
        return self.merge_gcode_files(gcode_paths, base_normals, source_name)

    def run_prusa_slicer(
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

    def merge_gcode_files(
        self,
        gcode_paths: list[Path],
        base_normals: list[np.ndarray | None],
        source_name: str,
    ) -> Path:
        final_gcode = []

        for index, gcode_path in enumerate(gcode_paths):
            lines = gcode_path.read_text().splitlines(keepends=True)

            if len(gcode_paths) == 1:
                pass
            elif index == 0:
                lines = self._remove_end(lines)
            elif index == len(gcode_paths) - 1:
                lines = self._remove_start(lines)
            else:
                lines = self._remove_start(self._remove_end(lines))

            final_gcode.extend(lines)

            if index < len(gcode_paths) - 1:
                final_gcode.extend(self._ab_transition(base_normals[index + 1]))

        output_path = self.out_dir / f"{source_name}.gcode"
        output_path.write_text("".join(final_gcode))
        return output_path

    @staticmethod
    def _remove_end(lines: list[str]) -> list[str]:
        end_start = max(
            i for i, line in enumerate(lines) if line.strip() == ";TYPE:Custom"
        )
        return lines[:end_start]

    @staticmethod
    def _remove_start(lines: list[str]) -> list[str]:
        start = next(
            i for i, line in enumerate(lines) if line.strip() == ";LAYER_CHANGE"
        )

        return lines[start:]

    def export_stl_chunks(
        self, mesh: trimesh.Trimesh, planes: list[MeshHandle], source_name: str
    ) -> list[tuple[Path, np.ndarray | None]]:
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

        chunks = []
        for index, (piece, base_normal) in enumerate(pieces):
            path = self.temp_dir / f"{source_name}_{mesh.identifier_hash}_{index}.stl"
            if base_normal is not None:
                piece = self._orient_base_to_bed(piece, base_normal)

            if piece.extents.max() < 1.0:
                piece.units = "m"
                piece.convert_units("mm")
            piece.export(path)
            chunks.append((path, base_normal))

        return chunks

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

    def _ab_transition(self, base_normal: np.ndarray | None) -> list[str]:
        if base_normal is None:
            a = 0.0
            b = 0.0
        else:
            normal = -base_normal
            normal = normal / np.linalg.norm(normal)
            nx, ny, nz = normal

            b = np.degrees(np.arctan2(ny, nx))
            a = np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))

        return [
            "\n; --- PENTOS A/B TRANSITION ---\n",
            "G1 E-5 F3600\n",
            "G91 ; relative movement for safe lift\n",
            "G1 Z10 F3000\n",
            "G90 ; absolute movement\n",
            f"G1 A{a:.3f} B{b:.3f} F1200\n",
            "G92 E0\n",
            "; --- END PENTOS A/B TRANSITION ---\n",
        ]
