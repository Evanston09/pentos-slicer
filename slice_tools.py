import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

from machine import BUILD_PLATE_CENTER


class SlicePlane(Protocol):
    position: np.ndarray
    wxyz: np.ndarray


@dataclass
class Chunk:
    path: Path
    base_normal: np.ndarray | None
    z_offset: float


class Slicer:
    def __init__(
        self,
        out_dir: Path = Path("output"),
        temp_dir: Path = Path("temp"),
        rotation_center: np.ndarray | tuple[float, float, float] = BUILD_PLATE_CENTER,
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
        final_gcode = []

        for index, gcode_path in enumerate(gcode_paths):
            lines = gcode_path.read_text().splitlines(keepends=True)
            lines = self._trim_gcode(lines, index, len(gcode_paths))
            lines = self._offset_absolute_z(lines, chunks[index].z_offset)
            final_gcode.extend(lines)

            if index < len(gcode_paths) - 1:
                final_gcode.extend(self._ab_transition(chunks[index + 1].base_normal))

        output_path = self.out_dir / f"{source_name}.gcode"
        output_path.write_text("".join(final_gcode))
        return output_path

    def _trim_gcode(
        self,
        lines: list[str],
        index: int,
        total: int,
    ) -> list[str]:
        if total == 1:
            return lines
        if index == 0:
            return self._remove_end(lines)
        if index == total - 1:
            return self._remove_start(lines)
        return self._remove_start(self._remove_end(lines))

    @staticmethod
    def _remove_end(lines: list[str]) -> list[str]:
        custom_blocks = [
            i for i, line in enumerate(lines) if line.strip() == ";TYPE:Custom"
        ]
        return lines[: custom_blocks[-1]] if custom_blocks else lines

    @staticmethod
    def _remove_start(lines: list[str]) -> list[str]:
        start = next(
            (i for i, line in enumerate(lines) if line.strip() == ";LAYER_CHANGE"),
            0,
        )
        return lines[start:]

    def export_stl_chunks(
        self, mesh: trimesh.Trimesh, planes: list[SlicePlane], source_name: str
    ) -> list[Chunk]:
        pieces = [(mesh, None)]

        for plane in planes:
            plane_position = plane.position
            normal = self._plane_normal(plane)
            next_pieces = []

            for piece, base_normal in pieces:
                part_a = piece.slice_plane(plane_position, normal, cap=True)
                part_b = piece.slice_plane(plane_position, -normal, cap=True)

                if self._is_valid(part_a):
                    next_pieces.append((part_a, -normal))
                if self._is_valid(part_b):
                    next_pieces.append((part_b, base_normal))

            pieces = next_pieces

        chunks = []
        for index, (piece, base_normal) in enumerate(pieces):
            path = self.temp_dir / f"{source_name}_{mesh.identifier_hash}_{index}.stl"
            piece, z_offset = self._orient(piece, base_normal)
            piece.export(path)
            chunks.append(Chunk(path=path, base_normal=base_normal, z_offset=z_offset))

        return chunks

    @staticmethod
    def _plane_normal(plane: SlicePlane) -> np.ndarray:
        # Only care about rotation and we start pointing up
        rotation_matrix = trimesh.transformations.quaternion_matrix(plane.wxyz)[:3, :3]
        normal = rotation_matrix @ np.array([0.0, 0.0, 1.0])
        return normal

    def _orient(
        self, mesh: trimesh.Trimesh, base_normal: np.ndarray | None
    ) -> tuple[trimesh.Trimesh, float]:
        if base_normal is not None:
            transform = trimesh.geometry.align_vectors(base_normal, [0, 0, -1])
            transform = trimesh.transformations.transform_around(
                transform,
                self.rotation_center,
            )
            mesh.apply_transform(transform)

        z_offset = float(mesh.bounds[0][2])
        mesh.apply_translation([0, 0, -z_offset])
        return mesh, z_offset

    @staticmethod
    def _is_valid(mesh: trimesh.Trimesh | None) -> bool:
        return mesh is not None and len(mesh.vertices) > 0 and len(mesh.faces) > 0

    @staticmethod
    def _offset_absolute_z(lines: list[str], z_offset: float) -> list[str]:
        absolute_xyz = True
        offset_lines = []

        for line in lines:
            newline = "\n" if line.endswith("\n") else ""
            content = line[:-1] if newline else line
            command_part, separator, comment = content.partition(";")
            tokens = command_part.split()

            if not tokens:
                offset_lines.append(line)
                continue

            command = tokens[0].upper()
            if command == "G90":
                absolute_xyz = True
            elif command == "G91":
                absolute_xyz = False
            elif absolute_xyz and command in {"G0", "G00", "G1", "G01"}:
                for token_index, token in enumerate(tokens[1:], start=1):
                    if token[:1].upper() == "Z":
                        z = float(token[1:]) + z_offset
                        tokens[token_index] = f"Z{z:.3f}".rstrip("0").rstrip(".")
                        break

            rebuilt = " ".join(tokens)
            if separator:
                rebuilt = f"{rebuilt} {separator}{comment}" if rebuilt else content
            offset_lines.append(f"{rebuilt}{newline}")

        return offset_lines

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
