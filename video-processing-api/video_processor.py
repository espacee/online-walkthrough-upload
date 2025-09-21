from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List


class VideoProcessor:
    """Simple FFmpeg wrapper for concatenating raw clips."""

    def __init__(self, project_id: str, raw_dir: Path, final_dir: Path) -> None:
        self.project_id = project_id
        self.raw_dir = raw_dir
        self.final_dir = final_dir
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def concatenate_clips(self, input_paths: List[str]) -> str:
        if not input_paths:
            raise ValueError("At least one clip is required for processing.")

        resolved_inputs = [self._resolve_path(path) for path in input_paths]
        output_path = self.final_dir / f"{self.project_id}_final.mp4"

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as concat_file:
            for clip_path in resolved_inputs:
                concat_file.write(f"file '{clip_path}'\n")
            concat_list_path = Path(concat_file.name)

        try:
            command = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
                str(output_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed: {result.stderr.strip() or 'unknown error'}"
                )
        finally:
            concat_list_path.unlink(missing_ok=True)

        return str(output_path)

    def _resolve_path(self, clip_name: str) -> Path:
        candidate = Path(clip_name)
        if not candidate.is_absolute():
            candidate = self.raw_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Clip not found: {candidate}")
        if self.raw_dir not in candidate.parents and candidate != self.raw_dir:
            raise ValueError(f"Clip path outside raw directory: {candidate}")
        return candidate
