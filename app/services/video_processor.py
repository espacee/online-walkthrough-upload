from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


logger = logging.getLogger(__name__)


class VideoProcessor:
    """FFmpeg-based walkthrough processor with per-clip stabilization."""

    def __init__(self, project_id: str, raw_dir: Path, final_dir: Path) -> None:
        self.project_id = project_id
        self.raw_dir = raw_dir
        self.final_dir = final_dir
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def process_walkthrough(self, clip_names: Iterable[str]) -> Path:
        clip_list = list(clip_names)
        if not clip_list:
            raise ValueError("At least one clip is required for processing.")

        resolved_inputs = [self._resolve_path(name) for name in clip_list]
        logger.info(
            "Resolved %d input clips for project %s: %s",
            len(resolved_inputs),
            self.project_id,
            [clip.name for clip in resolved_inputs],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(
                "Temporary working directory created for project %s: %s",
                self.project_id,
                temp_path,
            )
            stabilized_paths = []
            clip_total = len(resolved_inputs)
            for position, clip_path in enumerate(resolved_inputs, start=1):
                logger.info(
                    "Stabilizing clip %d/%d for project %s: %s",
                    position,
                    clip_total,
                    self.project_id,
                    clip_path.name,
                )
                zero_index = position - 1
                stabilized_path = temp_path / f"stabilized_{zero_index:03d}.mp4"
                self._stabilize_clip(clip_path, stabilized_path, zero_index, temp_path)
                stabilized_paths.append(stabilized_path)
                logger.debug(
                    "Stabilization complete for project %s clip %s -> %s",
                    self.project_id,
                    clip_path.name,
                    stabilized_path.name,
                )

            output_path = self._concat_paths(stabilized_paths)

        logger.debug(
            "Processing routine finished for project %s; output located at %s",
            self.project_id,
            output_path,
        )

        return output_path

    def concatenate_clips(self, clip_names: Iterable[str]) -> Path:
        """Fallback helper retained for backwards compatibility."""

        resolved_inputs = [self._resolve_path(name) for name in clip_names]
        return self._concat_paths(resolved_inputs)

    def _stabilize_clip(
        self,
        input_path: Path,
        output_path: Path,
        index: int,
        working_dir: Path,
    ) -> None:
        transform_path = working_dir / f"transform_{index:03d}.trf"

        detect_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"vidstabdetect=result={transform_path}",
            "-f",
            "null",
            "-",
        ]

        self._run_ffmpeg(detect_cmd, f"vidstabdetect for {input_path.name}")

        transform_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            f"vidstabtransform=input={transform_path}:smoothing=30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "copy",
            str(output_path),
        ]

        try:
            self._run_ffmpeg(transform_cmd, f"vidstabtransform for {input_path.name}")
        finally:
            transform_path.unlink(missing_ok=True)

    def _concat_paths(self, clip_paths: Sequence[Path]) -> Path:
        if not clip_paths:
            raise ValueError("At least one clip is required for processing.")

        output_path = self.final_dir / f"{self.project_id}_final.mp4"
        logger.info(
            "Concatenating %d stabilized clips for project %s into %s",
            len(clip_paths),
            self.project_id,
            output_path.name,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as concat_file:
            for clip_path in clip_paths:
                concat_file.write(f"file '{clip_path}'\n")
            concat_list_path = Path(concat_file.name)

        try:
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
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
            self._run_ffmpeg(concat_cmd, "concat stabilized clips")
        finally:
            concat_list_path.unlink(missing_ok=True)

        logger.info(
            "Concatenation complete for project %s: %s",
            self.project_id,
            output_path,
        )

        return output_path

    def _run_ffmpeg(self, command: list[str], description: str) -> None:
        quoted = " ".join(shlex.quote(str(part)) for part in command)
        logger.debug("Running FFmpeg command (%s): %s", description, quoted)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip()
            logger.error("FFmpeg command failed (%s): %s", description, error_output)
            raise RuntimeError(
                f"FFmpeg failed during {description}: {result.stderr.strip() or 'unknown error'}"
            )

    def _resolve_path(self, clip_name: str) -> Path:
        candidate = Path(clip_name)
        if not candidate.is_absolute():
            candidate = self.raw_dir / candidate
        candidate = candidate.resolve()
        raw_root = self.raw_dir.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Clip not found: {candidate.name}")
        if raw_root not in candidate.parents and candidate != raw_root:
            raise ValueError(f"Clip path outside raw directory: {candidate.name}")
        return candidate
