from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


logger = logging.getLogger(__name__)


CROSSFADE_DURATION = 0.5
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30
TARGET_PIXEL_FORMAT = "yuv420p"
DEFAULT_MAX_RETRIES = 1


@dataclass
class ProcessedClip:
    """Container for a fully processed clip ready for assembly."""

    path: Path
    duration: float
    frame_count: int


@dataclass
class TrimInfo:
    """Trim timings derived from freeze detection analysis."""

    start: float
    end: float

    @property
    def total_trim(self) -> float:
        return self.start + self.end



class VideoProcessor:
    """FFmpeg-based walkthrough processor with per-clip stabilization."""

    def __init__(
        self,
        project_id: str,
        raw_dir: Path,
        final_dir: Path,
        *,
        audio_enhancements: bool = True,
    ) -> None:
        self.project_id = project_id
        self.raw_dir = raw_dir
        self.final_dir = final_dir
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.audio_enhancements = audio_enhancements
        self.ffmpeg_retries = DEFAULT_MAX_RETRIES

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

            processed_clips: List[ProcessedClip] = []
            clip_total = len(resolved_inputs)
            for position, clip_path in enumerate(resolved_inputs, start=1):
                logger.info(
                    "Processing clip %d/%d for project %s: %s",
                    position,
                    clip_total,
                    self.project_id,
                    clip_path.name,
                )
                processed = self._process_clip(clip_path, position - 1, temp_path)
                processed_clips.append(processed)
                logger.debug(
                    "Clip %s processed for project %s; duration=%.3fs frames=%d",
                    clip_path.name,
                    self.project_id,
                    processed.duration,
                    processed.frame_count,
                )

            output_path = self._assemble_final_video(processed_clips, temp_path)

        logger.debug(
            "Processing routine finished for project %s; output located at %s",
            self.project_id,
            output_path,
        )

        return output_path

    def concatenate_clips(self, clip_names: Iterable[str]) -> Path:
        """Fallback helper retained for backwards compatibility."""

        resolved_inputs = [self._resolve_path(name) for name in clip_names]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(
                "Temporary directory for concat fallback (project %s): %s",
                self.project_id,
                temp_path,
            )
            processed_clips = [
                self._process_clip(clip_path, index, temp_path)
                for index, clip_path in enumerate(resolved_inputs)
            ]
            return self._assemble_final_video(processed_clips, temp_path)

    def _process_clip(self, input_path: Path, index: int, working_dir: Path) -> ProcessedClip:
        total_duration = self._probe_duration(input_path)
        if total_duration <= 0:
            raise ValueError(f"Clip has invalid duration: {input_path.name}")

        trim_info = self._detect_trim(input_path, total_duration)
        trimmed_duration = max(total_duration - trim_info.total_trim, 0.05)
        if trimmed_duration <= 0.05:
            logger.warning(
                "Trimmed duration for %s is extremely low (%.3fs); falling back to full clip",
                input_path.name,
                trimmed_duration,
            )
            trim_info = TrimInfo(start=0.0, end=0.0)
            trimmed_duration = total_duration

        logger.debug(
            "Trim info for %s: start=%.3fs end=%.3fs total=%.3fs (original=%.3fs)",
            input_path.name,
            trim_info.start,
            trim_info.end,
            trimmed_duration,
            total_duration,
        )

        transform_path = working_dir / f"transform_{index:03d}.trf"
        stabilized_path = working_dir / f"stabilized_{index:03d}.mp4"
        processed_path = working_dir / f"processed_{index:03d}.mp4"
        audio_path = working_dir / f"audio_{index:03d}.m4a"

        try:
            self._extract_audio(input_path, audio_path, trim_info, trimmed_duration)
            self._stabilize_and_color(
                input_path,
                stabilized_path,
                transform_path,
                trim_info,
                trimmed_duration,
            )
            self._remux_audio(stabilized_path, audio_path, processed_path)
        finally:
            transform_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            stabilized_path.unlink(missing_ok=True)

        duration = self._probe_duration(processed_path)
        frames = self._probe_frame_count(processed_path)
        self._validate_clip_duration(processed_path, trimmed_duration, duration)

        return ProcessedClip(path=processed_path, duration=duration, frame_count=frames)

    def _assemble_final_video(
        self, processed_clips: Sequence[ProcessedClip], working_dir: Path
    ) -> Path:
        if not processed_clips:
            raise ValueError("At least one processed clip is required for assembly.")

        if len(processed_clips) == 1:
            source_path = processed_clips[0].path
        else:
            source_path = self._crossfade_clips(processed_clips, working_dir)

        output_path = self.final_dir / f"{self.project_id}_final.mp4"
        logger.info(
            "Running two-pass encode for project %s into %s",
            self.project_id,
            output_path.name,
        )
        passlog = working_dir / f"{self.project_id}_2pass"
        self._run_two_pass_encode(source_path, output_path, passlog)
        self._cleanup_two_pass_logs(passlog)

        expected_duration = self._expected_final_duration(processed_clips)
        self._validate_final_output(output_path, expected_duration)
        return output_path

    def _crossfade_clips(
        self, processed_clips: Sequence[ProcessedClip], working_dir: Path
    ) -> Path:
        current_path = processed_clips[0].path
        current_duration = processed_clips[0].duration
        current_frames = processed_clips[0].frame_count

        for index, clip in enumerate(processed_clips[1:], start=1):
            crossfade_duration = min(CROSSFADE_DURATION, current_duration / 2, clip.duration / 2)
            if crossfade_duration <= 0:
                crossfade_duration = min(CROSSFADE_DURATION, current_duration, clip.duration)
            crossfade_offset = max(current_duration - crossfade_duration, 0)
            output_path = working_dir / f"crossfade_{index:03d}.mp4"
            filter_complex = (
                f"[0:v]format={TARGET_PIXEL_FORMAT},setsar=1[v0];"
                f"[1:v]format={TARGET_PIXEL_FORMAT},setsar=1[v1];"
                f"[v0][v1]xfade=transition=fade:duration={crossfade_duration:.3f}:offset={crossfade_offset:.3f}[vout];"
                f"[0:a][1:a]acrossfade=d={crossfade_duration:.3f}:curve1=tri:curve2=tri[aout]"
            )

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(current_path),
                "-i",
                str(clip.path),
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "16",
                "-pix_fmt",
                TARGET_PIXEL_FORMAT,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            self._run_ffmpeg(command, f"crossfade clips step {index}")
            if current_path not in {processed_clips[0].path, clip.path}:
                self._cleanup_file(current_path)
            current_path = output_path
            current_duration = self._probe_duration(current_path)
            current_frames = self._probe_frame_count(current_path)
            logger.debug(
                "Crossfade step %d duration %.3fs frames %d",
                index,
                current_duration,
                current_frames,
            )
            if clip.path != current_path:
                self._cleanup_file(clip.path)

        first_clip_path = processed_clips[0].path
        if first_clip_path != current_path:
            self._cleanup_file(first_clip_path)

        return current_path

    def _run_two_pass_encode(self, source_path: Path, output_path: Path, passlog: Path) -> None:
        gop = TARGET_FPS * 2
        first_pass = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "19",
            "-pass",
            "1",
            "-b:v",
            "0",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-pix_fmt",
            TARGET_PIXEL_FORMAT,
            "-passlogfile",
            str(passlog),
            "-an",
            "-f",
            "mp4",
            "/dev/null",
        ]
        self._run_ffmpeg(first_pass, "two-pass encode pass1")

        second_pass = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "19",
            "-pass",
            "2",
            "-b:v",
            "0",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-pix_fmt",
            TARGET_PIXEL_FORMAT,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-passlogfile",
            str(passlog),
            str(output_path),
        ]
        self._run_ffmpeg(second_pass, "two-pass encode pass2")

    def _cleanup_two_pass_logs(self, passlog: Path) -> None:
        base = Path(str(passlog))
        candidates = [
            base,
            Path(f"{base}-0.log"),
            Path(f"{base}-0.log.mbtree"),
            base.with_suffix(".log"),
            base.with_suffix(".log.mbtree"),
        ]
        for candidate in candidates:
            self._cleanup_file(candidate)

    def _expected_final_duration(self, clips: Sequence[ProcessedClip]) -> float:
        if not clips:
            return 0.0
        total = sum(clip.duration for clip in clips)
        crossfade_total = CROSSFADE_DURATION * max(len(clips) - 1, 0)
        return max(total - crossfade_total, 0.0)

    def _detect_trim(self, input_path: Path, total_duration: float) -> TrimInfo:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(input_path),
            "-vf",
            "freezedetect=n=-60dB:d=0.7",
            "-f",
            "null",
            "-",
        ]
        result = self._run_ffmpeg(command, f"freeze analysis for {input_path.name}", retries=0)
        intervals: List[tuple[float, float]] = []
        current_start: Optional[float] = None
        for line in result.stderr.splitlines():
            if "freeze_start" in line:
                try:
                    current_start = float(line.rsplit("freeze_start:", 1)[1])
                except ValueError:
                    current_start = None
            elif "freeze_end" in line and current_start is not None:
                try:
                    end_time = float(line.rsplit("freeze_end:", 1)[1])
                except ValueError:
                    current_start = None
                    continue
                intervals.append((current_start, end_time))
                current_start = None
            elif "freeze_duration" in line and current_start is not None:
                try:
                    duration = float(line.rsplit("freeze_duration:", 1)[1])
                    intervals.append((current_start, current_start + duration))
                except ValueError:
                    pass
                finally:
                    current_start = None

        start_trim = 0.0
        end_trim = 0.0
        for start, end in intervals:
            if start <= 1.0:
                start_trim = max(start_trim, min(end, total_duration))
            if end >= total_duration - 1.0:
                end_trim = max(end_trim, max(total_duration - start, 0.0))

        start_trim = min(start_trim, total_duration * 0.5)
        end_trim = min(end_trim, total_duration - start_trim)
        return TrimInfo(start=start_trim, end=end_trim)

    def _extract_audio(
        self,
        input_path: Path,
        output_path: Path,
        trim_info: TrimInfo,
        trimmed_duration: float,
    ) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
        ]
        if trim_info.start > 0:
            command.extend(["-ss", f"{trim_info.start:.3f}"])
        command.extend(["-t", f"{trimmed_duration:.3f}"])
        audio_filters: list[str] = []
        if getattr(self, "audio_enhancements", True):
            audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            audio_filters.append("afftdn=nf=-25")
        if audio_filters:
            command.extend(["-af", ",".join(audio_filters)])
        command.extend(
            [
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                str(output_path),
            ]
        )
        self._run_ffmpeg(command, f"audio extraction for {input_path.name}")

    def _stabilize_and_color(
        self,
        input_path: Path,
        output_path: Path,
        transform_path: Path,
        trim_info: TrimInfo,
        trimmed_duration: float,
    ) -> None:
        detect_filter = self._build_detect_filter(transform_path)
        detect_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(input_path),
        ]
        if trim_info.start > 0:
            detect_cmd.extend(["-ss", f"{trim_info.start:.3f}"])
        detect_cmd.extend(["-t", f"{trimmed_duration:.3f}"])
        detect_cmd.extend(
            [
                "-vf",
                detect_filter,
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
        self._run_ffmpeg(detect_cmd, f"vidstabdetect for {input_path.name}")

        transform_filter = self._build_transform_filter(transform_path)
        transform_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
        ]
        if trim_info.start > 0:
            transform_cmd.extend(["-ss", f"{trim_info.start:.3f}"])
        transform_cmd.extend(["-t", f"{trimmed_duration:.3f}"])
        transform_cmd.extend(
            [
                "-vf",
                transform_filter,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                TARGET_PIXEL_FORMAT,
                str(output_path),
            ]
        )
        self._run_ffmpeg(transform_cmd, f"vidstabtransform for {input_path.name}")

    def _build_detect_filter(self, transform_path: Path) -> str:
        return f"vidstabdetect=result={transform_path}:stepsize=4:shakiness=6:accuracy=15"

    def _build_transform_filter(self, transform_path: Path) -> str:
        components = [
            (
                "vidstabtransform="
                f"input={transform_path}:smoothing=15:maxshift=30:zoom=0.95:optalgo=gauss"
            ),
            self._normalization_filter_chain(include_format=False),
            self._color_filter_chain(),
            f"format={TARGET_PIXEL_FORMAT}",
            "setsar=1",
        ]
        return ",".join(components)

    def _normalization_filter_chain(self, *, include_format: bool = True) -> str:
        parts = [
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase",
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}",
            f"fps={TARGET_FPS}",
        ]
        if include_format:
            parts.extend([f"format={TARGET_PIXEL_FORMAT}", "setsar=1"])
        return ",".join(parts)

    def _color_filter_chain(self) -> str:
        return ",".join(
            [
                "histeq=strength=0.6:intensity=0.0",
                "eq=contrast=1.08:brightness=0.03:saturation=1.12",
            ]
        )

    def _remux_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        self._run_ffmpeg(command, f"remux audio for {video_path.name}")

    def _probe_duration(self, path: Path) -> float:
        result = self._run_ffprobe(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            f"probe duration for {path.name}",
        )
        try:
            return float(result["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            return 0.0

    def _probe_frame_count(self, path: Path) -> int:
        result = self._run_ffprobe(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "json",
                str(path),
            ],
            f"probe frames for {path.name}",
        )
        try:
            return int(result["streams"][0]["nb_read_frames"])
        except (KeyError, ValueError, TypeError, IndexError):
            return 0

    def _run_ffprobe(self, command: list[str], description: str) -> dict:
        quoted = " ".join(shlex.quote(str(part)) for part in command)
        logger.debug("Running FFprobe command (%s): %s", description, quoted)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            error_output = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            logger.error("FFprobe command failed (%s): %s", description, error_output)
            raise RuntimeError(f"FFprobe failed during {description}: {error_output}")
        if completed.stderr.strip():
            logger.debug("FFprobe stderr (%s): %s", description, completed.stderr.strip())
        try:
            return json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse FFprobe output (%s): %s", description, exc)
            return {}

    def _validate_clip_duration(self, clip_path: Path, expected: float, actual: float) -> None:
        if expected <= 0 or actual <= 0:
            return
        delta = abs(expected - actual)
        if delta > 0.15:
            logger.warning(
                "Duration mismatch for %s: expected %.3fs actual %.3fs",
                clip_path.name,
                expected,
                actual,
            )

    def _validate_final_output(self, output_path: Path, expected_duration: float) -> None:
        actual_duration = self._probe_duration(output_path)
        actual_frames = self._probe_frame_count(output_path)
        delta = abs(expected_duration - actual_duration)
        if delta > 0.25:
            logger.warning(
                "Final duration mismatch for %s: expected %.3fs actual %.3fs",
                output_path.name,
                expected_duration,
                actual_duration,
            )
        logger.info(
            "Final output ready %s | duration=%.3fs frames=%d",
            output_path,
            actual_duration,
            actual_frames,
        )

    def _cleanup_file(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.debug("Failed to delete %s: %s", path, exc)

    def _run_ffmpeg(
        self,
        command: list[str],
        description: str,
        *,
        retries: Optional[int] = None,
    ) -> subprocess.CompletedProcess[str]:
        if retries is None:
            retries = self.ffmpeg_retries
        quoted = " ".join(shlex.quote(str(part)) for part in command)
        attempt = 0
        last_error: Optional[str] = None
        while attempt <= retries:
            logger.debug(
                "Running FFmpeg command (%s) attempt %d/%d: %s",
                description,
                attempt + 1,
                retries + 1,
                quoted,
            )
            result = subprocess.run(command, capture_output=True, text=True)
            stderr_output = result.stderr.strip()
            if stderr_output:
                logger.debug("FFmpeg stderr (%s): %s", description, stderr_output)
            if result.returncode == 0:
                stdout_output = result.stdout.strip()
                if stdout_output:
                    logger.debug("FFmpeg stdout (%s): %s", description, stdout_output)
                return result

            last_error = stderr_output or result.stdout.strip() or "unknown error"
            logger.warning(
                "FFmpeg command failed (%s) attempt %d/%d: %s",
                description,
                attempt + 1,
                retries + 1,
                last_error,
            )
            attempt += 1

        logger.error(
            "FFmpeg command exhausted retries (%s): %s",
            description,
            last_error or "unknown error",
        )
        raise RuntimeError(f"FFmpeg failed during {description}: {last_error or 'unknown error'}")

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
