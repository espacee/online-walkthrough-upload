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

        output_path = self.final_dir / f"{self.project_id}_final.mp4"
        logger.info(
            "Running two-pass encode for project %s into %s",
            self.project_id,
            output_path.name,
        )
        passlog = working_dir / f"{self.project_id}_2pass"
        filter_graph, video_label, audio_label, combined_duration = self._build_assembly_filter(
            processed_clips
        )
        input_args: list[str] = []
        for clip in processed_clips:
            input_args.extend(["-i", str(clip.path)])
        logger.debug(
            "Assembly filter graph for project %s: %s",
            self.project_id,
            filter_graph,
        )
        self._run_two_pass_encode(
            input_args,
            filter_graph,
            video_label,
            audio_label,
            output_path,
            passlog,
        )
        self._cleanup_two_pass_logs(passlog)

        expected_duration = combined_duration
        self._validate_final_output(output_path, expected_duration)
        return output_path

    def _run_two_pass_encode(
        self,
        input_args: list[str],
        filter_graph: str,
        video_label: str,
        audio_label: str,
        output_path: Path,
        passlog: Path,
    ) -> None:
        gop = TARGET_FPS * 2
        target_bitrate = "8M"
        max_bitrate = "12M"
        bufsize = "24M"
        common = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
        ]
        first_pass = common + input_args + [
            "-filter_complex",
            filter_graph,
            "-map",
            video_label,
            "-map",
            audio_label,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-b:v",
            target_bitrate,
            "-maxrate",
            max_bitrate,
            "-bufsize",
            bufsize,
            "-pass",
            "1",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-pix_fmt",
            TARGET_PIXEL_FORMAT,
            "-passlogfile",
            str(passlog),
            "-f",
            "mp4",
            "/dev/null",
        ]
        self._run_ffmpeg(first_pass, "two-pass encode pass1")

        second_pass = common + input_args + [
            "-filter_complex",
            filter_graph,
            "-map",
            video_label,
            "-map",
            audio_label,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-b:v",
            target_bitrate,
            "-maxrate",
            max_bitrate,
            "-bufsize",
            bufsize,
            "-pass",
            "2",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
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

    def _build_assembly_filter(
        self, processed_clips: Sequence[ProcessedClip]
    ) -> tuple[str, str, str, float]:
        if not processed_clips:
            raise ValueError("At least one processed clip is required for assembly.")

        video_ops = (
            f"settb=AVTB,fps={TARGET_FPS},format={TARGET_PIXEL_FORMAT},setsar=1"
        )
        audio_ops = (
            "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,aresample=48000"
        )

        graph_parts: list[str] = []
        clip_infos: list[dict[str, Optional[str]]] = []
        eps = 1e-3

        def add_trim(
            source: str,
            start: float,
            duration: float,
            target: str,
            *,
            audio: bool = False,
        ) -> bool:
            if duration <= eps:
                return False
            start = max(start, 0.0)
            if audio:
                graph_parts.append(
                    f"[{source}]atrim=start={start:.6f}:duration={duration:.6f},asetpts=PTS-STARTPTS[{target}]"
                )
            else:
                graph_parts.append(
                    f"[{source}]trim=start={start:.6f}:duration={duration:.6f},setpts=PTS-STARTPTS,settb=AVTB[{target}]"
                )
            return True

        for index, clip in enumerate(processed_clips):
            video_norm = f"vnorm{index}"
            audio_norm = f"anorm{index}"
            graph_parts.append(f"[{index}:v]{video_ops}[{video_norm}]")
            graph_parts.append(f"[{index}:a]{audio_ops}[{audio_norm}]")
            clip_infos.append(
                {
                    "video_norm": video_norm,
                    "audio_norm": audio_norm,
                    "head_v": None,
                    "head_a": None,
                    "body_v": None,
                    "body_a": None,
                    "tail_v": None,
                    "tail_a": None,
                    "fade_in_v": None,
                    "fade_in_a": None,
                    "fade_out_v": None,
                    "fade_out_a": None,
                    "body_duration": 0.0,
                    "head_duration": 0.0,
                    "tail_duration": 0.0,
                }
            )

        if len(processed_clips) == 1:
            clip = processed_clips[0]
            info = clip_infos[0]
            segment_v = "vseg0"
            segment_a = "aseg0"
            if not add_trim(info["video_norm"], 0.0, clip.duration, segment_v):
                raise ValueError("Single clip has non-positive duration after trimming.")
            add_trim(info["audio_norm"], 0.0, clip.duration, segment_a, audio=True)
            graph_parts.append(f"[{segment_v}][{segment_a}]concat=n=1:v=1:a=1[vout][aout]")
            logger.debug("Assembly filter graph created | clips=1 fades=0.000")
            return ";".join(graph_parts), "[vout]", "[aout]", clip.duration

        pair_fades: list[float] = []
        for idx in range(len(processed_clips) - 1):
            current = processed_clips[idx]
            nxt = processed_clips[idx + 1]
            fade = min(
                CROSSFADE_DURATION,
                current.duration / 2.0,
                nxt.duration / 2.0,
            )
            if fade < eps:
                fade = 0.0
            pair_fades.append(fade)

        segments: list[tuple[str, str, float]] = []
        total_duration = 0.0

        # First clip head and fade-out
        first_clip = processed_clips[0]
        first_info = clip_infos[0]
        first_fade = pair_fades[0]
        head_duration = max(first_clip.duration - first_fade, 0.0)
        head_label_v = "vhead0"
        head_label_a = "ahead0"
        if add_trim(first_info["video_norm"], 0.0, head_duration, head_label_v):
            add_trim(first_info["audio_norm"], 0.0, head_duration, head_label_a, audio=True)
            segments.append((head_label_v, head_label_a, head_duration))
            total_duration += head_duration
            first_info["head_v"] = head_label_v
            first_info["head_a"] = head_label_a
            first_info["head_duration"] = head_duration
        if first_fade > eps:
            fade_start = max(first_clip.duration - first_fade, 0.0)
            fade_v = "vfade0_out"
            fade_a = "afade0_out"
            if add_trim(first_info["video_norm"], fade_start, first_fade, fade_v):
                add_trim(first_info["audio_norm"], fade_start, first_fade, fade_a, audio=True)
                first_info["fade_out_v"] = fade_v
                first_info["fade_out_a"] = fade_a

        # Middle clips
        for idx in range(1, len(processed_clips) - 1):
            clip = processed_clips[idx]
            info = clip_infos[idx]
            fade_in = pair_fades[idx - 1]
            fade_out = pair_fades[idx]
            if fade_in > eps:
                fade_v = f"vfade{idx}_in"
                fade_a = f"afade{idx}_in"
                if add_trim(info["video_norm"], 0.0, fade_in, fade_v):
                    add_trim(info["audio_norm"], 0.0, fade_in, fade_a, audio=True)
                    info["fade_in_v"] = fade_v
                    info["fade_in_a"] = fade_a
            body_start = max(fade_in, 0.0)
            body_end = max(clip.duration - fade_out, body_start)
            body_duration = max(body_end - body_start, 0.0)
            if body_duration > eps:
                body_v = f"vbody{idx}"
                body_a = f"abody{idx}"
                if add_trim(info["video_norm"], body_start, body_duration, body_v):
                    add_trim(info["audio_norm"], body_start, body_duration, body_a, audio=True)
                    info["body_v"] = body_v
                    info["body_a"] = body_a
                    info["body_duration"] = body_duration
            if fade_out > eps:
                fade_start = max(clip.duration - fade_out, 0.0)
                fade_v = f"vfade{idx}_out"
                fade_a = f"afade{idx}_out"
                if add_trim(info["video_norm"], fade_start, fade_out, fade_v):
                    add_trim(info["audio_norm"], fade_start, fade_out, fade_a, audio=True)
                    info["fade_out_v"] = fade_v
                    info["fade_out_a"] = fade_a

        # Last clip fade-in and tail
        last_idx = len(processed_clips) - 1
        last_clip = processed_clips[last_idx]
        last_info = clip_infos[last_idx]
        last_fade = pair_fades[-1]
        if last_fade > eps:
            fade_v = f"vfade{last_idx}_in"
            fade_a = f"afade{last_idx}_in"
            if add_trim(last_info["video_norm"], 0.0, last_fade, fade_v):
                add_trim(last_info["audio_norm"], 0.0, last_fade, fade_a, audio=True)
                last_info["fade_in_v"] = fade_v
                last_info["fade_in_a"] = fade_a
        tail_start = max(last_fade, 0.0)
        tail_duration = max(last_clip.duration - tail_start, 0.0)
        tail_v = f"vtail{last_idx}"
        tail_a = f"atail{last_idx}"
        if add_trim(last_info["video_norm"], tail_start, tail_duration, tail_v):
            add_trim(last_info["audio_norm"], tail_start, tail_duration, tail_a, audio=True)
            last_info["tail_v"] = tail_v
            last_info["tail_a"] = tail_a
            last_info["tail_duration"] = tail_duration

        # Assemble segments with crossfades
        for pair_idx, fade in enumerate(pair_fades):
            prev_info = clip_infos[pair_idx]
            next_info = clip_infos[pair_idx + 1]
            if fade > eps and prev_info.get("fade_out_v") and next_info.get("fade_in_v"):
                vxf = f"vxf{pair_idx}"
                axf = f"axf{pair_idx}"
                graph_parts.append(
                    f"[{prev_info['fade_out_v']}][{next_info['fade_in_v']}]xfade=transition=fade:duration={fade:.6f}:offset=0[{vxf}]"
                )
                graph_parts.append(
                    f"[{prev_info['fade_out_a']}][{next_info['fade_in_a']}]acrossfade=d={fade:.6f}:curve1=tri:curve2=tri[{axf}]"
                )
                segments.append((vxf, axf, fade))
                total_duration += fade
            else:
                logger.warning(
                    "Crossfade skipped for pair %d due to insufficient duration; performing hard cut",
                    pair_idx,
                )
            body_v = next_info.get("body_v")
            body_a = next_info.get("body_a")
            body_duration = next_info.get("body_duration", 0.0)
            if body_v and body_duration > eps:
                segments.append((body_v, body_a, body_duration))
                total_duration += body_duration

        # Append final tail
        if last_info.get("tail_v") and last_info.get("tail_duration", 0.0) > eps:
            segments.append((last_info["tail_v"], last_info["tail_a"], last_info["tail_duration"]))
            total_duration += last_info["tail_duration"]

        if not segments:
            raise ValueError("No segments generated for concat pipeline.")

        concat_inputs = ''.join(f"[{v}][{a}]" for v, a, _ in segments)
        graph_parts.append(
            f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vout][aout]"
        )

        fade_total = sum(f for f in pair_fades)
        logger.debug(
            "Assembly filter graph created | clips=%d fades=%.3fs",
            len(processed_clips),
            fade_total,
        )
        return ";".join(graph_parts), "[vout]", "[aout]", total_duration

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
                "-g",
                str(TARGET_FPS * 2),
                "-keyint_min",
                str(TARGET_FPS * 2),
                "-sc_threshold",
                "0",
                "-pix_fmt",
                TARGET_PIXEL_FORMAT,
                "-movflags",
                "+faststart",
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
