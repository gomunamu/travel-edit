"""ffmpeg 기반 클립 렌더링 및 병합"""
import subprocess
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple, Optional

from config import OUTPUT_RESOLUTION, CRF, FFMPEG_PRESET

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_day_resolution(clips: List[dict]) -> Tuple[int, int]:
    """config.OUTPUT_RESOLUTION 반환 (설정값 우선)"""
    return OUTPUT_RESOLUTION


def build_scale_filter(is_portrait: bool, out_w: int, out_h: int) -> str:
    """
    세로 영상: 높이 맞추고 좌우 블랙 패딩 (필러박스)
    가로 영상: 해상도 맞추고 필요시 레터박스
    """
    if is_portrait:
        return (
            f"scale=-2:{out_h}:flags=lanczos,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:0:black,"
            f"setsar=1"
        )
    else:
        return (
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )


def render_clip(
    clip: dict,
    output_path: str,
    out_res: Tuple[int, int],
    subtitle_path: Optional[str] = None,
    location_path: Optional[str] = None,
) -> bool:
    """
    단일 클립을 렌더링.
    - trim_start/trim_end 적용
    - 세로/가로 처리
    - 자막 번인
    - 장소 오버레이 번인
    """
    filepath = clip["filepath"]
    trim_start = clip.get("trim_start", 0.0)
    trim_end = clip.get("trim_end", clip.get("duration", 0.0))
    seg_duration = trim_end - trim_start
    is_portrait = clip.get("is_portrait", False)
    out_w, out_h = out_res

    if seg_duration <= 0.1:
        return False

    # 비디오 필터 체인 구성
    filters = []
    filters.append(build_scale_filter(is_portrait, out_w, out_h))

    # 장소 오버레이 (먼저)
    if location_path and Path(location_path).exists():
        esc = _esc_path(location_path)
        filters.append(f"ass='{esc}'")

    # 자막 (나중)
    if subtitle_path and Path(subtitle_path).exists():
        esc = _esc_path(subtitle_path)
        filters.append(f"ass='{esc}'")

    vf = ",".join(filters)

    # 타겟 비트레이트 (원본 비트레이트와 최대한 유사하게)
    v_bitrate = clip.get("video_bitrate_kbps", 0)
    if v_bitrate < 1000:
        v_bitrate = 8000  # 기본값
    a_bitrate = clip.get("audio_bitrate_kbps", 192)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{trim_start:.3f}",
        "-i", filepath,
        "-t", f"{seg_duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-preset", FFMPEG_PRESET,
        "-b:v", f"{v_bitrate}k",
        "-maxrate", f"{int(v_bitrate * 1.5)}k",
        "-bufsize", f"{v_bitrate * 2}k",
        "-r", "30",
        "-pix_fmt", "yuv420p",
    ]

    if clip.get("has_audio", True):
        cmd += [
            "-c:a", "aac",
            "-b:a", f"{a_bitrate}k",
            "-ar", "48000",
            "-ac", "2",
        ]
    else:
        cmd += [
            "-an",  # 오디오 없는 클립
        ]

    cmd += [
        "-movflags", "+faststart",
        output_path
    ]

    timeout = max(600, int(seg_duration * 20))
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        err = result.stderr[-600:].decode(errors="replace")
        print(f"  [오류] 렌더링 실패: {Path(filepath).name}\n    {err}")
        return False
    return True


def is_valid_video(path: str) -> bool:
    """ffprobe로 파일이 유효한 비디오인지 확인"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, timeout=30
    )
    return result.returncode == 0 and result.stdout.strip() != b""


def concat_day(segment_paths: List[str], output_path: str,
               clip_durations: List[float] = None) -> bool:
    """하루 분량 세그먼트를 하나로 합치기 (stream copy).

    clip_durations 를 넘기면 ffmpeg progress 파이프를 파싱해 클립 단위 프로그레스 바 표시.
    """
    n = len(segment_paths)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for p in segment_paths:
            # concat demuxer requires escaped absolute paths
            abs_p = os.path.abspath(p)
            escaped = abs_p.replace("\\", "/").replace("'", "\\'")
            f.write(f"file '{escaped}'\n")

    # 누적 타임라인 경계 계산 (어느 시각이 몇 번째 클립인지 매핑)
    boundaries: List[float] = []
    if clip_durations:
        t = 0.0
        for d in clip_durations:
            t += d
            boundaries.append(t)

    use_progress = bool(boundaries) and HAS_TQDM

    try:
        if use_progress:
            r_fd, w_fd = os.pipe()
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                "-movflags", "+faststart",
                "-progress", f"pipe:{w_fd}",
                "-nostats",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]

        if use_progress:
            pbar = _tqdm(total=n, desc="  병합", unit="클립")
            current = [0]  # mutable cell for thread

            def _watch_progress():
                try:
                    with os.fdopen(r_fd, "r") as pipe:
                        for line in pipe:
                            if not line.startswith("out_time="):
                                continue
                            # out_time 형식: HH:MM:SS.XXXXXX
                            try:
                                ts = line.split("=", 1)[1].strip()
                                h, m, s = ts.split(":")
                                out_sec = int(h) * 3600 + int(m) * 60 + float(s)
                            except (ValueError, IndexError):
                                continue
                            # 현재 클립 인덱스 계산
                            new_idx = next(
                                (i for i, b in enumerate(boundaries) if out_sec < b),
                                n - 1
                            )
                            if new_idx > current[0]:
                                pbar.update(new_idx - current[0])
                                current[0] = new_idx
                except OSError:
                    pass

            watcher = threading.Thread(target=_watch_progress, daemon=True)
            watcher.start()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(w_fd,),
            )
            os.close(w_fd)   # 부모 프로세스에서 쓰기 끝 닫기
            proc.wait()
            watcher.join(timeout=5)

            # 마지막 클립까지 채워줌
            if current[0] < n:
                pbar.update(n - current[0])
            pbar.close()

            if proc.returncode != 0:
                err = (proc.stderr.read() if proc.stderr else b"")[-400:].decode(errors="replace")
                print(f"  [오류] 병합 실패: {err}")
                return False
            return True
        else:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                err = result.stderr[-400:].decode(errors="replace")
                print(f"  [오류] 병합 실패: {err}")
                return False
            return True
    finally:
        try:
            os.unlink(concat_file)
        except OSError:
            pass


def _esc_path(path: str) -> str:
    """ffmpeg filter 경로 이스케이프"""
    return path.replace("\\", "/").replace("'", "\\'").replace(":", "\\:")
