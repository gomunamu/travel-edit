"""ffmpeg 기반 클립 렌더링 및 병합"""
import subprocess
import os
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


def render_day_onepass(
    clips_info: List[dict],
    output_path: str,
    out_res: Tuple[int, int],
) -> bool:
    """
    선택된 클립들을 filter_complex 로 한 번에 트림·스케일·자막 번인·병합.
    중간 파일 없음. 원본 파일에서 직접 읽어 최종 출력물 하나로 인코딩.

    clips_info 각 항목에 필요한 키:
        filepath, _src_start, trim_start, trim_end,
        is_portrait, has_audio, sub_path (or None), loc_path (or None)
    """
    out_w, out_h = out_res
    n = len(clips_info)

    cmd_inputs: List[str] = []
    filter_parts: List[str] = []
    vstreams: List[str] = []
    astreams: List[str] = []

    # 프로그레스 바용 누적 경계
    total_dur = 0.0
    boundaries: List[float] = []

    for i, clip in enumerate(clips_info):
        src = clip["filepath"]
        src_start = clip.get("_src_start", 0.0)
        trim_start = clip.get("trim_start", 0.0)
        trim_end = clip.get("trim_end", clip["duration"])
        abs_start = src_start + trim_start
        abs_dur = max(0.1, trim_end - trim_start)
        total_dur += abs_dur
        boundaries.append(total_dur)

        # 입력: fast seek + 정확한 길이 제한
        cmd_inputs += ["-ss", f"{abs_start:.3f}", "-t", f"{abs_dur:.3f}", "-i", src]

        # 비디오 필터 체인
        vf = f"[{i}:v]setpts=PTS-STARTPTS,{build_scale_filter(clip.get('is_portrait', False), out_w, out_h)}"
        loc_path = clip.get("loc_path")
        sub_path = clip.get("sub_path")
        if loc_path and Path(loc_path).exists():
            vf += f",ass='{_esc_path(loc_path)}'"
        if sub_path and Path(sub_path).exists():
            vf += f",ass='{_esc_path(sub_path)}'"
        vf += f"[v{i}]"
        filter_parts.append(vf)
        vstreams.append(f"[v{i}]")

        # 오디오 필터 체인
        if clip.get("has_audio", True):
            filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        else:
            # 오디오 트랙 없는 클립 → 무음 대체
            filter_parts.append(
                f"anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={abs_dur:.3f}[a{i}]"
            )
        astreams.append(f"[a{i}]")

    # concat 필터
    concat_in = "".join(vstreams) + "".join(astreams)
    filter_parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]")

    # 프로그레스 파이프 설정
    r_fd, w_fd = os.pipe()
    cmd = (
        ["ffmpeg", "-y"]
        + cmd_inputs
        + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-crf", str(CRF),
            "-preset", FFMPEG_PRESET,
            "-r", "30",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            "-progress", f"pipe:{w_fd}",
            "-nostats",
            output_path,
        ]
    )

    if HAS_TQDM:
        pbar = _tqdm(total=n, desc="  렌더링+병합", unit="클립")
        current = [0]

        def _watch():
            try:
                with os.fdopen(r_fd, "r") as pipe:
                    for line in pipe:
                        if not line.startswith("out_time="):
                            continue
                        try:
                            ts = line.split("=", 1)[1].strip()
                            h, m, s = ts.split(":")
                            out_sec = int(h) * 3600 + int(m) * 60 + float(s)
                        except (ValueError, IndexError):
                            continue
                        new_idx = next(
                            (i for i, b in enumerate(boundaries) if out_sec < b),
                            n - 1
                        )
                        if new_idx > current[0]:
                            pbar.update(new_idx - current[0])
                            current[0] = new_idx
            except OSError:
                pass

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
    else:
        watcher = None
        pbar = None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        pass_fds=(w_fd,),
    )
    os.close(w_fd)
    proc.wait()

    if watcher:
        watcher.join(timeout=5)
    if pbar:
        if current[0] < n:
            pbar.update(n - current[0])
        pbar.close()

    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else b"")[-800:].decode(errors="replace")
        print(f"  [오류] 렌더링 실패:\n{err}")
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


def _esc_path(path: str) -> str:
    """ffmpeg filter 경로 이스케이프"""
    return path.replace("\\", "/").replace("'", "\\'").replace(":", "\\:")
