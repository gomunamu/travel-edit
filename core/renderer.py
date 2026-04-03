"""ffmpeg 기반 클립 렌더링 및 병합"""
import subprocess
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Tuple, Optional

from config import OUTPUT_RESOLUTION, CRF, FFMPEG_PRESET, RENDER_BATCH_SIZE

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
    클립들을 RENDER_BATCH_SIZE 단위 배치로 나눠 렌더링한 뒤 stream copy로 병합.

    배치마다 ffmpeg filter_complex 호출 → 한 번에 여는 파일 수를 제한해
    NAS 동시 연결 폭주와 초기화 지연을 방지한다.
    최종 병합은 재인코딩 없는 stream copy라 빠르다.
    """
    total = len(clips_info)
    batches = [
        clips_info[i:i + RENDER_BATCH_SIZE]
        for i in range(0, total, RENDER_BATCH_SIZE)
    ]
    n_batches = len(batches)

    # 전체 프로그레스 바 (클립 단위)
    if HAS_TQDM:
        pbar = _tqdm(total=total, desc="  렌더링", unit="클립")
    else:
        pbar = None

    out_dir = Path(output_path).parent
    stem = Path(output_path).stem
    batch_paths: List[str] = []

    try:
        for b_idx, batch in enumerate(batches):
            batch_path = str(out_dir / f".{stem}_batch{b_idx:03d}.mp4")
            label = f"배치 {b_idx+1}/{n_batches} ({len(batch)}개 클립)"
            if pbar:
                pbar.set_postfix_str(f"{label} — 준비 중...", refresh=True)
            else:
                print(f"  {label}")

            ok = _render_batch(batch, batch_path, out_res, pbar, label)
            if not ok:
                return False
            batch_paths.append(batch_path)

        if pbar:
            pbar.set_postfix_str("배치 병합 중...", refresh=True)

    finally:
        if pbar:
            pbar.close()

    # 배치가 하나면 이름만 바꾸면 됨
    if len(batch_paths) == 1:
        Path(batch_paths[0]).rename(output_path)
        return True

    # 배치가 여럿이면 stream copy로 concat
    ok = _concat_batches(batch_paths, output_path)
    for p in batch_paths:
        Path(p).unlink(missing_ok=True)
    return ok


def _render_batch(
    clips: List[dict],
    output_path: str,
    out_res: Tuple[int, int],
    pbar=None,
    batch_label: str = "",
) -> bool:
    """clips 를 filter_complex 로 인코딩해 output_path 에 저장."""
    out_w, out_h = out_res
    n = len(clips)

    cmd_inputs: List[str] = []
    filter_parts: List[str] = []
    vstreams: List[str] = []
    astreams: List[str] = []
    boundaries: List[float] = []
    total_dur = 0.0

    for i, clip in enumerate(clips):
        src_start = clip.get("_src_start", 0.0)
        trim_start = clip.get("trim_start", 0.0)
        trim_end = clip.get("trim_end", clip["duration"])
        abs_start = src_start + trim_start
        abs_dur = max(0.1, trim_end - trim_start)
        total_dur += abs_dur
        boundaries.append(total_dur)

        cmd_inputs += ["-ss", f"{abs_start:.3f}", "-t", f"{abs_dur:.3f}", "-i", clip["filepath"]]

        vf = (f"[{i}:v]setpts=PTS-STARTPTS,"
              f"{build_scale_filter(clip.get('is_portrait', False), out_w, out_h)}")
        loc_path = clip.get("loc_path")
        sub_path = clip.get("sub_path")
        if loc_path and Path(loc_path).exists():
            vf += f",ass='{_esc_path(loc_path)}'"
        if sub_path and Path(sub_path).exists():
            vf += f",ass='{_esc_path(sub_path)}'"
        vf += f"[v{i}]"
        filter_parts.append(vf)
        vstreams.append(f"[v{i}]")

        if clip.get("has_audio", True):
            filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        else:
            filter_parts.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={abs_dur:.3f}[a{i}]"
            )
        astreams.append(f"[a{i}]")

    concat_in = "".join(vstreams) + "".join(astreams)
    filter_parts.append(f"{concat_in}concat=n={n}:v=1:a=1[vout][aout]")

    r_fd, w_fd = os.pipe()
    cmd = (
        ["ffmpeg", "-y"]
        + cmd_inputs
        + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-crf", str(CRF), "-preset", FFMPEG_PRESET,
            "-r", "30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            "-progress", f"pipe:{w_fd}", "-nostats",
            output_path,
        ]
    )

    completed = [0]

    def _watch():
        speed = ""
        try:
            with os.fdopen(r_fd, "r") as pipe:
                for line in pipe:
                    line = line.strip()
                    if line.startswith("speed="):
                        val = line.split("=", 1)[1].strip()
                        if val not in ("N/A", "0.000x", ""):
                            speed = val
                    elif line.startswith("out_time="):
                        ts = line.split("=", 1)[1].strip()
                        if ts in ("N/A", "00:00:00.000000", ""):
                            continue
                        try:
                            h, m, s = ts.split(":")
                            out_sec = int(h) * 3600 + int(m) * 60 + float(s)
                        except (ValueError, IndexError):
                            continue
                        new_idx = next(
                            (i for i, b in enumerate(boundaries) if out_sec < b),
                            n - 1
                        )
                        if pbar and new_idx > completed[0]:
                            pbar.update(new_idx - completed[0])
                            completed[0] = new_idx
                        if pbar and speed:
                            pbar.set_postfix_str(
                                f"{batch_label} — speed={speed}", refresh=True
                            )
        except OSError:
            pass

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(w_fd,)
    )
    os.close(w_fd)
    proc.wait()
    watcher.join(timeout=5)

    if pbar and completed[0] < n:
        pbar.update(n - completed[0])
        completed[0] = n

    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else b"")[-800:].decode(errors="replace")
        print(f"\n  [오류] 렌더링 실패 ({batch_label}):\n{err}")
        return False
    return True


def _concat_batches(batch_paths: List[str], output_path: str) -> bool:
    """인코딩된 배치 파일들을 stream copy로 이어 붙인다."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for p in batch_paths:
            escaped = os.path.abspath(p).replace("\\", "/").replace("'", "\\'")
            f.write(f"file '{escaped}'\n")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr[-400:].decode(errors="replace")
            print(f"  [오류] 배치 병합 실패: {err}")
            return False
        return True
    finally:
        try:
            os.unlink(concat_file)
        except OSError:
            pass


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
