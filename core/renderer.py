"""ffmpeg 기반 클립 렌더링 및 병합"""
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

from config import CRF, FFMPEG_PRESET, RENDER_WORKERS

try:
    from tqdm import tqdm as _tqdm_cls
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# 해상도 자동 선택 계단 (긴 변 기준, 내림차순)
_RES_TIERS: List[Tuple[int, int]] = [
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # 1440p QHD
    (1920, 1080),  # 1080p FHD
    (1280,  720),  # 720p HD
]


def get_day_resolution(clips: List[dict]) -> Tuple[int, int]:
    """
    OUTPUT_RESOLUTION이 None(auto)이면 클립 중 가장 긴 변을 기준으로
    4K / 1440p / FHD / 720p 중 업스케일이 없는 최고 계단을 선택.
    고정값이 설정돼 있으면 그대로 사용.
    """
    from config import OUTPUT_RESOLUTION
    if OUTPUT_RESOLUTION is not None:
        return OUTPUT_RESOLUTION

    max_long = 0
    for clip in clips:
        w = clip.get("display_width",  clip.get("raw_width",  0))
        h = clip.get("display_height", clip.get("raw_height", 0))
        max_long = max(max_long, w, h)

    if max_long == 0:
        return (1920, 1080)

    for tier_w, tier_h in _RES_TIERS:
        if max_long >= tier_w:
            return (tier_w, tier_h)

    return (1280, 720)  # 최소


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


def _worker_count() -> int:
    if RENDER_WORKERS is not None:
        return max(1, RENDER_WORKERS)
    return max(1, (os.cpu_count() or 2) // 2)


def render_day_onepass(
    clips_info: List[dict],
    output_path: str,
    out_res: Tuple[int, int],
) -> bool:
    """
    클립마다 독립적인 ffmpeg 프로세스로 병렬 인코딩한 뒤 stream copy로 병합.

    동시 실행 수 = RENDER_WORKERS (기본: cpu_count // 2).
    동시 실행 수가 곧 NAS 동시 연결 수 상한이 되므로 별도 배치 분할 불필요.
    최종 병합은 재인코딩 없는 stream copy라 빠르다.
    """
    n = len(clips_info)
    workers = _worker_count()
    out_dir = Path(output_path).parent
    stem = Path(output_path).stem

    print(f"  병렬 렌더링: {n}개 클립 / 워커 {workers}개 (cpu={os.cpu_count()})")

    # 순서 보장을 위해 인덱스 기반 임시 경로 사전 할당
    temp_paths = [str(out_dir / f".{stem}_clip{i:04d}.mp4") for i in range(n)]

    if HAS_TQDM:
        pbar = _tqdm_cls(total=n, desc="  렌더링", unit="클립")
    else:
        pbar = None

    active: dict = {}  # index → 파일명 (진행 중 클립 표시용)
    active_lock = threading.Lock()
    failed = threading.Event()

    def _render_one(i: int, clip: dict) -> bool:
        if failed.is_set():
            return False

        label = Path(clip["filepath"]).stem
        with active_lock:
            active[i] = label
            if pbar:
                pbar.set_postfix_str(" | ".join(active.values()), refresh=True)

        ok = _render_clip(clip, temp_paths[i], out_res)

        with active_lock:
            active.pop(i, None)
        if pbar:
            pbar.update(1)
            with active_lock:
                pbar.set_postfix_str(" | ".join(active.values()), refresh=True)

        if not ok:
            failed.set()
        return ok

    success = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_render_one, i, clip): i
                   for i, clip in enumerate(clips_info)}
        for future in as_completed(futures):
            i = futures[future]
            success[i] = future.result()

    if pbar:
        pbar.close()

    if not all(success.get(i, False) for i in range(n)):
        for p in temp_paths:
            Path(p).unlink(missing_ok=True)
        return False

    # 클립 1개면 rename만
    if n == 1:
        Path(temp_paths[0]).rename(output_path)
        return True

    total_sec = sum(
        max(0.1, clip.get("trim_end", clip["duration"]) - clip.get("trim_start", 0.0))
        for clip in clips_info
    )
    ok = _concat_clips(temp_paths, output_path, total_sec)
    for p in temp_paths:
        Path(p).unlink(missing_ok=True)
    return ok


def _render_clip(clip: dict, output_path: str, out_res: Tuple[int, int]) -> bool:
    """클립 1개를 ffmpeg로 인코딩해 output_path에 저장."""
    out_w, out_h = out_res
    src_start = clip.get("_src_start", 0.0)
    trim_start = clip.get("trim_start", 0.0)
    trim_end = clip.get("trim_end", clip["duration"])
    abs_start = src_start + trim_start
    abs_dur = max(0.1, trim_end - trim_start)

    vf = (f"setpts=PTS-STARTPTS,"
          f"{build_scale_filter(clip.get('is_portrait', False), out_w, out_h)}")
    loc_path = clip.get("loc_path")
    sub_path = clip.get("sub_path")
    if loc_path and Path(loc_path).exists():
        vf += f",ass='{_esc_path(loc_path)}'"
    if sub_path and Path(sub_path).exists():
        vf += f",ass='{_esc_path(sub_path)}'"

    encode_args = [
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(CRF), "-preset", FFMPEG_PRESET,
        "-r", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]

    if clip.get("has_audio", True):
        cmd = (
            ["ffmpeg", "-y",
             "-ss", f"{abs_start:.3f}", "-t", f"{abs_dur:.3f}", "-i", clip["filepath"]]
            + encode_args
        )
    else:
        # 오디오 없는 클립: lavfi 무음 소스를 두 번째 입력으로 추가
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{abs_start:.3f}", "-t", f"{abs_dur:.3f}", "-i", clip["filepath"],
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-map", "0:v", "-map", "1:a",
            "-t", f"{abs_dur:.3f}",  # anullsrc는 무한이므로 출력 길이 제한
        ] + encode_args

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr[-600:].decode(errors="replace")
        print(f"\n  [오류] 클립 인코딩 실패 ({Path(clip['filepath']).name}):\n{err}")
        return False
    return True


def _concat_clips(clip_paths: List[str], output_path: str, total_sec: float = 0.0) -> bool:
    """인코딩된 클립들을 stream copy로 이어 붙인다. ffmpeg progress로 진행률 표시."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for p in clip_paths:
            escaped = os.path.abspath(p).replace("\\", "/").replace("'", "\\'")
            f.write(f"file '{escaped}'\n")

    r_fd, w_fd = os.pipe()
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-c", "copy", "-movflags", "+faststart",
        "-progress", f"pipe:{w_fd}", "-nostats",
        output_path,
    ]

    total_rounded = max(1, round(total_sec))
    if HAS_TQDM:
        pbar: Optional[object] = _tqdm_cls(
            total=total_rounded,
            desc="  병합",
            unit="s",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}]",
        )
    else:
        pbar = None
        print(f"  병합 중 ({len(clip_paths)}개 클립, {total_sec:.0f}초)...")

    last_sec = [0.0]

    def _watch():
        try:
            with os.fdopen(r_fd, "r") as pipe:
                for line in pipe:
                    if not line.startswith("out_time="):
                        continue
                    ts = line.split("=", 1)[1].strip()
                    if ts in ("N/A", "00:00:00.000000", ""):
                        continue
                    try:
                        h, m, s = ts.split(":")
                        out_sec = int(h) * 3600 + int(m) * 60 + float(s)
                    except (ValueError, IndexError):
                        continue
                    if pbar:
                        delta = out_sec - last_sec[0]
                        if delta > 0:
                            pbar.update(delta)  # type: ignore[attr-defined]
                            last_sec[0] = out_sec
        except OSError:
            pass

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(w_fd,)
        )
        os.close(w_fd)
        proc.wait()
        watcher.join(timeout=5)

        if pbar:
            remaining = total_rounded - round(last_sec[0])
            if remaining > 0:
                pbar.update(remaining)  # type: ignore[attr-defined]
            pbar.close()  # type: ignore[attr-defined]

        if proc.returncode != 0:
            err = (proc.stderr.read() if proc.stderr else b"")[-400:].decode(errors="replace")
            print(f"  [오류] 클립 병합 실패: {err}")
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
