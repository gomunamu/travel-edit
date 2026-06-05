"""ffmpeg 기반 클립 렌더링 및 병합"""
import atexit
import hashlib
import os
import platform
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

from config import (CRF, FFMPEG_PRESET, RENDER_WORKERS, VIDEO_CODEC,
                    USE_NVENC, NVENC_PRESET, NVENC_MAX_SESSIONS,
                    USE_VIDEOTOOLBOX, VIDEOTOOLBOX_MAX_SESSIONS, VIDEOTOOLBOX_QUALITY)

_IS_MACOS = platform.system() == "Darwin"

try:
    from tqdm import tqdm as _tqdm_cls
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _detect_nvenc() -> bool:
    """ffmpeg에서 h264_nvenc 사용 가능 여부 확인 (실제 인코딩 시도).
    NVENC 최소 해상도 제약(145px) 때문에 256x256 사용."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


_NVENC_AVAILABLE: Optional[bool] = None  # lazy 감지
_nvenc_semaphore: Optional[threading.Semaphore] = None  # 동시 세션 제한


def _use_nvenc() -> bool:
    global _NVENC_AVAILABLE, _nvenc_semaphore
    if _IS_MACOS:
        return False  # macOS에서는 NVENC 불가
    if USE_NVENC == "false":
        return False
    if USE_NVENC == "true":
        if _nvenc_semaphore is None:
            _nvenc_semaphore = threading.Semaphore(NVENC_MAX_SESSIONS)
        return True
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = _detect_nvenc()
        if _NVENC_AVAILABLE:
            _nvenc_semaphore = threading.Semaphore(NVENC_MAX_SESSIONS)
            print(f"  [NVENC] GPU 하드웨어 인코딩 활성화 (동시 세션 최대 {NVENC_MAX_SESSIONS}개)")
        else:
            print("  [NVENC] GPU 인코딩 불가 → CPU 인코딩 사용")
    return bool(_NVENC_AVAILABLE)


def _is_nvenc_session_error(stderr: str) -> bool:
    """NVENC 세션 한도 초과 또는 VRAM 부족 에러 감지."""
    return any(k in stderr for k in (
        "out of memory",
        "incompatible client key",
        "OpenEncodeSessionEx failed",
        "InitializeEncoder failed",
        "CreateInputBuffer failed",
    ))


# ── VideoToolbox (macOS) ─────────────────────────────────────────────────────

def _detect_videotoolbox() -> bool:
    """ffmpeg에서 h264_videotoolbox 사용 가능 여부 확인."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-c:v", "h264_videotoolbox", "-f", "null", "-"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


_VT_AVAILABLE: Optional[bool] = None
_vt_semaphore: Optional[threading.Semaphore] = None


def _use_videotoolbox() -> bool:
    global _VT_AVAILABLE, _vt_semaphore
    if not _IS_MACOS:
        return False
    if USE_VIDEOTOOLBOX == "false":
        return False
    if USE_VIDEOTOOLBOX == "true":
        if _vt_semaphore is None:
            _vt_semaphore = threading.Semaphore(VIDEOTOOLBOX_MAX_SESSIONS)
        return True
    if _VT_AVAILABLE is None:
        _VT_AVAILABLE = _detect_videotoolbox()
        if _VT_AVAILABLE:
            _vt_semaphore = threading.Semaphore(VIDEOTOOLBOX_MAX_SESSIONS)
            print(f"  [VideoToolbox] GPU 하드웨어 인코딩 활성화 (동시 세션 최대 {VIDEOTOOLBOX_MAX_SESSIONS}개)")
        else:
            print("  [VideoToolbox] GPU 인코딩 불가 → CPU 인코딩 사용")
    return bool(_VT_AVAILABLE)


def _is_vt_session_error(stderr: str) -> bool:
    """VideoToolbox 세션 한도 초과 또는 인코더 오류 감지."""
    return any(k in stderr for k in (
        "videotoolbox_encode_frame",
        "Error while encoding",
        "cannot open encoder",
        "VTCompressionSessionCreate",
    ))


def _build_encode_args(codec_extra_tag: bool = False, force_cpu: bool = False) -> list:
    """
    인코더 설정 인수 반환.
    NVENC:         -c:v h264_nvenc  -preset p4 -cq {CRF} -b:v 0
    VideoToolbox:  -c:v h264_videotoolbox -q:v {QUALITY} -b:v 0
    CPU:           -c:v libx264    -crf {CRF} -preset medium
    force_cpu=True 이면 하드웨어 인코더 무시하고 CPU 인수 반환.
    """
    if not force_cpu and _use_nvenc():
        nvenc_codec = "hevc_nvenc" if VIDEO_CODEC == "libx265" else "h264_nvenc"
        args = ["-c:v", nvenc_codec, "-preset", NVENC_PRESET, "-cq", str(CRF), "-b:v", "0"]
        if codec_extra_tag and VIDEO_CODEC == "libx265":
            args += ["-tag:v", "hvc1"]
        return args
    if not force_cpu and _use_videotoolbox():
        vt_codec = "hevc_videotoolbox" if VIDEO_CODEC == "libx265" else "h264_videotoolbox"
        args = ["-c:v", vt_codec, "-q:v", str(VIDEOTOOLBOX_QUALITY), "-b:v", "0"]
        if codec_extra_tag and VIDEO_CODEC == "libx265":
            args += ["-tag:v", "hvc1"]
        return args
    codec_extra = ["-tag:v", "hvc1"] if codec_extra_tag and VIDEO_CODEC == "libx265" else []
    return ["-c:v", VIDEO_CODEC, "-crf", str(CRF), "-preset", FFMPEG_PRESET] + codec_extra


# 해상도 자동 선택 계단 (긴 변 기준, 내림차순)
_RES_TIERS: List[Tuple[int, int]] = [
    (3840, 2160),  # 4K UHD
    (2560, 1440),  # 1440p QHD
    (1920, 1080),  # 1080p FHD
    (1280,  720),  # 720p HD
]


_STANDARD_FPS = [24, 25, 30, 48, 50, 60, 90, 120]


def get_day_fps(clips: List[dict]) -> int:
    """
    클립 전체의 FPS를 검사해 출력 FPS를 결정한다.
    - 모든 클립이 30fps 초과이면 그 중 최솟값을 가장 가까운 표준 FPS로 내림
    - 한 클립이라도 30fps 이하이면 30 반환
    """
    fpss = [clip.get("fps", 30.0) for clip in clips if clip.get("fps")]
    if not fpss:
        return 30
    min_fps = min(fpss)
    if min_fps <= 30.0:
        return 30
    # 30fps 초과: 표준 값 중 min_fps 이하 최댓값 선택
    candidates = [f for f in _STANDARD_FPS if f <= min_fps]
    return max(candidates) if candidates else 30


def get_day_resolution(clips: List[dict]) -> Tuple[int, int]:
    """
    OUTPUT_RESOLUTION이 None(auto)이면 클립 중 가장 긴 변을 기준으로
    4K / 1440p / FHD / 720p 중 업스케일이 없는 최고 계단을 선택.
    고정값이 설정돼 있으면 그대로 사용.
    SPLIT_ORIENTATION이 켜져 있고 그룹이 세로 전용이면 해상도를 90도 회전해 반환.
    """
    from config import OUTPUT_RESOLUTION, SPLIT_ORIENTATION

    # 세로 전용 그룹 여부 (SPLIT_ORIENTATION 활성 시에만 의미 있음)
    is_portrait_group = SPLIT_ORIENTATION and all(
        c.get("is_portrait") for c in clips
    )

    if OUTPUT_RESOLUTION is not None:
        w, h = OUTPUT_RESOLUTION
        return (h, w) if is_portrait_group else (w, h)

    max_long = 0
    for clip in clips:
        w = clip.get("display_width",  clip.get("raw_width",  0))
        h = clip.get("display_height", clip.get("raw_height", 0))
        max_long = max(max_long, w, h)

    if max_long == 0:
        base = (1920, 1080)
    else:
        base = next(
            ((tw, th) for tw, th in _RES_TIERS if max_long >= tw),
            (1280, 720),
        )

    return (base[1], base[0]) if is_portrait_group else base


def build_scale_filter(is_portrait: bool, out_w: int, out_h: int) -> str:
    """
    입력 영상을 out_w×out_h 안에 비율 유지하며 맞추고 남은 영역을 블랙으로 패딩.
    가로/세로 출력 모두 동일한 공식으로 처리 (필러박스·레터박스 자동).
    """
    return (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1"
    )


def _worker_count(out_res: Tuple[int, int] = (1920, 1080)) -> int:
    if RENDER_WORKERS is not None:
        return max(1, RENDER_WORKERS)
    cpu = os.cpu_count() or 2
    # NVENC/VideoToolbox: 하드웨어 인코딩으로 CPU 부하가 크게 줄어 cpu//2 까지 병렬 가능
    if _use_nvenc() or _use_videotoolbox():
        return max(1, cpu // 2)
    # CPU 인코딩: 고해상도일수록 클립당 CPU/메모리 부하 증가
    long_side = max(out_res)
    if long_side >= 3840:   # 4K
        return max(1, cpu // 8)
    elif long_side >= 2560: # 1440p
        return max(1, cpu // 6)
    elif long_side >= 1920: # 1080p
        return max(1, cpu // 4)
    else:                   # 720p 이하
        return max(1, cpu // 2)


def render_day_onepass(
    clips_info: List[dict],
    output_path: str,
    out_res: Tuple[int, int],
    out_fps: int = 30,
) -> bool:
    """
    클립마다 독립적인 ffmpeg 프로세스로 병렬 인코딩한 뒤 stream copy로 병합.

    동시 실행 수 = RENDER_WORKERS (기본: cpu_count // 2).
    동시 실행 수가 곧 NAS 동시 연결 수 상한이 되므로 별도 배치 분할 불필요.
    최종 병합은 재인코딩 없는 stream copy라 빠르다.
    """
    n = len(clips_info)
    workers = _worker_count(out_res)
    stem = Path(output_path).stem

    print(f"  병렬 렌더링: {n}개 클립 / 워커 {workers}개 (cpu={os.cpu_count()})")

    # 임시 클립을 /tmp/ 아래에 생성 — 출력 경로에 아포스트로피/공백 등 특수문자가
    # 있어도 FFmpeg concat 파서가 경로를 안전하게 읽을 수 있다.
    tmp_dir = Path(tempfile.mkdtemp(prefix="_tve_render_"))
    temp_paths = [str(tmp_dir / f"{stem}_clip{i:04d}.mp4") for i in range(n)]

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

        ok = _render_clip(clip, temp_paths[i], out_res, out_fps)

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
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        return False

    # 클립 1개면 rename만
    if n == 1:
        Path(temp_paths[0]).rename(output_path)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        return True

    total_sec = sum(
        max(0.1, clip.get("trim_end", clip["duration"]) - clip.get("trim_start", 0.0))
        / max(1.0, float(clip.get("speed", 1.0) or 1.0))
        for clip in clips_info
    )
    ok = _concat_clips(temp_paths, output_path, total_sec)
    for p in temp_paths:
        Path(p).unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return ok


def _render_clip(clip: dict, output_path: str, out_res: Tuple[int, int], out_fps: int = 30) -> bool:
    """클립 1개를 ffmpeg로 인코딩해 output_path에 저장."""
    out_w, out_h = out_res
    src_start = clip.get("_src_start", 0.0)
    trim_start = clip.get("trim_start", 0.0)
    trim_end = clip.get("trim_end", clip["duration"])
    abs_start = src_start + trim_start
    abs_dur = max(0.1, trim_end - trim_start)   # 읽어들일 원본 길이
    speed = max(1.0, float(clip.get("speed", 1.0) or 1.0))
    out_dur = abs_dur / speed                   # 배속 적용 후 출력 길이

    # 배속: setpts 로 영상을 압축. 무음 컷만 배속되므로 자막(sub) 동기화 문제 없음.
    setpts = f"setpts=(PTS-STARTPTS)/{speed:g}" if speed > 1.0 else "setpts=PTS-STARTPTS"
    vf = (f"{setpts},"
          f"{build_scale_filter(clip.get('is_portrait', False), out_w, out_h)}")
    loc_path = clip.get("loc_path")
    sub_path = clip.get("sub_path")
    if loc_path and Path(loc_path).exists():
        vf += f",ass={_safe_filter_path(loc_path)}"
    if sub_path and Path(sub_path).exists():
        vf += f",ass={_safe_filter_path(sub_path)}"

    encode_args = [
        "-vf", vf,
        *_build_encode_args(codec_extra_tag=True),
        "-r", str(out_fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]

    hw = _use_nvenc() or _use_videotoolbox()
    return _run_encode(clip, output_path, abs_start, abs_dur, encode_args,
                       use_gpu=hw, out_fps=out_fps, speed=speed, out_dur=out_dur)


def _build_cmd(clip: dict, abs_start: float, abs_dur: float,
               encode_args: list, use_gpu: bool,
               speed: float = 1.0, out_dur: Optional[float] = None) -> list:
    if out_dur is None:
        out_dur = abs_dur
    if not use_gpu:
        hwaccel = []
    elif _use_nvenc():
        hwaccel = ["-hwaccel", "cuda"]
    else:
        hwaccel = ["-hwaccel", "videotoolbox"]
    base = ["ffmpeg", "-y"] + hwaccel + [
        "-ss", f"{abs_start:.3f}", "-t", f"{abs_dur:.3f}", "-i", clip["filepath"],
    ]
    if clip.get("has_audio", True) and speed <= 1.0:
        return base + encode_args
    # 오디오 없는 클립 또는 배속 컷: lavfi 무음 소스로 대체 (출력 길이 = out_dur).
    # 배속 컷의 원본 오디오는 무음/피치왜곡 방지를 위해 사용하지 않는다.
    return base + [
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-map", "0:v", "-map", "1:a",
        "-t", f"{out_dur:.3f}",
    ] + encode_args


def _run_encode(clip: dict, output_path: str,
                abs_start: float, abs_dur: float,
                encode_args: list, use_gpu: bool, out_fps: int = 30,
                speed: float = 1.0, out_dur: Optional[float] = None) -> bool:
    """ffmpeg 실행. use_gpu=True 시 하드웨어 세마포어 획득 후 실행."""
    sem = (_nvenc_semaphore or _vt_semaphore) if use_gpu else None
    if sem:
        sem.acquire()
    try:
        cmd = _build_cmd(clip, abs_start, abs_dur, encode_args, use_gpu,
                         speed=speed, out_dur=out_dur)
        result = subprocess.run(cmd, capture_output=True, timeout=600)  # 10분 상한
    except subprocess.TimeoutExpired:
        fname = Path(clip.get("filepath", "?")).name
        print(f"\n  [오류] 클립 인코딩 타임아웃 (10분 초과): {fname}")
        return False
    finally:
        if sem:
            sem.release()

    if result.returncode != 0:
        raw = result.stderr.decode(errors="replace")

        def _cpu_retry(label: str) -> bool:
            print(f"\n  [{label}] {Path(clip['filepath']).name} → CPU 인코딩으로 재시도")
            cpu_encode_args = _build_encode_args(codec_extra_tag=True, force_cpu=True)
            vf_idx = encode_args.index("-vf")
            vf_val = encode_args[vf_idx + 1]
            out_idx = next(i for i, a in enumerate(encode_args)
                           if not a.startswith("-") and i > 4)
            tail_args = encode_args[out_idx:]
            cpu_full = ["-vf", vf_val] + cpu_encode_args + [
                "-r", str(out_fps), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart",
            ] + tail_args
            return _run_encode(clip, output_path, abs_start, abs_dur,
                               cpu_full, use_gpu=False, out_fps=out_fps,
                               speed=speed, out_dur=out_dur)

        if use_gpu and _is_nvenc_session_error(raw):
            return _cpu_retry("NVENC 한도")
        if use_gpu and _is_vt_session_error(raw):
            return _cpu_retry("VideoToolbox 한도")

        head = raw[:800]
        tail = raw[-400:] if len(raw) > 800 else ""
        err = head + ("\n...\n" + tail if tail else "")
        print(f"\n  [오류] 클립 인코딩 실패 ({Path(clip['filepath']).name}):\n{err}")
        return False
    return True


def _concat_clips(clip_paths: List[str], output_path: str, total_sec: float = 0.0) -> bool:
    """인코딩된 클립들을 stream copy로 이어 붙인다. ffmpeg progress로 진행률 표시."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                    delete=False, encoding="utf-8") as f:
        concat_file = f.name
        for p in clip_paths:
            # 임시 클립 경로는 /tmp/_tve_render_*/ 아래 — 특수문자 없음
            f.write(f"file '{os.path.abspath(p)}'\n")

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


_safe_symlinks: set = set()


def _cleanup_safe_symlinks():
    for p in _safe_symlinks:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_safe_symlinks)


def _safe_filter_path(path: str) -> str:
    """ffmpeg -vf filter 옵션 값으로 안전하게 사용할 수 있는 경로를 반환한다.
    경로에 아포스트로피·공백·콜론 등 FFmpeg filter 파서가 특수문자로 처리하는
    문자가 포함된 경우, /tmp/ 아래 해시 기반 심링크를 생성해 반환한다.
    심링크 경로는 영문자·숫자·하이픈·점만 포함하므로 파서 오류가 없다.
    """
    _unsafe = frozenset("'\"\\:,;[] \t")
    if not any(c in path for c in _unsafe):
        return path
    h = hashlib.md5(path.encode()).hexdigest()[:20]
    safe = Path(f"/tmp/_tve_{h}{Path(path).suffix}")
    try:
        safe.unlink(missing_ok=True)
        safe.symlink_to(path)
        _safe_symlinks.add(str(safe))
    except OSError:
        return path  # 심링크 생성 실패 시 원본 경로로 폴백
    return str(safe)
