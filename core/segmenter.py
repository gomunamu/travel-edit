"""긴 클립을 일정 길이 세그먼트로 분할"""
import subprocess
from pathlib import Path
from typing import List

from config import MAX_SEGMENT_DURATION, MIN_SEGMENT_DURATION
from core.cache import make_segment_hash


def _ffmpeg_worker(args: tuple) -> str:
    """
    ProcessPoolExecutor용 모듈 레벨 워커 (picklable).
    같은 소스 파일의 세그먼트들을 시작 시간 순으로 순차 처리해
    HDD seek를 최소화한다.
    반환값: 실패한 경로들의 집합(set)
    """
    segs = args  # list of (src, start, duration, dst)
    failed = set()
    for src, start, duration, dst in segs:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", src,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            dst,
        ]
        # stdout/stderr=DEVNULL → posix_spawn 사용 가능 → fork 직렬화 없음
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        if result.returncode != 0:
            failed.add(dst)
    return failed


def plan_segments(clip: dict, out_dir: Path) -> List[dict]:
    """
    분할 계획 수립 (I/O 없음).
    - 짧은 클립: 원본 그대로 반환 (_src 없음 → 추출 불필요)
    - 긴 클립: out_dir에 저장될 세그먼트 목록 반환 (_src/_seg_start 포함)
    """
    duration = clip["duration"]

    if duration <= MAX_SEGMENT_DURATION:
        seg = dict(clip)
        seg["segment_index"] = 0
        seg["parent_hash"] = None
        seg["trim_start"] = 0.0
        seg["trim_end"] = duration
        return [seg]

    filepath = clip["filepath"]
    parent_hash = clip["clip_hash"]
    stem = Path(filepath).stem

    plan = []
    seg_idx = 0
    current = 0.0

    while current < duration - MIN_SEGMENT_DURATION:
        end = min(current + MAX_SEGMENT_DURATION, duration)
        seg_len = end - current
        if seg_len < MIN_SEGMENT_DURATION:
            break

        seg_hash = make_segment_hash(parent_hash, seg_idx)
        seg_path = out_dir / f"{stem}_s{seg_idx:03d}_{seg_hash}.mp4"

        seg = dict(clip)
        seg["filepath"] = str(seg_path)
        seg["clip_hash"] = seg_hash
        seg["parent_hash"] = parent_hash
        seg["segment_index"] = seg_idx
        seg["duration"] = seg_len
        seg["trim_start"] = 0.0
        seg["trim_end"] = seg_len
        seg["_src"] = filepath
        seg["_seg_start"] = current

        plan.append(seg)
        current = end
        seg_idx += 1

    return plan if plan else [dict(clip)]


def extract_segment(src: str, start: float, duration: float, dst: str):
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        dst,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"세그먼트 분할 실패: {result.stderr[-500:].decode(errors='replace')}"
        )
