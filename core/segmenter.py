"""긴 클립을 일정 길이 세그먼트로 분할 (클립 간/내부 병렬 처리)"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from config import MAX_SEGMENT_DURATION, MIN_SEGMENT_DURATION
from core.cache import make_segment_hash


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
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"세그먼트 분할 실패: {result.stderr[-500:].decode(errors='replace')}"
        )
