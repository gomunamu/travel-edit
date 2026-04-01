"""긴 클립을 일정 길이 세그먼트로 분할 (클립 간/내부 병렬 처리)"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from config import MAX_SEGMENT_DURATION, MIN_SEGMENT_DURATION
from core.cache import make_segment_hash


def _plan_segments(clip: dict, out_dir: Path) -> List[dict]:
    """분할 계획만 수립 (파일 추출 없이 메타 dict 목록 반환)"""
    filepath = clip["filepath"]
    duration = clip["duration"]
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

    return plan


def _extract_segment(src: str, start: float, duration: float, dst: str):
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


def split_clip(clip: dict, segments_dir: str, inner_workers: int = 4) -> List[dict]:
    """
    클립이 MAX_SEGMENT_DURATION보다 길면 분할.
    하나의 클립 내 세그먼트들을 inner_workers 만큼 병렬 추출.
    """
    if clip["duration"] <= MAX_SEGMENT_DURATION:
        clip["segment_index"] = 0
        clip["parent_hash"] = None
        clip["trim_start"] = 0.0
        clip["trim_end"] = clip["duration"]
        return [clip]

    out_dir = Path(segments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = _plan_segments(clip, out_dir)
    if not plan:
        return [clip]

    # 이미 추출된 세그먼트 건너뜀, 나머지 병렬 추출
    to_extract = [
        seg for seg in plan
        if not Path(seg["filepath"]).exists()
        or Path(seg["filepath"]).stat().st_size < 10_000
    ]

    failed_paths = set()
    if to_extract:
        def _do(seg):
            _extract_segment(seg["_src"], seg["_seg_start"], seg["duration"], seg["filepath"])

        with ThreadPoolExecutor(max_workers=inner_workers) as ex:
            futures = {ex.submit(_do, seg): seg for seg in to_extract}
            for future in as_completed(futures):
                seg = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [경고] 세그먼트 추출 실패, 건너뜀: {Path(seg['filepath']).name}\n    {e}")
                    failed_paths.add(seg["filepath"])

    # 임시 키 제거 후 반환 (순서 보존, 실패 세그먼트 제외)
    result = []
    for seg in plan:
        seg.pop("_src", None)
        seg.pop("_seg_start", None)
        if seg["filepath"] not in failed_paths:
            result.append(seg)

    return result if result else [clip]
