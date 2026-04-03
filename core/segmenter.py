"""긴 클립을 일정 길이 세그먼트로 논리 분할 (파일 추출 없음)"""
from typing import List

from config import MAX_SEGMENT_DURATION, MIN_SEGMENT_DURATION
from core.cache import make_segment_hash


def plan_segments(clip: dict) -> List[dict]:
    """
    클립을 논리 세그먼트로 분할 — 파일 추출 없음.
    filepath 는 항상 원본 파일을 가리키며,
    _src_start 로 원본 내 절대 시작 위치를 기록한다.
    """
    duration = clip["duration"]

    if duration <= MAX_SEGMENT_DURATION:
        seg = dict(clip)
        seg["segment_index"] = 0
        seg["parent_hash"] = None
        seg["trim_start"] = 0.0
        seg["trim_end"] = duration
        seg["_src_start"] = 0.0
        return [seg]

    parent_hash = clip["clip_hash"]
    plan = []
    seg_idx = 0
    current = 0.0

    while current < duration - MIN_SEGMENT_DURATION:
        end = min(current + MAX_SEGMENT_DURATION, duration)
        seg_len = end - current
        if seg_len < MIN_SEGMENT_DURATION:
            break

        seg_hash = make_segment_hash(parent_hash, seg_idx)
        seg = dict(clip)
        # filepath 는 원본 그대로 유지 (추출 안 함)
        seg["clip_hash"] = seg_hash
        seg["parent_hash"] = parent_hash
        seg["segment_index"] = seg_idx
        seg["duration"] = seg_len
        seg["trim_start"] = 0.0
        seg["trim_end"] = seg_len
        seg["_src_start"] = current   # 원본 파일 내 절대 위치

        plan.append(seg)
        current = end
        seg_idx += 1

    return plan if plan else [dict(clip)]
