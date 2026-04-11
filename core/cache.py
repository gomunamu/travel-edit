"""캐시 관리 모듈 - 클립별 중간 결과물 저장/로드"""
import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Any


def make_clip_hash(filepath: str) -> str:
    """파일 경로 + mtime + size 기반 캐시 키 생성"""
    try:
        stat = os.stat(filepath)
        raw = f"{filepath}|{stat.st_mtime}|{stat.st_size}"
    except OSError:
        raw = filepath
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def variant_tag(*parts: str) -> str:
    """설정값 조합을 8자 해시로 변환 — 캐시 키 네임스페이스 구분용.

    예: variant_tag("large-v3", "ko") → "a1b2c3d4"
    같은 파일이라도 설정이 다르면 별도 캐시 슬롯을 사용하도록 한다.
    """
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def make_segment_hash(parent_hash: str, segment_index: int) -> str:
    """분할된 세그먼트의 캐시 키"""
    raw = f"{parent_hash}|seg{segment_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Cache:
    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, clip_hash: str, suffix: str) -> Path:
        return self.root / f"{clip_hash}_{suffix}.json"

    def _mp4_path(self, clip_hash: str) -> Path:
        return self.root / f"{clip_hash}_segment.mp4"

    def _ass_path(self, clip_hash: str, kind: str) -> Path:
        return self.root / f"{clip_hash}_{kind}.ass"

    # --- JSON 캐시 ---
    def load(self, clip_hash: str, suffix: str) -> Optional[dict]:
        p = self._json_path(clip_hash, suffix)
        if p.exists() and p.stat().st_size > 0:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def save(self, clip_hash: str, suffix: str, data: Any):
        p = self._json_path(clip_hash, suffix)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 세그먼트 MP4 캐시 ---
    def segment_path(self, clip_hash: str) -> str:
        return str(self._mp4_path(clip_hash))

    def segment_exists(self, clip_hash: str) -> bool:
        p = self._mp4_path(clip_hash)
        return p.exists() and p.stat().st_size > 10_000  # 최소 10KB

    def segment_needs_rerender(self, clip_hash: str) -> bool:
        """eval이 segment보다 최신이면 재렌더 필요"""
        seg = self._mp4_path(clip_hash)
        ev = self._json_path(clip_hash, "eval")
        if not seg.exists():
            return True
        if ev.exists() and ev.stat().st_mtime > seg.stat().st_mtime:
            return True
        return False

    # --- ASS 파일 캐시 ---
    def ass_path(self, clip_hash: str, kind: str) -> str:
        return str(self._ass_path(clip_hash, kind))

    def ass_exists(self, clip_hash: str, kind: str) -> bool:
        return self._ass_path(clip_hash, kind).exists()
