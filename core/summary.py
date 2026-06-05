"""컷별 요약 리포트 생성 (clips_summary.json/csv, selected_clips.json)

모든 컷의 편집 결정·품질 점수·판단 이유·음성·장소를 사람이 읽을 수 있는
형태로 내보낸다. 어떤 장면이 왜 선택·제외됐는지 확인하고, 편집 스타일·규칙을
보정하는 데 쓰인다. (fullstack 참조본의 clips_summary 출력과 동일한 목적)
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

# CSV 열 순서 (가독성 우선: 핵심 → 점수 → 부가 정보)
_COLUMNS = [
    "clip_id", "day", "source_file", "segment_index", "orientation",
    "duration_sec", "decision", "selected",
    "kept_start_sec", "kept_end_sec", "kept_duration_sec", "speed", "output_sec",
    "score_total", "score_visual", "score_speech", "score_scene", "score_flow",
    "has_speech", "location", "speech_text", "reason",
]

_DECISION_LABEL = {"keep": "살림", "trim": "트림", "discard": "버림"}


def _speech_excerpt(transcript: dict, limit: int = 200) -> str:
    """전사 세그먼트를 한 줄로 합쳐 발췌 (음성 컷 내용 미리보기용)."""
    if not transcript:
        return ""
    segs = [s.get("text", "").strip() for s in transcript.get("segments", [])
            if s.get("no_speech_prob", 1) < 0.5 and s.get("text", "").strip()]
    text = " ".join(segs).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def build_clip_rows(
    segments: List[dict],
    evaluations: Dict[str, dict],
    transcripts: Dict[str, dict],
    selection_info: Dict[str, dict],
) -> List[dict]:
    """전체 세그먼트에 대해 요약 행 목록을 만든다.

    selection_info: render_day가 채운 {clip_hash: {selected, location, trim_start, trim_end}}.
      해당 날이 이미 렌더되어 건너뛴 경우 항목이 없을 수 있다 → selected=None("렌더 건너뜀").
    """
    rows: List[dict] = []
    for seg in segments:
        h = seg.get("clip_hash", "")
        ev = evaluations.get(h, {})
        sc = ev.get("score", {})
        tr = transcripts.get(h, {})
        sel = selection_info.get(h)

        if sel is not None:
            selected = sel.get("selected")
            location = sel.get("location", "")
            kept_start = round(float(sel.get("trim_start", 0.0)), 2)
            kept_end = round(float(sel.get("trim_end", seg.get("duration", 0.0))), 2)
            speed = round(float(sel.get("speed", 1.0)), 2)
        else:
            selected = None  # 렌더 건너뜀(기존 출력 재사용) → 선택 여부 불명
            location = ""
            kept_start = round(float(ev.get("keep_start", 0.0)), 2)
            kept_end = round(float(ev.get("keep_end", seg.get("duration", 0.0))), 2)
            speed = 1.0

        kept_dur = max(0.0, kept_end - kept_start)
        decision = ev.get("decision", "keep")
        rows.append({
            "clip_id": (h or "")[:8],
            "day": seg.get("day_key", ""),
            "source_file": seg.get("filename", ""),
            "segment_index": seg.get("segment_index", 0),
            "orientation": "세로" if seg.get("is_portrait") else "가로",
            "duration_sec": round(float(seg.get("duration", 0.0)), 2),
            "decision": _DECISION_LABEL.get(decision, decision),
            "selected": "" if selected is None else selected,
            "kept_start_sec": kept_start,
            "kept_end_sec": kept_end,
            "kept_duration_sec": round(kept_dur, 2),
            "speed": speed,
            "output_sec": round(kept_dur / speed, 2) if speed else round(kept_dur, 2),
            "score_total": sc.get("total", ""),
            "score_visual": sc.get("visual", ""),
            "score_speech": sc.get("speech", ""),
            "score_scene": sc.get("scene", ""),
            "score_flow": sc.get("flow", ""),
            "has_speech": tr.get("has_speech", False),
            "location": location,
            "speech_text": _speech_excerpt(tr),
            "reason": ev.get("reason", ""),
        })
    return rows


def write_clip_summary(output_dir: str, rows: List[dict]) -> Optional[dict]:
    """clips_summary.json/csv 와 selected_clips.json 을 출력 폴더에 저장.

    반환: {"total", "selected", "csv", "json", "selected_json"} 또는 행이 없으면 None.
    """
    if not rows:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "clips_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out / "clips_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    selected_rows = [r for r in rows if r.get("selected") is True]
    selected_path = out / "selected_clips.json"
    selected_path.write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "total": len(rows),
        "selected": len(selected_rows),
        "csv": str(csv_path),
        "json": str(json_path),
        "selected_json": str(selected_path),
    }
