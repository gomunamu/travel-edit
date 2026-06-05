"""전체 편집 파이프라인 조율"""
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import threading

import config as _config
from config import (
    METADATA_WORKERS, TRANSCRIBE_WORKERS,
    CLAUDE_MAX_CONCURRENT, LOCATION_DISPLAY_DURATION, LOCATION_FADE_DURATION,
    SUBTITLE_FONT, SUBTITLE_FONT_SIZE, SUBTITLE_MARGIN_V,
    LOCATION_FONT_SIZE, LOCATION_MARGIN,
)
from core.token_tracker import tracker as _token_tracker
from core.cache import Cache, make_clip_hash, variant_tag
from core.metadata import get_video_info, is_video
from core.segmenter import plan_segments
from core.transcriber import transcribe, init_model_pool, get_pool_size, release_model_pool
from core.refiner import refine_transcript
from core.evaluator import evaluate_clip
from core.geocoder import coords_to_str, get_location_hints
from core.subtitle import make_subtitle_ass, make_subtitle_srt, make_location_ass
from core.evaluator import _adaptive as _eval_adaptive
from core.renderer import get_day_resolution, get_day_fps, render_day_onepass, is_valid_video

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _tqdm(iterable, **kwargs):
    if HAS_TQDM:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    return iterable


# ─── 1. 스캔 ──────────────────────────────────────────────────────────────
def scan_videos(input_folder: str) -> List[str]:
    # rglob 이터레이터를 바로 tqdm에 넘겨 발견 즉시 카운트 표시
    # (sorted()로 전체를 모으면 그동안 아무것도 안 보임)
    videos = []
    for p in _tqdm(Path(input_folder).rglob("*"), desc="  스캔", unit="파일"):
        if p.is_file() and is_video(str(p)):
            videos.append(str(p))
    videos.sort()
    return videos


# ─── 2. 메타데이터 추출 (병렬) ─────────────────────────────────────────────
def extract_all_metadata(video_files: List[str], cache: Cache) -> List[dict]:
    def _extract(fp):
        clip_hash = make_clip_hash(fp)
        cached = cache.load(clip_hash, "meta")
        if cached:
            return cached

        info = get_video_info(fp)
        if not info:
            return None

        info["clip_hash"] = clip_hash
        info["trim_start"] = 0.0
        info["trim_end"] = info["duration"]
        info["segment_index"] = 0
        info["parent_hash"] = None
        cache.save(clip_hash, "meta", info)
        return info

    results = []
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as ex:
        futures = {ex.submit(_extract, fp): fp for fp in video_files}
        for future in _tqdm(as_completed(futures), total=len(futures), desc="  메타데이터"):
            r = future.result()
            if r:
                results.append(r)

    # 촬영 시간 순 정렬 (creation_time은 항상 존재 — fallback 보장)
    def sort_key(c):
        try:
            return datetime.fromisoformat(c["creation_time"])
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    results.sort(key=sort_key)

    # 메타데이터 날짜 없이 fallback 사용한 파일 안내
    fallback_files = [r for r in results if r.get("creation_time_source", "metadata") != "metadata"]
    if fallback_files:
        print(f"  [알림] {len(fallback_files)}개 파일은 날짜 메타데이터 없음 → 경로/파일명/mtime 으로 날짜 추정")
        for r in fallback_files:
            print(f"    {r['filename']}  →  {r['day_key']} ({r.get('creation_time_source', 'path/filename/mtime')})")

    return results


# ─── 3. 세그먼트 계획 (파일 추출 없음, JSON 캐시) ────────────────────────────
def segment_all(clips: List[dict], cache: Cache) -> List[dict]:
    """
    클립을 논리 세그먼트로 분할한다.
    계획 결과는 {cache}/.segment_plan.json 에 저장되어
    재시작 시 바로 불러온다. 입력 클립이 바뀌면 자동으로 재계획.
    """
    plan_path = cache.root / ".segment_plan.json"

    # 클립 목록 지문 — clip_hash(파일 경로+mtime+size) + 분할 설정까지 포함
    # 파일 변경 또는 MAX/MIN_SEGMENT_DURATION 변경 시 자동 재계획
    _seg_settings = f"|max={_config.MAX_SEGMENT_DURATION}|min={_config.MIN_SEGMENT_DURATION}"
    fingerprint = hashlib.sha256(
        ("|".join(c["clip_hash"] for c in clips) + _seg_settings).encode()
    ).hexdigest()

    # 기존 계획 로드
    if plan_path.exists():
        try:
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint:
                segs = saved["segments"]
                print(f"  → {len(segs)}개 세그먼트 (계획 파일 로드)")
                return segs
        except (json.JSONDecodeError, KeyError):
            pass  # 손상됐으면 재계획

    # 새로 계획 수립
    all_segs: List[dict] = []
    n_split = 0
    for clip in clips:
        if clip["duration"] <= 0:
            continue
        segs = plan_segments(clip)
        if len(segs) > 1:
            n_split += 1
        for seg in segs:
            if seg.get("parent_hash") and not cache.load(seg["clip_hash"], "meta"):
                cache.save(seg["clip_hash"], "meta", seg)
            all_segs.append(seg)

    if n_split:
        print(f"  {n_split}개 클립 → 복수 세그먼트로 논리 분할")
    print(f"  → 총 {len(all_segs)}개 세그먼트")

    # 계획 저장
    plan_path.write_text(
        json.dumps({"fingerprint": fingerprint, "segments": all_segs},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return all_segs


# ─── 4. 음성 인식 (모델 풀 병렬) ─────────────────────────────────────────
def transcribe_all(segments: List[dict], cache: Cache) -> Dict[str, dict]:
    # 캐시 키: whisper 모델 + 언어 설정이 바뀌면 재전사
    _t_suffix = "transcript_" + variant_tag(_config.WHISPER_MODEL, _config.SUBTITLE_LANG)

    transcripts = {}
    need_transcribe = []

    for seg in segments:
        h = seg["clip_hash"]
        cached = cache.load(h, _t_suffix)
        if cached:
            transcripts[h] = cached
        else:
            need_transcribe.append(seg)

    if not need_transcribe:
        return transcripts

    print(f"  음성 인식 필요: {len(need_transcribe)}개 (캐시 {len(transcripts)}개 재사용)")

    # 모델 풀 초기화 (TRANSCRIBE_WORKERS 수만큼 시도, VRAM 부족 시 자동 축소)
    init_model_pool(TRANSCRIBE_WORKERS)
    actual_workers = get_pool_size()  # 실제 로드된 인스턴스 수로 스레드 수 맞춤

    lock = threading.Lock()
    active: dict = {}   # thread_ident → 표시 이름
    active_lock = threading.Lock()

    _lang_map = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}
    force_lang = _lang_map.get(_config.SUBTITLE_LANG)  # None = auto

    if HAS_TQDM:
        from tqdm import tqdm as _tqdm_cls
        pbar = _tqdm_cls(total=len(need_transcribe), desc="  음성 인식")
    else:
        pbar = None

    def _transcribe_one(seg):
        src_start = seg.get("_src_start", 0.0)
        dur = seg.get("duration", 0)
        # 현재 처리 중인 파일을 프로그레스 바 postfix에 표시
        label = f"{Path(seg['filepath']).stem}  {src_start:.0f}~{src_start+dur:.0f}s"
        tid = threading.current_thread().ident
        with active_lock:
            active[tid] = label
            if pbar:
                pbar.set_postfix_str(" | ".join(active.values()), refresh=True)

        h = seg["clip_hash"]
        result = transcribe(seg["filepath"], start=src_start,
                            duration=dur, force_lang=force_lang)
        cache.save(h, _t_suffix, result)

        with active_lock:
            active.pop(tid, None)
            transcripts[h] = result
        if pbar:
            pbar.update(1)
            with active_lock:
                pbar.set_postfix_str(" | ".join(active.values()), refresh=True)

    with ThreadPoolExecutor(max_workers=actual_workers) as ex:
        futures = [ex.submit(_transcribe_one, seg) for seg in need_transcribe]
        for future in as_completed(futures):
            future.result()

    if pbar:
        pbar.close()

    return transcripts


# ─── 4-b. STT 정제 (LLM 교정) ─────────────────────────────────────────────
def refine_all(
    segments: List[dict],
    transcripts: Dict[str, dict],
    cache: Cache,
) -> Dict[str, dict]:
    """
    Whisper 결과를 LLM으로 한 번 더 정제한다.
    폴백 체인: Claude → OpenAI → Gemini.
    캐시 키: STT_REFINE_MODEL이 바뀌면 재정제.
    """
    from config import STT_REFINE_MODEL, CLAUDE_MAX_CONCURRENT

    # 캐시 키: 정제 모델이 바뀌면 새 슬롯 사용
    _r_suffix = "transcript_refined_" + variant_tag(STT_REFINE_MODEL)

    refined_map: Dict[str, dict] = {}
    need_refine = []

    for seg in segments:
        h = seg["clip_hash"]
        if h not in transcripts:
            continue
        cached = cache.load(h, _r_suffix)
        if cached:
            refined_map[h] = cached
        else:
            need_refine.append(seg)

    if not need_refine:
        print(f"  → 전체 {len(refined_map)}개 캐시 재사용 (STT 정제 건너뜀)")
        for h, t in transcripts.items():
            if h not in refined_map:
                refined_map[h] = t
        return refined_map

    print(f"  STT 정제 필요: {len(need_refine)}개 (캐시 {len(refined_map)}개 재사용)")

    lock = threading.Lock()
    semaphore = threading.Semaphore(CLAUDE_MAX_CONCURRENT)

    def _refine_one(seg):
        h = seg["clip_hash"]
        original = transcripts[h]
        # GPS에서 지역명 힌트 추출 (국가 제외)
        hints = []
        gps = seg.get("gps")
        if gps and len(gps) >= 2:
            hints = get_location_hints(float(gps[0]), float(gps[1]))
        try:
            with semaphore:
                result = refine_transcript(original, location_hints=hints or None)
        except Exception as e:
            print(f"\n  [경고] STT 정제 실패 ({Path(seg['filepath']).name}): {e}")
            result = original  # 실패해도 원본으로 캐시 저장 → 재실행 시 재시도 안 함
        cache.save(h, _r_suffix, result)
        with lock:
            refined_map[h] = result

    with ThreadPoolExecutor(max_workers=CLAUDE_MAX_CONCURRENT) as ex:
        futures = [ex.submit(_refine_one, seg) for seg in need_refine]
        for future in _tqdm(as_completed(futures), total=len(futures), desc="  STT 정제"):
            future.result()

    # 정제 대상 아닌 세그먼트는 원본 사용
    for h, t in transcripts.items():
        if h not in refined_map:
            refined_map[h] = t

    return refined_map


# ─── 4-c. 자막 / 위치 미리보기 출력 ──────────────────────────────────────
def print_clip_preview(segments: List[dict], transcripts: Dict[str, dict]) -> None:
    """
    STT 정제 완료 후 클립별 최종 자막 텍스트와 위치 정보를 출력한다.
    위치는 이전 클립과 달라질 때만 표시.
    """
    prev_location: Optional[str] = None
    print()

    for seg in segments:
        h = seg["clip_hash"]
        transcript = transcripts.get(h)

        # 위치 (변경 시에만)
        location: Optional[str] = None
        if seg.get("gps"):
            location = coords_to_str(seg["gps"])
        if location and location != prev_location:
            print(f"  📍 {location}")
            prev_location = location

        # 자막 텍스트 수집
        speech_lines = []
        if transcript:
            for s in transcript.get("segments", []):
                text = s.get("text", "").strip()
                if text and s.get("no_speech_prob", 1.0) < 0.5:
                    speech_lines.append(text)

        filename = Path(seg["filename"]).stem
        src_start = seg.get("_src_start", 0.0)
        dur = seg.get("duration", 0.0)
        label = f"  [{filename}  {src_start:.0f}~{src_start+dur:.0f}s]"

        if speech_lines:
            # 첫 줄은 label과 같은 줄에, 나머지는 들여쓰기
            print(f"{label} {speech_lines[0]}")
            for line in speech_lines[1:]:
                print(f"{'':>{len(label)+1}}{line}")
        # 무음 클립은 출력 생략


# ─── 5. AI 평가 (병렬 - 폴백 체인: Claude → OpenAI → Gemini → 규칙 기반) ──
def evaluate_all(
    segments: List[dict],
    transcripts: Dict[str, dict],
    cache: Cache,
) -> Dict[str, dict]:
    # 캐시 키: EDIT_STYLE이 바뀌면 재평가 (같은 클립도 스타일에 따라 결과가 다름)
    _e_suffix = "eval_" + variant_tag(_config.EDIT_STYLE)

    evaluations = {}
    need_eval = []

    for seg in segments:
        h = seg["clip_hash"]
        cached = cache.load(h, _e_suffix)
        if cached is not None:
            evaluations[h] = cached
        else:
            need_eval.append(seg)

    if need_eval:
        print(f"  AI 평가 필요: {len(need_eval)}개 (캐시 {len(evaluations)}개 재사용, 스타일={_config.EDIT_STYLE})")

    lock = threading.Lock()

    def _eval_one(seg):
        h = seg["clip_hash"]
        transcript = transcripts.get(h, {"segments": [], "has_speech": False, "total_speech_sec": 0})
        result = evaluate_clip(seg, transcript)
        try:
            cache.save(h, _e_suffix, result)
        except Exception as e:
            print(f"  [경고] 평가 캐시 저장 실패 ({seg.get('filename', h)}): {e}")
        with lock:
            evaluations[h] = result
        return h, result

    # 소스 파일 단위로 그룹화해서 평가
    # 파일 사이에 API 상태를 리셋 → 이전 파일에서 rate limit으로 비활성화된 API를
    # 다음 파일에서 다시 시도 (처리 시간이 길어 rate limit이 회복된다고 가정)
    from core.evaluator import reset_api_state

    file_groups: dict = OrderedDict()
    for seg in need_eval:
        fname = seg.get("filename", "")
        file_groups.setdefault(fname, []).append(seg)

    total = len(need_eval)
    if HAS_TQDM:
        from tqdm import tqdm as _tqdm_cls
        pbar = _tqdm_cls(total=total, desc="  AI 평가")
    else:
        pbar = None

    for fname, file_segs in file_groups.items():
        reset_api_state()
        with ThreadPoolExecutor(max_workers=50) as ex:
            futures = [ex.submit(_eval_one, seg) for seg in file_segs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  [경고] 평가 작업 실패 ({fname}): {e}")
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    return evaluations


# ─── 6. 일자별 렌더링 ─────────────────────────────────────────────────────
def render_day(
    day_key: str,
    day_segments: List[dict],
    transcripts: Dict[str, dict],
    evaluations: Dict[str, dict],
    cache: Cache,
    output_path: str,
    selection_info: Optional[Dict[str, dict]] = None,
) -> bool:
    """
    하루치 세그먼트를 평가 결과에 따라 선별하고,
    filter_complex 로 한 번에 트림·스케일·자막·병합 (중간 파일 없음).
    """
    out_res = get_day_resolution(day_segments)
    out_fps = get_day_fps(day_segments)

    def _make_clip(seg, ev):
        clip = dict(seg)
        decision = ev.get("decision", "keep")
        if decision == "trim":
            clip["trim_start"] = float(ev.get("keep_start", 0))
            clip["trim_end"]   = float(ev.get("keep_end", seg["duration"]))
        else:
            clip["trim_start"] = 0.0
            clip["trim_end"]   = seg["duration"]
        clip["eval"] = ev
        return clip

    def _print_clip(clip, tag):
        ev = clip["eval"]
        sc = ev.get("score", {})
        total  = sc.get("total",  "?")
        visual = sc.get("visual", "-")
        speech = sc.get("speech", "-")
        scene  = sc.get("scene",  "-")
        flow   = sc.get("flow",   "-")
        print(
            f"    [{tag} {total:>3}점"
            f"  시각{visual}/음성{speech}/장면{scene}/흐름{flow}] "
            f"{clip['filename']} {clip['trim_start']:.1f}~{clip['trim_end']:.1f}s"
            f"  {ev.get('reason','')}"
        )

    selected = []   # 평가 통과
    discarded = []  # 버림 (min_day_duration 구제 후보)

    for seg in day_segments:
        h = seg["clip_hash"]
        ev = evaluations.get(h, {"decision": "keep", "keep_start": 0, "keep_end": seg["duration"]})
        decision = ev.get("decision", "keep")

        if decision == "discard":
            sc = ev.get("score", {})
            total = sc.get("total", "?")
            print(f"    [버림 {total:>3}점] {seg['filename']} - {ev.get('reason','')}")
            discarded.append((seg, ev))
            continue

        clip = _make_clip(seg, ev)
        selected.append(clip)
        _print_clip(clip, "트림" if decision == "trim" else "살림")

    # ── 최소 분량 보장 ─────────────────────────────────────────────────────
    min_dur = getattr(_config, "MIN_DAY_DURATION", 0)  # seconds
    if min_dur > 0:
        selected_dur = sum(c["trim_end"] - c["trim_start"] for c in selected)
        if selected_dur < min_dur:
            # 단계 1: 버린 클립을 고득점 순으로 추가
            discarded_sorted = sorted(
                discarded,
                key=lambda x: x[1].get("score", {}).get("total", 0),
                reverse=True,
            )
            rescued = []
            for seg, ev in discarded_sorted:
                if selected_dur >= min_dur:
                    break
                clip = _make_clip(seg, ev)
                rescued.append(clip)
                selected_dur += clip["trim_end"] - clip["trim_start"]

            # 단계 2: 그래도 부족하면 남은 버린 클립 전부 추가
            if selected_dur < min_dur:
                rescued_hashes = {c["clip_hash"] for c in rescued}
                for seg, ev in discarded_sorted:
                    if seg["clip_hash"] not in rescued_hashes:
                        clip = _make_clip(seg, ev)
                        rescued.append(clip)
                        rescued_hashes.add(seg["clip_hash"])

            if rescued:
                mode = "전체 포함" if selected_dur < min_dur else "고득점 구제"
                print(f"  [최소분량] {min_dur//60}분 미달 → {mode}: {len(rescued)}개 클립 추가")
                for clip in rescued:
                    _print_clip(clip, "구제")
                selected.extend(rescued)
                # 촬영 시간순 재정렬
                selected.sort(key=lambda c: c.get("creation_time", ""))

    if not selected:
        if selection_info is not None:
            for seg in day_segments:
                selection_info[seg["clip_hash"]] = {
                    "selected": False, "location": "",
                    "trim_start": 0.0, "trim_end": seg.get("duration", 0.0),
                    "speed": 1.0,
                }
        print(f"  → {day_key}: 선택된 클립 없음, 건너뜀")
        return False

    # ── GPS 전파: GPS 없는 클립에 인접 클립의 GPS 보완 ────────────────────────
    # 대부분의 영상(iPhone/DJI 등)은 GPS EXIF가 파일에 없을 수 있다.
    # GPS가 없으면 지역명 오버레이 불가 → 같은 날 촬영 클립끼리
    # forward/backward 패스로 보완한다 (selected는 이미 creation_time 정렬됨).
    if any(c.get("gps") for c in selected):
        # 1) Forward pass: 앞 클립의 GPS를 뒤로 전파
        last_gps = None
        for clip in selected:
            if clip.get("gps"):
                last_gps = clip["gps"]
            elif last_gps:
                clip["gps"] = last_gps
                clip["_gps_interpolated"] = True
        # 2) Backward pass: 앞쪽에 GPS가 없었던 클립을 뒤에서 앞으로 채움
        last_gps = None
        for clip in reversed(selected):
            if clip.get("gps"):
                last_gps = clip["gps"]
            elif last_gps:
                clip["gps"] = last_gps
                clip["_gps_interpolated"] = True

    # 무음 컷 배속 결정 (음성/자막 있는 컷은 항상 1.0배 → 자막 동기화 보존)
    _silent_speed = getattr(_config, "SILENT_SPEEDUP", 1.0)
    for clip in selected:
        has_sp = transcripts.get(clip["clip_hash"], {}).get("has_speech", False)
        clip["speed"] = _silent_speed if (_silent_speed > 1.0 and not has_sp) else 1.0

    # 장소 오버레이 (최종 순서 확정 후 적용)
    # carry-forward: 클립이 짧아서 오버레이가 충분히 표시되지 못하면
    # 같은 장소의 다음 클립으로 오버레이를 이전한다.
    # clip_dur 는 배속 반영한 '출력' 길이 기준으로 계산한다.
    _min_overlay_sec = LOCATION_DISPLAY_DURATION  # 이 시간 미만이면 다음 클립으로 넘김
    prev_location = None
    location_remaining = 0.0  # 아직 표시해야 할 오버레이 잔여 시간
    for clip in selected:
        location_name = None
        if clip.get("gps"):
            location_name = coords_to_str(clip["gps"])
        clip_dur = (clip["trim_end"] - clip["trim_start"]) / clip.get("speed", 1.0)

        if location_name and location_name != prev_location:
            # 새 장소 진입
            clip["show_location"] = location_name
            prev_location = location_name
            location_remaining = max(0.0, _min_overlay_sec - clip_dur)
        elif location_remaining > 0 and location_name == prev_location:
            # 같은 장소인데 이전 클립이 너무 짧아서 오버레이를 이어서 표시
            clip["show_location"] = prev_location
            location_remaining = max(0.0, location_remaining - clip_dur)
        else:
            clip["show_location"] = None
            location_remaining = 0.0

    # 클립 요약용 선택 정보 기록 (최종 선택 + 위치 확정 후)
    if selection_info is not None:
        sel_meta = {
            c["clip_hash"]: (
                coords_to_str(c["gps"]) if c.get("gps") else "",
                c["trim_start"], c["trim_end"], c.get("speed", 1.0),
            )
            for c in selected
        }
        for seg in day_segments:
            h = seg["clip_hash"]
            if h in sel_meta:
                loc, ts, te, sp = sel_meta[h]
                selection_info[h] = {"selected": True, "location": loc,
                                     "trim_start": ts, "trim_end": te, "speed": sp}
            else:
                selection_info[h] = {"selected": False, "location": "",
                                     "trim_start": 0.0, "trim_end": seg.get("duration", 0.0),
                                     "speed": 1.0}

    print(f"  출력 해상도: {out_res[0]}x{out_res[1]}, {out_fps}fps, {len(selected)}개 클립")

    # ASS 파일 생성 (자막·장소) — 캐시 활용
    clips_info = []
    for clip in selected:
        h = clip["clip_hash"]
        trim_start = clip["trim_start"]
        trim_end   = clip["trim_end"]
        speed      = clip.get("speed", 1.0)
        clip_dur   = (trim_end - trim_start) / speed   # 배속 반영 출력 길이

        # 자막 ASS (overlay 모드)
        sub_path = None
        if _config.SUBTITLE_MODE == "overlay":
            transcript = transcripts.get(h, {})
            segs = transcript.get("segments", [])
            if segs and transcript.get("has_speech"):
                sub_path = cache.ass_path(h, "subtitle")
                if not cache.ass_exists(h, "subtitle"):
                    adjusted = [
                        dict(s, start=s["start"] - trim_start,
                                end=s["end"]   - trim_start)
                        for s in segs if s["end"] > trim_start
                    ]
                    make_subtitle_ass(
                        adjusted, sub_path, out_res,
                        font=SUBTITLE_FONT, font_size=SUBTITLE_FONT_SIZE,
                        margin_v=SUBTITLE_MARGIN_V,
                    )

        # 장소 ASS (배속 시 출력 길이가 달라지므로 캐시 태그에 배속 반영)
        loc_path = None
        loc_name = clip.get("show_location")
        if loc_name:
            _loc_tag = "location_r" if speed == 1.0 else f"location_r_s{speed:g}"
            loc_path = cache.ass_path(h, _loc_tag)
            if not cache.ass_exists(h, _loc_tag):
                make_location_ass(
                    loc_name, clip_dur, loc_path, out_res,
                    display_duration=LOCATION_DISPLAY_DURATION,
                    fade_duration=LOCATION_FADE_DURATION,
                    font=SUBTITLE_FONT, font_size=LOCATION_FONT_SIZE,
                    margin=LOCATION_MARGIN,
                )

        clips_info.append({**clip, "sub_path": sub_path, "loc_path": loc_path})

    # ── 얼굴 모자이크: 렌더링 전 컷 클립 단위 처리 ──────────────────────────────
    # 가족 제외 옵션 시: 1단계(임베딩 추출) → 가족 판별 → 2단계(모자이크 적용)
    # 가족 제외 미사용 시: 단순 모자이크 적용
    if getattr(_config, "FACE_MOSAIC", False):
        from core.mosaic import apply_face_mosaic, extract_clip_embeddings, is_korea
        korea_only = getattr(_config, "FACE_MOSAIC_KOREA_ONLY", False)
        if not korea_only or is_korea(selected):
            from tve.tier import detect as _detect_tier
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _tier   = _detect_tier()
            use_gpu = _tier.tier.name in ("A", "B")
            codec_m = getattr(_config, "VIDEO_CODEC", "libx265")
            crf_m   = getattr(_config, "CRF", 23)
            cpu_cnt = os.cpu_count() or 4
            if use_gpu:
                try:
                    # torch.cuda.mem_get_info() 는 CUDA context를 새로 초기화하면서
                    # 이전 run 의 잔여 CUDA 상태와 충돌해 무한 대기할 수 있음.
                    # nvidia-smi subprocess 로 대체 (timeout=5, CUDA context 불필요)
                    _r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.free",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    )
                    free_mb = (int(_r.stdout.strip().split("\n")[0].strip())
                               if _r.returncode == 0 else 0)
                    workers = max(4, min(10, (free_mb - 3072) // 150))
                except Exception:
                    workers = 6
                detect_interval_m = 3
            else:
                workers = max(1, cpu_cnt // 4)
                detect_interval_m = 5

            n_clips = len(clips_info)
            family_embs = None
            family_exclude = getattr(_config, "FACE_MOSAIC_FAMILY_EXCLUDE", False)

            # ── 1단계: 임베딩 추출 (가족 제외 옵션 시) ──────────────────────────
            if family_exclude:
                # 임베딩 단계는 디코딩+GPU 추론 혼합 — 모자이크보다 CPU 비율 높음
                # workers // 2 로 제한해 CPU 코어 포화 방지
                emb_workers = max(2, workers // 2)
                print(f"\n[얼굴인식] 임베딩 시작: {n_clips}개 클립 "
                      f"({'GPU' if use_gpu else 'CPU'}, 병렬 {emb_workers}개)")
                all_embs: list = []
                emb_lock  = threading.Lock()
                emb_done  = [0]

                def _embed_one(i_clip):
                    i, clip = i_clip
                    t_start = clip.get("_src_start", 0.0) + clip.get("trim_start", 0.0)
                    t_end   = clip.get("_src_start", 0.0) + clip.get("trim_end", clip["duration"])
                    embs = extract_clip_embeddings(
                        clip["filepath"], use_gpu=use_gpu,
                        detect_interval=detect_interval_m,
                        trim_start=t_start, trim_end=t_end,
                    )
                    with emb_lock:
                        all_embs.extend(embs)
                        emb_done[0] += 1
                        cnt = emb_done[0]
                    fname = Path(clip["filepath"]).name
                    print(f"[얼굴인식] [{cnt}/{n_clips}] 임베딩 {len(embs)}개: {fname}")

                with ThreadPoolExecutor(max_workers=emb_workers) as ex:
                    futs = {ex.submit(_embed_one, (i, c)): i
                            for i, c in enumerate(clips_info, 1)}
                    for fut in as_completed(futs):
                        try:
                            fut.result()
                        except Exception as exc:
                            print(f"  [임베딩] 오류: {exc}")

                # 가족 판별
                from core.face_profile import identify_family_from_embeddings
                family_embs = identify_family_from_embeddings(all_embs) or None
                fam_cnt = len(family_embs) if family_embs else 0
                print(f"[얼굴인식] 가족 {fam_cnt}명 확정 → 모자이크 2단계 진행")

            # ── 2단계: 모자이크 적용 ─────────────────────────────────────────────
            done_count = 0
            done_lock  = threading.Lock()
            # 가족 제외 + 결과 있을 때만 별도 캐시 suffix (재실행 시 혼용 방지)
            cache_suffix = "_mosaic_fam.mp4" if family_embs else "_mosaic.mp4"

            fam_note = f", 가족 {len(family_embs)}명 제외" if family_embs else ""
            print(f"\n[얼굴인식] 얼굴 모자이크 시작: {n_clips}개 클립 "
                  f"({'GPU' if use_gpu else 'CPU'}, 병렬 {workers}개, "
                  f"detect_interval={detect_interval_m}{fam_note})")

            def _mosaic_one(clip_idx_clip):
                idx, clip = clip_idx_clip
                nonlocal done_count
                h = clip["clip_hash"]
                t_start = clip.get("_src_start", 0.0) + clip.get("trim_start", 0.0)
                t_end   = clip.get("_src_start", 0.0) + clip.get("trim_end", clip["duration"])
                mosaic_path = str(cache.root / f"{h}{cache_suffix}")
                fname = Path(clip["filepath"]).name

                if Path(mosaic_path).exists():
                    print(f"[얼굴인식] [{idx}/{n_clips}] 캐시 재사용: {fname}")
                    ok_m = True
                else:
                    print(f"[얼굴인식] [{idx}/{n_clips}] {fname} ({t_start:.1f}~{t_end:.1f}초)")
                    ok_m = apply_face_mosaic(
                        clip["filepath"], mosaic_path,
                        use_gpu=use_gpu, codec=codec_m, crf=crf_m,
                        trim_start=t_start, trim_end=t_end,
                        detect_interval=detect_interval_m,
                        family_embeddings=family_embs,
                    )

                with done_lock:
                    done_count += 1
                    cnt = done_count
                print(f"[얼굴인식] [{cnt}/{n_clips}]")

                return idx, ok_m, mosaic_path, t_start, t_end

            results = {}
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_mosaic_one, (i, c)): i
                        for i, c in enumerate(clips_info, 1)}
                for fut in as_completed(futs):
                    try:
                        idx, ok_m, mosaic_path, t_start, t_end = fut.result()
                        results[idx] = (ok_m, mosaic_path, t_start, t_end)
                    except Exception as exc:
                        i = futs[fut]
                        print(f"  [모자이크] 클립 {i} 오류: {exc}")

            for i, clip in enumerate(clips_info, 1):
                if i not in results:
                    continue
                ok_m, mosaic_path, t_start, t_end = results[i]
                if ok_m and Path(mosaic_path).exists():
                    clip["filepath"]   = mosaic_path
                    clip["_src_start"] = 0.0
                    clip["trim_start"] = 0.0
                    clip["trim_end"]   = t_end - t_start

            print(f"[얼굴인식] 완료 [{n_clips}/{n_clips}]")

    print(f"  렌더링+병합 → {Path(output_path).name}")
    ok = render_day_onepass(clips_info, output_path, out_res, out_fps)

    # SRT 모드: 병합 성공 후 타임라인 맞춰 단일 SRT 파일 생성
    if ok and _config.SUBTITLE_MODE == "srt":
        cumulative = 0.0
        srt_segs = []
        for clip in clips_info:
            h = clip["clip_hash"]
            transcript = transcripts.get(h, {})
            segs = transcript.get("segments", [])
            trim_start = clip["trim_start"]
            clip_dur   = clip["trim_end"] - trim_start
            if segs and transcript.get("has_speech"):
                for seg in segs:
                    if seg.get("no_speech_prob", 1.0) < 0.5 and seg.get("text", "").strip():
                        start = max(0.0, seg["start"] - trim_start) + cumulative
                        end   = min(max(start + 0.5, seg["end"] - trim_start + cumulative),
                                    cumulative + clip_dur)
                        srt_segs.append({"start": start, "end": end,
                                         "text": seg["text"].strip()})
            cumulative += clip_dur
        if srt_segs:
            srt_path = output_path.replace(".mp4", ".srt")
            make_subtitle_srt(srt_segs, srt_path)
            print(f"  ✓ SRT 자막: {Path(srt_path).name} ({len(srt_segs)}개)")

    return ok


# ─── 전체 파이프라인 ──────────────────────────────────────────────────────
def run(input_folder: str, output_folder: str):
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache = Cache(str(cache_dir))

    from config import OUTPUT_RESOLUTION
    if OUTPUT_RESOLUTION is None:
        res_str = "자동 (원본 최고 해상도 기준)"
    else:
        res_str = f"{OUTPUT_RESOLUTION[0]}x{OUTPUT_RESOLUTION[1]}"
    orient_str = "가로/세로 분리" if _config.SPLIT_ORIENTATION else "혼합"
    print(f"\n{'='*60}")
    print(f"  여행 영상 자동 편집기")
    print(f"  입력: {input_folder}")
    print(f"  출력: {output_folder}")
    print(f"  해상도: {res_str}")
    print(f"  방향: {orient_str}")
    print(f"{'='*60}\n")

    run_start = time.time()
    file_report: List[dict] = []   # {"name", "elapsed", "size_mb"}
    selection_info: Dict[str, dict] = {}   # render_day가 채움 → 클립 요약 리포트용

    # 1. 스캔
    print("[1/6] 비디오 파일 스캔...")
    videos = scan_videos(input_folder)
    print(f"  → {len(videos)}개 비디오 발견")
    if not videos:
        print("  비디오 파일을 찾을 수 없습니다.")
        return

    # 2. 메타데이터
    print("\n[2/6] 메타데이터 추출...")
    clips = extract_all_metadata(videos, cache)
    print(f"  → {len(clips)}개 처리됨")

    # 3. 세그먼트 계획
    print("\n[3/6] 세그먼트 계획 수립...")
    segments = segment_all(clips, cache)

    # 4. 음성 인식
    print("\n[4/6] 음성 인식 (Whisper)...")
    if _config.SUBTITLE_LANG == "off":
        print("  → 자막 비활성화, 건너뜀")
        transcripts = {}
    else:
        transcripts = transcribe_all(segments, cache)
        release_model_pool()   # GPU 메모리 해제 → 렌더링에서 활용 가능

        # 4-b. STT 정제
        if _config.STT_REFINE:
            _refine_keys = [
                k for k, v in [
                    ("Claude",  _config.ANTHROPIC_API_KEY),
                    ("OpenAI",  _config.OPENAI_API_KEY),
                    ("Gemini",  _config.GEMINI_API_KEY),
                ] if v
            ]
            _refine_label = " → ".join(_refine_keys + ["CPU(반복제거)"])
            print(f"\n[4-b] STT 정제 (폴백 체인: {_refine_label})...")
            transcripts = refine_all(segments, transcripts, cache)
            _token_tracker.print_current("4-b STT 정제 후")

        # 4-c. 자막 / 위치 미리보기
        print("\n[4-c] 자막 미리보기 (최종):")
        print_clip_preview(segments, transcripts)

    # 5. AI 평가 (폴백 체인: Claude → OpenAI → Gemini → 규칙 기반)
    _eval_keys = [
        k for k, v in [
            ("Claude",  _config.ANTHROPIC_API_KEY),
            ("OpenAI",  _config.OPENAI_API_KEY),
            ("Gemini",  _config.GEMINI_API_KEY),
        ] if v
    ]
    _eval_label = " → ".join(_eval_keys + ["규칙 기반"])
    print(f"\n[5/6] AI 클립 평가 (폴백 체인: {_eval_label})...")
    evaluations = evaluate_all(segments, transcripts, cache)
    _token_tracker.print_current("5단계 AI 평가 후")

    # 6. 일자별 렌더링
    print("\n[6/6] 일자별 편집 및 렌더링...")
    if _config.SPLIT_ORIENTATION:
        print("  [방향 분리] 가로/세로 영상을 별도 파일로 출력")

    # 날짜별 (+ 방향별) 그룹화
    # key: (day_key, suffix)  suffix = "" | "_vertical"
    day_groups: dict = defaultdict(list)
    for seg in segments:
        day_key = seg.get("day_key", "unknown")
        if _config.SPLIT_ORIENTATION:
            suffix = "_vertical" if seg.get("is_portrait") else ""
        else:
            suffix = ""
        day_groups[(day_key, suffix)].append(seg)

    for (day_key, suffix) in sorted(day_groups.keys()):
        day_segs = day_groups[(day_key, suffix)]
        orientation_label = " [세로]" if suffix == "_vertical" else (" [가로]" if _config.SPLIT_ORIENTATION else "")
        print(f"\n  ── {day_key}{orientation_label} ({len(day_segs)}개 클립) ──")

        output_path = str(out_dir / f"travel_{day_key}{suffix}.mp4")

        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / 1_048_576
            print(f"  → 이미 존재: {output_path} (건너뜀, {size_mb:.1f} MB)")
            file_report.append({"name": Path(output_path).name, "path": output_path, "elapsed": 0.0, "size_mb": size_mb, "skipped": True})
            continue

        day_start = time.time()
        ok = render_day(
            day_key, day_segs, transcripts, evaluations,
            cache, output_path, selection_info=selection_info
        )
        if ok:
            elapsed = time.time() - day_start
            final_path = output_path
            archive_dir = getattr(_config, "ARCHIVE_DIR", None)
            if archive_dir:
                arch_dir = Path(archive_dir)
                arch_dir.mkdir(parents=True, exist_ok=True)
                dest = arch_dir / Path(output_path).name
                shutil.move(output_path, dest)
                final_path = str(dest)
                # SRT가 있으면 같이 이동
                srt_src = Path(output_path.replace(".mp4", ".srt"))
                if srt_src.exists():
                    shutil.move(str(srt_src), str(arch_dir / srt_src.name))
                print(f"  → 아카이브 이동: {dest}")
            size_mb = Path(final_path).stat().st_size / 1_048_576
            file_report.append({"name": Path(final_path).name, "path": final_path, "elapsed": elapsed, "size_mb": size_mb, "skipped": False})
            print(f"  ✓ 완료: {final_path} ({size_mb:.1f} MB, {elapsed:.0f}초)")

    total_elapsed = time.time() - run_start
    total_size_mb  = sum(r["size_mb"] for r in file_report)

    # ── 요약 문자열 빌드 ──────────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  편집 완료! 출력 폴더: {output_folder}")
    lines.append(f"{'='*60}")

    if file_report:
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {'파일명':<35} {'작업시간':>10}  {'용량':>10}")
        lines.append(f"{'─'*60}")
        for r in file_report:
            if r["skipped"]:
                time_str = "(재사용)"
            else:
                m, s = divmod(int(r["elapsed"]), 60)
                h2, m2 = divmod(m, 60)
                if h2:
                    time_str = f"{h2}시간 {m2}분 {s:02d}초"
                elif m:
                    time_str = f"{m}분 {s:02d}초"
                else:
                    time_str = f"{s}초"
            lines.append(f"  {r['name']:<35} {time_str:>10}  {r['size_mb']:>8.1f} MB")
        lines.append(f"{'─'*60}")
        tm, ts = divmod(int(total_elapsed), 60)
        th, tm2 = divmod(tm, 60)
        if th:
            total_time_str = f"{th}시간 {tm2}분 {ts:02d}초"
        elif tm:
            total_time_str = f"{tm}분 {ts:02d}초"
        else:
            total_time_str = f"{ts}초"
        lines.append(f"  {'합계':<35} {total_time_str:>10}  {total_size_mb:>8.1f} MB")
        lines.append(f"{'─'*60}")

    token_text = _token_tracker.format_summary()
    if token_text:
        lines.append(token_text)

    summary = "\n".join(lines)
    print(summary)

    # ── txt 파일로 저장 ───────────────────────────────────────────────────────
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_folder) / f"travel_report_{run_time}.txt"
    try:
        report_path.write_text(summary, encoding="utf-8")
        print(f"  리포트 저장: {report_path}")
    except OSError as e:
        print(f"  [경고] 리포트 저장 실패: {e}")

    # ── 클립별 요약 리포트 (clips_summary.json/csv, selected_clips.json) ──────────
    if getattr(_config, "CLIP_SUMMARY", True):
        try:
            from core.summary import build_clip_rows, write_clip_summary
            rows = build_clip_rows(segments, evaluations, transcripts, selection_info)
            info = write_clip_summary(output_folder, rows)
            if info:
                print(f"  클립 요약 저장: {info['csv']}")
                print(f"                 (전체 {info['total']}개 · 선택 {info['selected']}개"
                      f" · selected_clips.json/clips_summary.json 동시 생성)")
        except Exception as e:
            print(f"  [경고] 클립 요약 저장 실패: {e}")
