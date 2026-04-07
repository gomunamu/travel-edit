"""전체 편집 파이프라인 조율"""
import hashlib
import json
import os
import shutil
import time
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
from core.cache import Cache, make_clip_hash
from core.metadata import get_video_info, is_video
from core.segmenter import plan_segments
from core.transcriber import transcribe, init_model_pool, get_pool_size, release_model_pool
from core.refiner import refine_transcript
from core.evaluator import evaluate_clip
from core.geocoder import coords_to_str
from core.subtitle import make_subtitle_ass, make_subtitle_srt, make_location_ass
from core.evaluator import _adaptive as _eval_adaptive
from core.renderer import get_day_resolution, render_day_onepass, is_valid_video

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

    # 촬영 시간 순 정렬
    def sort_key(c):
        ct = c.get("creation_time")
        if ct:
            try:
                return datetime.fromisoformat(ct)
            except Exception:
                pass
        return datetime.min.replace(tzinfo=timezone.utc)

    results.sort(key=sort_key)
    return results


# ─── 3. 세그먼트 계획 (파일 추출 없음, JSON 캐시) ────────────────────────────
def segment_all(clips: List[dict], cache: Cache) -> List[dict]:
    """
    클립을 논리 세그먼트로 분할한다.
    계획 결과는 {cache}/.segment_plan.json 에 저장되어
    재시작 시 바로 불러온다. 입력 클립이 바뀌면 자동으로 재계획.
    """
    plan_path = cache.root / ".segment_plan.json"

    # 클립 목록 지문 — clip_hash 는 파일 경로+mtime+size 기반이므로
    # 파일이 추가·삭제·변경되면 지문이 달라진다
    fingerprint = hashlib.sha256(
        "|".join(c["clip_hash"] for c in clips).encode()
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
    transcripts = {}
    need_transcribe = []

    for seg in segments:
        h = seg["clip_hash"]
        cached = cache.load(h, "transcript")
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
        cache.save(h, "transcript", result)

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
    Whisper 결과를 Claude로 한 번 더 정제한다.
    캐시 키: "transcript_refined" — 재실행 시 재사용.
    """
    from config import ANTHROPIC_API_KEY, STT_REFINE_MODEL, CLAUDE_MAX_CONCURRENT

    refined_map: Dict[str, dict] = {}
    need_refine = []

    for seg in segments:
        h = seg["clip_hash"]
        if h not in transcripts:
            continue
        cached = cache.load(h, "transcript_refined")
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
        try:
            with semaphore:
                result = refine_transcript(original, ANTHROPIC_API_KEY, STT_REFINE_MODEL)
        except Exception as e:
            print(f"\n  [경고] STT 정제 실패 ({Path(seg['filepath']).name}): {e}")
            result = original  # 실패해도 원본으로 캐시 저장 → 재실행 시 재시도 안 함
        cache.save(h, "transcript_refined", result)
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


# ─── 5. AI 평가 (병렬 - Claude API) ───────────────────────────────────────
def evaluate_all(
    segments: List[dict],
    transcripts: Dict[str, dict],
    cache: Cache,
) -> Dict[str, dict]:
    evaluations = {}
    need_eval = []

    for seg in segments:
        h = seg["clip_hash"]
        cached = cache.load(h, "eval")
        if cached:
            evaluations[h] = cached
        else:
            need_eval.append(seg)

    if need_eval:
        print(f"  AI 평가 필요: {len(need_eval)}개 (캐시 {len(evaluations)}개 재사용)")

    lock = threading.Lock()

    def _eval_one(seg):
        h = seg["clip_hash"]
        transcript = transcripts.get(h, {"segments": [], "has_speech": False, "total_speech_sec": 0})
        result = evaluate_clip(seg, transcript)
        cache.save(h, "eval", result)
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
                future.result()
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
) -> bool:
    """
    하루치 세그먼트를 평가 결과에 따라 선별하고,
    filter_complex 로 한 번에 트림·스케일·자막·병합 (중간 파일 없음).
    """
    out_res = get_day_resolution(day_segments)
    selected = []
    prev_location = None

    for seg in day_segments:
        h = seg["clip_hash"]
        ev = evaluations.get(h, {"decision": "keep", "keep_start": 0, "keep_end": seg["duration"]})
        decision = ev.get("decision", "keep")

        if decision == "discard":
            sc = ev.get("score", {})
            total = sc.get("total", "?")
            print(f"    [버림 {total:>3}점] {seg['filename']} - {ev.get('reason','')}")
            continue

        clip = dict(seg)
        if decision == "trim":
            clip["trim_start"] = float(ev.get("keep_start", 0))
            clip["trim_end"]   = float(ev.get("keep_end", seg["duration"]))
        else:
            clip["trim_start"] = 0.0
            clip["trim_end"]   = seg["duration"]

        # 장소 결정
        location_name = None
        if seg.get("gps"):
            location_name = coords_to_str(seg["gps"])
        if location_name and location_name != prev_location:
            clip["show_location"] = location_name
            prev_location = location_name
        else:
            clip["show_location"] = None

        clip["eval"] = ev
        selected.append(clip)
        sc = ev.get("score", {})
        total  = sc.get("total",  "?")
        visual = sc.get("visual", "-")
        speech = sc.get("speech", "-")
        scene  = sc.get("scene",  "-")
        flow   = sc.get("flow",   "-")
        tag = "트림" if decision == "trim" else "살림"
        print(
            f"    [{tag} {total:>3}점"
            f"  시각{visual}/음성{speech}/장면{scene}/흐름{flow}] "
            f"{seg['filename']} {clip['trim_start']:.1f}~{clip['trim_end']:.1f}s"
            f"  {ev.get('reason','')}"
        )

    if not selected:
        print(f"  → {day_key}: 선택된 클립 없음, 건너뜀")
        return False

    print(f"  출력 해상도: {out_res[0]}x{out_res[1]}, {len(selected)}개 클립")

    # ASS 파일 생성 (자막·장소) — 캐시 활용
    clips_info = []
    for clip in selected:
        h = clip["clip_hash"]
        trim_start = clip["trim_start"]
        trim_end   = clip["trim_end"]
        clip_dur   = trim_end - trim_start

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

        # 장소 ASS
        loc_path = None
        loc_name = clip.get("show_location")
        if loc_name:
            loc_path = cache.ass_path(h, "location_r")
            if not cache.ass_exists(h, "location_r"):
                make_location_ass(
                    loc_name, clip_dur, loc_path, out_res,
                    display_duration=LOCATION_DISPLAY_DURATION,
                    fade_duration=LOCATION_FADE_DURATION,
                    font=SUBTITLE_FONT, font_size=LOCATION_FONT_SIZE,
                    margin=LOCATION_MARGIN,
                )

        clips_info.append({**clip, "sub_path": sub_path, "loc_path": loc_path})

    print(f"  렌더링+병합 → {Path(output_path).name}")
    ok = render_day_onepass(clips_info, output_path, out_res)

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
    print(f"\n{'='*60}")
    print(f"  여행 영상 자동 편집기")
    print(f"  입력: {input_folder}")
    print(f"  출력: {output_folder}")
    print(f"  해상도: {res_str}")
    print(f"{'='*60}\n")

    run_start = time.time()
    file_report: List[dict] = []   # {"name", "elapsed", "size_mb"}

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
        if _config.STT_REFINE and _config.ANTHROPIC_API_KEY:
            print(f"\n[4-b] STT 정제 (LLM: {_config.STT_REFINE_MODEL})...")
            transcripts = refine_all(segments, transcripts, cache)
            _token_tracker.print_current("4-b STT 정제 후")
        elif _config.STT_REFINE and not _config.ANTHROPIC_API_KEY:
            print("\n[4-b] STT 정제 건너뜀 (ANTHROPIC_API_KEY 없음)")

        # 4-c. 자막 / 위치 미리보기
        print("\n[4-c] 자막 미리보기 (최종):")
        print_clip_preview(segments, transcripts)

    # 5. AI 평가
    print(f"\n[5/6] AI 클립 평가...")
    evaluations = evaluate_all(segments, transcripts, cache)
    _token_tracker.print_current("5단계 AI 평가 후")

    # 6. 일자별 렌더링
    print("\n[6/6] 일자별 편집 및 렌더링...")

    # 날짜별 그룹화
    day_groups = defaultdict(list)
    for seg in segments:
        day_groups[seg.get("day_key", "unknown")].append(seg)

    for day_key in sorted(day_groups.keys()):
        day_segs = day_groups[day_key]
        print(f"\n  ── {day_key} ({len(day_segs)}개 클립) ──")

        output_path = str(out_dir / f"travel_{day_key}.mp4")

        if Path(output_path).exists():
            size_mb = Path(output_path).stat().st_size / 1_048_576
            print(f"  → 이미 존재: {output_path} (건너뜀, {size_mb:.1f} MB)")
            file_report.append({"name": Path(output_path).name, "path": output_path, "elapsed": 0.0, "size_mb": size_mb, "skipped": True})
            continue

        day_start = time.time()
        ok = render_day(
            day_key, day_segs, transcripts, evaluations,
            cache, output_path
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
