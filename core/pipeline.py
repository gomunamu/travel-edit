"""전체 편집 파이프라인 조율"""
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from typing import List, Dict, Optional
import threading

import config as _config
from config import (
    METADATA_WORKERS, SEGMENT_WORKERS, TRANSCRIBE_WORKERS, RENDER_WORKERS,
    CLAUDE_MAX_CONCURRENT, LOCATION_DISPLAY_DURATION, LOCATION_FADE_DURATION,
    SUBTITLE_FONT, SUBTITLE_FONT_SIZE, SUBTITLE_MARGIN_V,
    LOCATION_FONT_SIZE, LOCATION_MARGIN,
)
from core.cache import Cache, make_clip_hash
from core.metadata import get_video_info, is_video
from core.segmenter import split_clip
from core.transcriber import transcribe, init_model_pool
from core.evaluator import evaluate_clip
from core.geocoder import coords_to_str
from core.subtitle import make_subtitle_ass, make_subtitle_srt, make_location_ass
from core.renderer import get_day_resolution, render_clip, concat_day, is_valid_video

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
    videos = []
    for p in sorted(Path(input_folder).rglob("*")):
        if p.is_file() and is_video(str(p)):
            videos.append(str(p))
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


# ─── 3. 세그먼트 분할 (클립 간 병렬, 클립 내부도 병렬) ──────────────────────
def segment_all(clips: List[dict], segments_dir: str, cache: Cache) -> List[dict]:
    # 워커 수 결정:
    #   - 클립 간(outer): 각 클립이 독립적이므로 파일 수만큼 병렬
    #   - 클립 내(inner): 한 클립의 세그먼트들, 같은 소스 파일 읽기
    # outer * inner 가 총 ffmpeg 프로세스 수 → 디스크 I/O 포화 방지
    cpu = os.cpu_count() or 4
    outer = SEGMENT_WORKERS or max(2, cpu // 2)
    inner = max(2, cpu // outer)          # 전체 프로세스 수 ≒ cpu 수 유지

    needs_split = [c for c in clips if c["duration"] > 0]

    results: List[tuple] = []  # (original_index, segs)

    def _split_one(idx_clip):
        idx, clip = idx_clip
        segs = split_clip(clip, segments_dir, inner_workers=inner)
        for seg in segs:
            if seg["clip_hash"] != clip["clip_hash"]:
                if not cache.load(seg["clip_hash"], "meta"):
                    cache.save(seg["clip_hash"], "meta", seg)
        return idx, segs

    with ThreadPoolExecutor(max_workers=outer) as ex:
        futures = {ex.submit(_split_one, (i, c)): i for i, c in enumerate(needs_split)}
        for future in _tqdm(as_completed(futures), total=len(futures), desc="  클립 분할"):
            try:
                idx, segs = future.result()
                results.append((idx, segs))
            except Exception as e:
                orig_idx = futures[future]
                clip_name = needs_split[orig_idx].get("filename", "?")
                print(f"\n  [경고] 클립 분할 실패, 건너뜀: {clip_name}\n    {e}")

    # 원본 순서(촬영 시간 순) 복원
    results.sort(key=lambda x: x[0])
    all_segs = [seg for _, segs in results for seg in segs]
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

    # 모델 풀 초기화 (TRANSCRIBE_WORKERS 수만큼 인스턴스 로드)
    init_model_pool(TRANSCRIBE_WORKERS)

    lock = threading.Lock()

    _lang_map = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}
    force_lang = _lang_map.get(_config.SUBTITLE_LANG)  # None = auto

    def _transcribe_one(seg):
        h = seg["clip_hash"]
        result = transcribe(seg["filepath"], force_lang=force_lang)
        cache.save(h, "transcript", result)
        with lock:
            transcripts[h] = result

    with ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS) as ex:
        futures = [ex.submit(_transcribe_one, seg) for seg in need_transcribe]
        for future in _tqdm(as_completed(futures), total=len(futures), desc="  음성 인식"):
            future.result()

    return transcripts


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
    semaphore = threading.Semaphore(CLAUDE_MAX_CONCURRENT)

    def _eval_one(seg):
        h = seg["clip_hash"]
        transcript = transcripts.get(h, {"segments": [], "has_speech": False, "total_speech_sec": 0})
        with semaphore:
            result = evaluate_clip(seg, transcript)
        cache.save(h, "eval", result)
        with lock:
            evaluations[h] = result
        return h, result

    with ThreadPoolExecutor(max_workers=CLAUDE_MAX_CONCURRENT) as ex:
        futures = [ex.submit(_eval_one, seg) for seg in need_eval]
        for future in _tqdm(as_completed(futures), total=len(futures), desc="  AI 평가"):
            future.result()

    return evaluations


# ─── 6. 일자별 렌더링 ─────────────────────────────────────────────────────
def render_day(
    day_key: str,
    day_segments: List[dict],
    transcripts: Dict[str, dict],
    evaluations: Dict[str, dict],
    cache: Cache,
    rendered_dir: Path,
    output_path: str,
) -> bool:
    # 선택된 클립 필터링
    selected = []
    prev_location = None

    for seg in day_segments:
        h = seg["clip_hash"]
        ev = evaluations.get(h, {"decision": "keep", "keep_start": 0, "keep_end": seg["duration"]})
        decision = ev.get("decision", "keep")

        if decision == "discard":
            print(f"    [버림 {ev.get('interest_score','?')}점] {seg['filename']} - {ev.get('reason','')}")
            continue

        clip = dict(seg)
        if decision == "trim":
            clip["trim_start"] = float(ev.get("keep_start", 0))
            clip["trim_end"] = float(ev.get("keep_end", seg["duration"]))
        else:
            clip["trim_start"] = 0.0
            clip["trim_end"] = seg["duration"]

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
        print(
            f"    [{'트림' if decision=='trim' else '살림'} {ev.get('interest_score','?')}점] "
            f"{seg['filename']} {clip['trim_start']:.1f}~{clip['trim_end']:.1f}s"
            f"  {ev.get('reason','')}"
        )

    if not selected:
        print(f"  → {day_key}: 선택된 클립 없음, 건너뜀")
        return False

    out_res = get_day_resolution(selected)
    print(f"  출력 해상도: {out_res[0]}x{out_res[1]}, {len(selected)}개 클립")

    # 병렬 렌더링
    render_args = []
    for idx, clip in enumerate(selected):
        h = clip["clip_hash"]
        out_clip = rendered_dir / f"clip_{idx:04d}_{h}.mp4"

        # 자막 ASS 생성 (overlay 모드에서만 번인)
        sub_path = None
        transcript = transcripts.get(h, {})
        segs = transcript.get("segments", [])
        if segs and transcript.get("has_speech") and _config.SUBTITLE_MODE == "overlay":
            sub_path = cache.ass_path(h, "subtitle")
            if not cache.ass_exists(h, "subtitle"):
                trim_offset = clip["trim_start"]
                adjusted = [
                    dict(s, start=s["start"] - trim_offset, end=s["end"] - trim_offset)
                    for s in segs
                    if s["end"] > trim_offset
                ]
                make_subtitle_ass(
                    adjusted, sub_path, out_res,
                    font=SUBTITLE_FONT, font_size=SUBTITLE_FONT_SIZE,
                    margin_v=SUBTITLE_MARGIN_V
                )

        # 장소 ASS 생성
        loc_path = None
        loc_name = clip.get("show_location")
        if loc_name:
            loc_path = cache.ass_path(h, "location")
            if not cache.ass_exists(h, "location"):
                clip_dur = clip["trim_end"] - clip["trim_start"]
                make_location_ass(
                    loc_name, clip_dur, loc_path, out_res,
                    display_duration=LOCATION_DISPLAY_DURATION,
                    fade_duration=LOCATION_FADE_DURATION,
                    font=SUBTITLE_FONT, font_size=LOCATION_FONT_SIZE,
                    margin=LOCATION_MARGIN,
                )

        render_args.append((clip, str(out_clip), out_res, sub_path, loc_path))

    # 이미 렌더링된 클립 건너뜀 (corrupt 파일은 재렌더링)
    def _needs_render(path: str, loc_path: str = None) -> bool:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 10_000:
            return True
        if not is_valid_video(path):
            p.unlink(missing_ok=True)
            return True
        # 위치 ASS가 렌더링 클립보다 새로우면 위치 오버레이 반영 위해 재렌더
        if loc_path:
            lp = Path(loc_path)
            if lp.exists() and lp.stat().st_mtime > p.stat().st_mtime:
                return True
        return False

    to_render = [(args, i) for i, args in enumerate(render_args)
                 if _needs_render(args[1], args[4])]

    failed_indices = set()

    if to_render:
        n_workers = RENDER_WORKERS or max(1, (os.cpu_count() or 2) // 2)
        print(f"  렌더링: {len(to_render)}개 (워커 {n_workers}개)")

        lock2 = threading.Lock()

        def _render(item):
            args, idx = item
            clip, out_clip, out_res_, sub_path_, loc_path_ = args
            ok = render_clip(clip, out_clip, out_res_, sub_path_, loc_path_)
            if not ok:
                with lock2:
                    failed_indices.add(idx)
            return ok

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(_tqdm(ex.map(_render, to_render), total=len(to_render), desc="  렌더링"))

        if failed_indices:
            print(f"  [경고] {len(failed_indices)}개 클립 렌더링 실패, 병합에서 제외")

    # 최종 병합 (존재하고 유효한 클립만 포함, 실패/손상 클립 제외)
    included = [
        args for i, args in enumerate(render_args)
        if i not in failed_indices
        and Path(args[1]).exists() and Path(args[1]).stat().st_size >= 10_000
        and is_valid_video(args[1])
    ]
    clip_paths = [args[1] for args in included]
    if not clip_paths:
        print(f"  → {day_key}: 렌더링된 클립 없음")
        return False

    print(f"  병합 중: {len(clip_paths)}개 → {Path(output_path).name}")
    ok = concat_day(clip_paths, output_path)

    # SRT 모드: 병합 성공 후 타임라인 맞춰 단일 SRT 파일 생성
    if ok and _config.SUBTITLE_MODE == "srt":
        cumulative = 0.0
        srt_segs = []
        for args in included:
            clip, _, _, _, _ = args
            h = clip["clip_hash"]
            transcript = transcripts.get(h, {})
            segs = transcript.get("segments", [])
            trim_offset = clip["trim_start"]
            clip_dur = clip["trim_end"] - clip["trim_start"]
            if segs and transcript.get("has_speech"):
                for seg in segs:
                    if seg.get("no_speech_prob", 1.0) < 0.5 and seg.get("text", "").strip():
                        start = max(0.0, seg["start"] - trim_offset) + cumulative
                        end   = min(max(start + 0.5, seg["end"] - trim_offset) + cumulative,
                                    cumulative + clip_dur)
                        srt_segs.append({"start": start, "end": end,
                                         "text": seg["text"].strip()})
            cumulative += clip_dur
        if srt_segs:
            srt_path = output_path.replace(".mp4", ".srt")
            make_subtitle_srt(srt_segs, srt_path)
            print(f"  ✓ SRT 자막: {Path(srt_path).name} ({len(srt_segs)}개)")

    # 병합 완료 후 개별 렌더링 클립 삭제
    if ok:
        for p in clip_paths:
            Path(p).unlink(missing_ok=True)

    return ok


# ─── 전체 파이프라인 ──────────────────────────────────────────────────────
def run(input_folder: str, output_folder: str):
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache = Cache(str(cache_dir))
    segments_dir = cache_dir / "segments"

    print(f"\n{'='*60}")
    print(f"  여행 영상 자동 편집기")
    print(f"  입력: {input_folder}")
    print(f"  출력: {output_folder}")
    print(f"{'='*60}\n")

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

    # 3. 세그먼트 분할
    print("\n[3/6] 긴 클립 분할...")
    segments = segment_all(clips, str(segments_dir), cache)
    print(f"  → {len(segments)}개 세그먼트 (원본 {len(clips)}개)")

    # 4. 음성 인식
    print("\n[4/6] 음성 인식 (Whisper)...")
    if _config.SUBTITLE_LANG == "off":
        print("  → 자막 비활성화, 건너뜀")
        transcripts = {}
    else:
        transcripts = transcribe_all(segments, cache)

    # 5. AI 평가
    print("\n[5/6] AI 클립 평가...")
    evaluations = evaluate_all(segments, transcripts, cache)

    # 6. 일자별 렌더링
    print("\n[6/6] 일자별 편집 및 렌더링...")

    # 날짜별 그룹화
    day_groups = defaultdict(list)
    for seg in segments:
        day_groups[seg.get("day_key", "unknown")].append(seg)

    for day_key in sorted(day_groups.keys()):
        day_segs = day_groups[day_key]
        print(f"\n  ── {day_key} ({len(day_segs)}개 클립) ──")

        rendered_dir = out_dir / "rendered" / day_key
        rendered_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"travel_{day_key}.mp4")

        if Path(output_path).exists():
            print(f"  → 이미 존재: {output_path} (건너뜀)")
            continue

        ok = render_day(
            day_key, day_segs, transcripts, evaluations,
            cache, rendered_dir, output_path
        )
        if ok:
            size_mb = Path(output_path).stat().st_size / 1_048_576
            print(f"  ✓ 완료: {output_path} ({size_mb:.1f} MB)")

            # 렌더링 완료 후 해당 날짜 세그먼트 삭제 (원본 클립은 유지)
            freed = 0
            for seg in day_segs:
                seg_path = Path(seg["filepath"])
                if seg_path.exists() and seg.get("parent_hash"):  # 분할된 세그먼트만
                    freed += seg_path.stat().st_size
                    seg_path.unlink()
            if freed:
                print(f"  → 세그먼트 정리: {freed / 1_048_576:.1f} MB 해제")

    print(f"\n{'='*60}")
    print(f"  편집 완료! 출력 폴더: {output_folder}")
    print(f"{'='*60}\n")
