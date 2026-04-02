"""
속도 개선 벤치마크 테스트
음성 있는 원본클립 4개 → 세그먼트 분리 → STT → 자막 생성 → 렌더링 → 병합
각 단계별 시간 측정 및 요약 출력
"""
import os
import sys
import time
import tempfile
import shutil
import argparse
from pathlib import Path
from contextlib import contextmanager

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

import config as _config
from core.metadata import get_video_info, is_video
from core.segmenter import split_clip
from core.transcriber import transcribe, init_model_pool
from core.subtitle import make_subtitle_ass
from core.renderer import render_clip, concat_day, get_day_resolution, is_valid_video
from core.cache import make_clip_hash, make_segment_hash

# ─── 타이머 유틸 ─────────────────────────────────────────────────────────────

_timings: list[dict] = []


@contextmanager
def timer(label: str):
    print(f"\n[타이머] {label} 시작...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _timings.append({"label": label, "sec": elapsed})
        print(f"[타이머] {label} 완료: {elapsed:.2f}s")


def print_summary():
    total = sum(t["sec"] for t in _timings)
    print("\n" + "=" * 60)
    print("  벤치마크 결과 요약")
    print("=" * 60)
    bar_max = max(t["sec"] for t in _timings) if _timings else 1
    for t in _timings:
        pct = t["sec"] / total * 100 if total > 0 else 0
        bar_len = int(t["sec"] / bar_max * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {t['label']:<22} {t['sec']:6.2f}s  {pct:5.1f}%  {bar}")
    print("-" * 60)
    print(f"  {'합계':<22} {total:6.2f}s  100.0%")
    print("=" * 60)


# ─── 음성 있는 클립 선택 ──────────────────────────────────────────────────────

def find_clips_with_audio(input_folder: str, n: int = 4) -> list[dict]:
    """입력 폴더에서 음성 스트림이 있는 클립 n개 선택"""
    videos = sorted(Path(input_folder).rglob("*"))
    videos = [str(p) for p in videos if p.is_file() and is_video(str(p))]

    if not videos:
        raise FileNotFoundError(f"비디오 파일 없음: {input_folder}")

    print(f"  전체 비디오: {len(videos)}개 스캔 중...")
    found = []
    for fp in videos:
        info = get_video_info(fp)
        if info is None:
            continue
        if info.get("has_audio"):
            info["clip_hash"] = make_clip_hash(fp)
            info["trim_start"] = 0.0
            info["trim_end"] = info["duration"]
            info["segment_index"] = 0
            info["parent_hash"] = None
            found.append(info)
            print(f"  ✓ {info['filename']}  {info['duration']:.1f}s  {info['display_width']}x{info['display_height']}")
        if len(found) >= n:
            break

    if len(found) < n:
        print(f"  [경고] 음성 있는 클립 {len(found)}개만 발견 (요청: {n}개)")

    return found


# ─── 단계별 파이프라인 ────────────────────────────────────────────────────────

def run_benchmark(input_folder: str, output_dir: str, n_clips: int = 4):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    segments_dir = out_path / "segments"
    segments_dir.mkdir(exist_ok=True)
    rendered_dir = out_path / "rendered"
    rendered_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  속도 벤치마크 테스트")
    print(f"  Whisper 모델: {_config.WHISPER_MODEL}  장치: {_config.WHISPER_DEVICE}")
    print(f"  입력: {input_folder}")
    print(f"  출력: {output_dir}")
    print(f"{'='*60}")

    # ── STEP 0: 클립 선택 ────────────────────────────────────────────────────
    print("\n[STEP 0] 음성 있는 클립 선택...")
    with timer("0. 클립 선택 (메타데이터)"):
        clips = find_clips_with_audio(input_folder, n=n_clips)

    if not clips:
        print("[오류] 음성 있는 클립을 찾을 수 없습니다.")
        return

    print(f"\n  → 선택된 {len(clips)}개 클립:")
    for c in clips:
        print(f"     - {c['filename']}  {c['duration']:.1f}s  오디오: {c['audio_codec']}")

    # ── STEP 1: 세그먼트 분리 ─────────────────────────────────────────────────
    print("\n[STEP 1] 세그먼트 분리...")
    all_segments = []
    with timer("1. 세그먼트 분리"):
        import os
        cpu = os.cpu_count() or 4
        inner = max(2, cpu // 2)
        for clip in clips:
            segs = split_clip(clip, str(segments_dir), inner_workers=inner)
            all_segments.extend(segs)
            print(f"  {clip['filename']}: {len(segs)}개 세그먼트")

    print(f"  → 총 {len(all_segments)}개 세그먼트 (원본 {len(clips)}개 클립)")

    # ── STEP 2: STT (Whisper) ─────────────────────────────────────────────────
    print("\n[STEP 2] 음성 인식 (Whisper STT)...")
    transcripts: dict[str, dict] = {}

    with timer("2a. Whisper 모델 로드"):
        init_model_pool(1)  # 모델 1개로 직렬 테스트 (정확한 STT 시간 측정)

    stt_start = time.perf_counter()
    with timer("2b. STT 전체"):
        for i, seg in enumerate(all_segments):
            t0 = time.perf_counter()
            result = transcribe(seg["filepath"])
            elapsed = time.perf_counter() - t0
            transcripts[seg["clip_hash"]] = result
            speech_info = f"음성 {result['total_speech_sec']:.1f}s" if result["has_speech"] else "음성 없음"
            lang = result.get("language", "?")
            print(f"  [{i+1}/{len(all_segments)}] {Path(seg['filepath']).name}  "
                  f"{seg['duration']:.1f}s → {elapsed:.2f}s  ({speech_info}, lang={lang})")

    # RTF 계산 (Real-Time Factor)
    total_audio_dur = sum(s["duration"] for s in all_segments)
    stt_elapsed = sum(t["sec"] for t in _timings if "STT" in t["label"])
    stt_wall = next((t["sec"] for t in _timings if "2b" in t["label"]), 1)
    rtf = stt_wall / total_audio_dur if total_audio_dur > 0 else 0
    print(f"  → RTF (실시간 배율): {rtf:.3f}x  (1.0 미만 = 실시간보다 빠름)")

    # ── STEP 3: 자막 ASS 생성 ────────────────────────────────────────────────
    print("\n[STEP 3] 자막 ASS 파일 생성...")
    subtitle_paths: dict[str, str] = {}
    ass_dir = out_path / "ass"
    ass_dir.mkdir(exist_ok=True)

    with timer("3. 자막 ASS 생성"):
        out_res = get_day_resolution(all_segments) if all_segments else (1920, 1080)
        for seg in all_segments:
            h = seg["clip_hash"]
            transcript = transcripts.get(h, {})
            segs_data = transcript.get("segments", [])
            if segs_data and transcript.get("has_speech"):
                ass_path = str(ass_dir / f"{h}_subtitle.ass")
                adjusted = [
                    dict(s, start=s["start"], end=s["end"])
                    for s in segs_data
                    if s.get("no_speech_prob", 1.0) < 0.5 and s.get("text", "").strip()
                ]
                if adjusted:
                    make_subtitle_ass(
                        adjusted, ass_path, out_res,
                        font=_config.SUBTITLE_FONT,
                        font_size=_config.SUBTITLE_FONT_SIZE,
                        margin_v=_config.SUBTITLE_MARGIN_V,
                    )
                    subtitle_paths[h] = ass_path
                    print(f"  ✓ {Path(seg['filepath']).name}: {len(adjusted)}개 자막 세그먼트")
            else:
                print(f"  - {Path(seg['filepath']).name}: 음성 없음, 자막 건너뜀")

    # ── STEP 4: 클립 렌더링 ──────────────────────────────────────────────────
    print("\n[STEP 4] 클립 렌더링...")
    rendered_paths: list[str] = []

    with timer("4. 클립 렌더링"):
        for i, seg in enumerate(all_segments):
            h = seg["clip_hash"]
            out_clip = str(rendered_dir / f"clip_{i:04d}_{h}.mp4")
            sub_path = subtitle_paths.get(h)

            t0 = time.perf_counter()
            ok = render_clip(seg, out_clip, out_res, subtitle_path=sub_path)
            elapsed = time.perf_counter() - t0

            if ok and Path(out_clip).exists():
                size_mb = Path(out_clip).stat().st_size / 1_048_576
                rendered_paths.append(out_clip)
                print(f"  [{i+1}/{len(all_segments)}] {Path(seg['filepath']).name} → "
                      f"{elapsed:.2f}s  {size_mb:.1f}MB  자막={'O' if sub_path else 'X'}")
            else:
                print(f"  [{i+1}/{len(all_segments)}] {Path(seg['filepath']).name} → 실패")

    # ── STEP 5: 최종 병합 ────────────────────────────────────────────────────
    print("\n[STEP 5] 최종 병합...")
    final_output = str(out_path / "benchmark_output.mp4")

    # 유효한 클립만 포함
    valid_paths = [
        p for p in rendered_paths
        if Path(p).exists()
        and Path(p).stat().st_size >= 10_000
        and is_valid_video(p)
    ]
    print(f"  유효한 클립: {len(valid_paths)}/{len(rendered_paths)}개")

    with timer("5. 최종 병합"):
        ok = concat_day(valid_paths, final_output)

    if ok and Path(final_output).exists():
        size_mb = Path(final_output).stat().st_size / 1_048_576
        print(f"  ✓ 병합 완료: {final_output}  ({size_mb:.1f} MB)")
    else:
        print("  [오류] 병합 실패")

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    print_summary()

    print(f"\n  총 오디오 길이: {total_audio_dur:.1f}s")
    total_wall = sum(t["sec"] for t in _timings)
    speedup = total_audio_dur / total_wall if total_wall > 0 else 0
    print(f"  전체 처리 시간: {total_wall:.2f}s")
    print(f"  오디오 대비 속도: {speedup:.2f}x 실시간\n")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel-Edit 속도 벤치마크")
    parser.add_argument(
        "input_folder",
        nargs="?",
        default="./input",
        help="원본 영상 폴더 (기본값: ./input)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./benchmark_out",
        help="벤치마크 출력 폴더 (기본값: ./benchmark_out)",
    )
    parser.add_argument(
        "--clips", "-n",
        type=int,
        default=4,
        help="테스트할 클립 수 (기본값: 4)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="실행 전 출력 폴더 초기화",
    )
    args = parser.parse_args()

    if args.clean and Path(args.output).exists():
        print(f"[초기화] {args.output} 삭제 중...")
        shutil.rmtree(args.output)

    run_benchmark(args.input_folder, args.output, n_clips=args.clips)
