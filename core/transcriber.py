"""faster-whisper 기반 음성 인식 (한국어/영어 자동 감지)"""
import os
import platform
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

_IS_MACOS = platform.system() == "Darwin"

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_pool: Optional[queue.Queue] = None
_pool_lock = threading.Lock()
_pool_total = 0          # 체크아웃 중 포함 전체 살아있는 인스턴스 수
_pool_total_lock = threading.Lock()

# distil-whisper 단축명 → HuggingFace 모델 ID 매핑
_DISTIL_ALIASES = {
    "distil-large-v3":  "Systran/faster-distil-whisper-large-v3",
    "distil-large-v2":  "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en":  "Systran/faster-distil-whisper-small.en",
}

def _resolve_model_name(name: str) -> str:
    return _DISTIL_ALIASES.get(name.lower(), name)


def _load_one_model():
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    import ctranslate2

    device = WHISPER_DEVICE
    compute = WHISPER_COMPUTE_TYPE
    model_name = _resolve_model_name(WHISPER_MODEL)

    # MPS: ctranslate2가 Apple MPS를 미지원 → CPU로 전환
    if device == "mps":
        print("  [정보] ctranslate2는 MPS 미지원 → CPU로 전환")
        device = "cpu"
        compute = "int8"
    elif device == "cuda":
        if _IS_MACOS:
            # macOS에서 cuda 강제 지정 시에도 CPU로 전환
            print("  [정보] macOS에서 CUDA 미지원 → CPU로 전환")
            device = "cpu"
            compute = "int8"
        elif ctranslate2.get_cuda_device_count() == 0:
            print("  [경고] CUDA 장치 없음, CPU로 전환")
            device = "cpu"
            compute = "int8"
        elif compute not in ctranslate2.get_supported_compute_types("cuda"):
            compute = "float16"

    base = WhisperModel(
        model_name,
        device=device,
        compute_type=compute,
        num_workers=1,
        cpu_threads=4,
    )
    # BatchedInferencePipeline: 오디오 청크를 GPU에서 병렬 처리 → 특히 긴 파일에서 빠름
    return BatchedInferencePipeline(model=base), device


_MAX_AUTO_WORKERS = 8  # auto 모드(n=0)일 때 시도할 최대 인스턴스 수
# 추론 시 idle VRAM 대비 추가 사용량 비율 (KV cache, activation 등)
# large-v3 batch_size=4 기준 약 30~70% 추가 사용 (GPU/워크로드마다 다름)
# .env: WHISPER_VRAM_OVERHEAD=0.40  (낮출수록 더 많은 인스턴스 로드, OOM 위험 증가)
_INFERENCE_OVERHEAD = float(os.environ.get("WHISPER_VRAM_OVERHEAD", "0.40"))
_VRAM_FLOOR_MB = 1500        # 로드 후 최소 유지해야 할 절대 여유 VRAM (MB)


def _get_vram_free_mb() -> Optional[int]:
    """현재 GPU 여유 VRAM (MB). 측정 불가 시 None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0].strip())
    except Exception:
        pass
    return None


def init_model_pool(n: int = 0):
    """
    Whisper 모델 인스턴스를 풀에 적재.
    n=0 (auto): VRAM 여유를 추론 헤드룸까지 고려해 인스턴스 수 결정.
    n>0: 지정한 수만큼 시도, VRAM 부족 시 자동 축소.

    idle 상태 VRAM만 측정하면 실제 추론 시 OOM이 발생한다.
    각 인스턴스 로드 후 소비한 VRAM의 (1 + _INFERENCE_OVERHEAD)배를
    헤드룸으로 예약해 동시 추론 시 OOM을 방지한다.
    """
    global _pool, _pool_total
    with _pool_lock:
        if _pool is not None:
            return
        auto = (n == 0)
        target = _MAX_AUTO_WORKERS if auto else n
        label = f"최대 {target}개 (VRAM 자동)" if auto else f"최대 {target}개"
        print(f"  Whisper 모델 로드 중: {WHISPER_MODEL} × {label}")
        _pool = queue.Queue()
        loaded = 0
        vram_before = _get_vram_free_mb()  # 최초 여유 VRAM

        for i in range(target):
            # 로드 전 여유 VRAM 측정
            free_before = _get_vram_free_mb()
            try:
                model, device = _load_one_model()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if loaded == 0:
                        raise RuntimeError(
                            "Whisper 모델을 하나도 로드할 수 없습니다. VRAM을 확인하세요."
                        )
                    print(f"  VRAM 한계 도달 (로드 실패) → {loaded}개 인스턴스로 확정")
                    break
                raise

            _pool.put(model)
            loaded += 1
            free_after = _get_vram_free_mb()
            print(f"  ✓ 인스턴스 {loaded} 로드 완료 ({device.upper()})", end="")

            # VRAM 여유 체크 (auto 모드에서만)
            if auto and free_after is not None and free_before is not None:
                model_vram = max(0, free_before - free_after)  # 이번 인스턴스가 사용한 VRAM
                # 추론 헤드룸: 이번 인스턴스 VRAM × overhead 비율
                inference_headroom = int(model_vram * _INFERENCE_OVERHEAD)
                # 다음 인스턴스를 로드해도 헤드룸이 확보되는지 확인
                needed = model_vram + inference_headroom  # 다음 로드 + 현재 인스턴스 헤드룸
                floor = max(_VRAM_FLOOR_MB, inference_headroom)
                print(f"  (여유 {free_after}MB, 모델 {model_vram}MB, 헤드룸 {inference_headroom}MB)")
                if free_after < needed + floor:
                    print(f"  VRAM 헤드룸 부족 → {loaded}개 인스턴스로 확정 (추론 중 OOM 방지)")
                    break
            else:
                print()

        if loaded == 0:
            raise RuntimeError("Whisper 모델을 하나도 로드할 수 없습니다. VRAM을 확인하세요.")
        _pool_total = loaded
        print(f"  → TRANSCRIBE_WORKERS={loaded} 확정")


def get_pool_size() -> int:
    """실제 로드된 모델 인스턴스 수 반환 (init_model_pool 호출 후 사용)."""
    return _pool.qsize() if _pool is not None else 0


def release_model_pool():
    """풀의 모든 Whisper 모델을 언로드해 GPU 메모리를 해제한다."""
    global _pool
    with _pool_lock:
        if _pool is None:
            return
        freed = 0
        while not _pool.empty():
            try:
                _pool.get_nowait()
                freed += 1
            except Exception:
                break
        _pool = None
    if freed:
        print(f"  Whisper 모델 {freed}개 언로드 (GPU 메모리 해제)")


def _get_pool() -> queue.Queue:
    global _pool
    if _pool is None:
        init_model_pool(1)
    return _pool


def _extract_audio(video_path: str, wav_path: str,
                   start: float = 0.0, duration: float = None):
    """비디오에서 오디오 추출 (16kHz mono WAV). start/duration 으로 구간 지정 가능."""
    cmd = ["ffmpeg", "-y"]
    if start > 0.001:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [wav_path]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        # 타임아웃 시 빈 WAV로 처리 (호출자가 파일 크기로 판단)
        pass


def _is_cuda_oom(e: Exception) -> bool:
    err = str(e).lower()
    return any(k in err for k in ("out of memory", "cublas", "cuda"))


def transcribe(video_path: str, start: float = 0.0, duration: float = None,
               force_lang: str = None) -> dict:
    """
    음성 인식 실행, TranscriptDict 반환.
    CUDA OOM 발생 시 해당 인스턴스를 즉시 폐기(풀에 반납하지 않음)하고
    다음 가용 인스턴스로 재시도. 인스턴스가 모두 소진되면 예외 발생.
    """
    import time
    global _pool_total
    pool = _get_pool()

    while True:
        try:
            model = pool.get(timeout=120)  # 최대 2분 대기 (다른 워커 추론 완료 대기)
        except queue.Empty:
            raise RuntimeError("Whisper 모델 대기 타임아웃: 모든 인스턴스가 2분 이상 응답 없음.")

        discard = False
        try:
            result = _transcribe_with(model, video_path, start=start,
                                      duration=duration, force_lang=force_lang)
            return result
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                raise
            # OOM: 이 인스턴스를 즉시 폐기 (풀에 반납하지 않음)
            # → 같은 OOM 모델이 다른 워커에 재배분되는 것을 방지
            discard = True
            with _pool_total_lock:
                _pool_total -= 1
                remaining = _pool_total
            print(f"  [VRAM] OOM → 인스턴스 1개 폐기, 전체 {remaining}개 남음")
            del model
            if remaining == 0:
                raise RuntimeError("CUDA OOM: 인스턴스가 모두 소진됨. VRAM 부족.")
            time.sleep(2)  # 다른 워커들이 추론을 마칠 시간 확보 후 재시도
        finally:
            # OOM 폐기가 아닌 경우 반드시 모델 반납 (예외 종류 무관)
            if not discard:
                pool.put(model)


# Whisper 언어 코드 매핑 (config SUBTITLE_LANG → Whisper language code)
_LANG_MAP = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}


def _transcribe_with(model, video_path: str, start: float = 0.0,
                     duration: float = None, force_lang: str = None) -> dict:

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    whisper_lang = _LANG_MAP.get(force_lang)  # None이면 자동 감지

    try:
        _extract_audio(video_path, wav_path, start=start, duration=duration)

        if not Path(wav_path).exists() or Path(wav_path).stat().st_size < 1000:
            return _empty_transcript()

        def _run_transcribe(lang):
            kwargs = dict(
                language=lang,
                batch_size=4,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=600,
                    speech_pad_ms=400,
                ),
                word_timestamps=True,
                condition_on_previous_text=False,
            )
            return model.transcribe(wav_path, **kwargs)

        def _run_and_collect(lang) -> tuple:
            """transcribe 실행 + 세그먼트 eager 소비. OOM은 호출자(transcribe)가 처리."""
            seg_iter, inf = _run_transcribe(lang)
            return list(seg_iter), inf  # lazy generator 즉시 소비

        raw_segments, info = _run_and_collect(whisper_lang)

        # auto 모드: 한국어/영어 외 감지되면 한국어로 재시도
        # (distil 모델은 영어 전용이므로 재시도 불필요)
        is_distil = WHISPER_MODEL.lower().startswith("distil")
        if whisper_lang is None and not is_distil and info.language not in ("ko", "en"):
            raw_segments, info = _run_and_collect("ko")

        segments = []
        for seg in raw_segments:
            words = []
            if seg.words:
                words = [
                    {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                    for w in seg.words
                ]
            segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "avg_logprob": round(seg.avg_logprob, 4),
                "words": words,
            })

        # 의미 있는 음성 세그먼트만 필터
        speech_segs = [s for s in segments if s["no_speech_prob"] < 0.5 and s["text"]]
        total_speech = sum(s["end"] - s["start"] for s in speech_segs)

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "segments": segments,
            "has_speech": len(speech_segs) > 0,
            "total_speech_sec": round(total_speech, 2),
        }
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _empty_transcript() -> dict:
    return {
        "language": "unknown",
        "language_probability": 0.0,
        "segments": [],
        "has_speech": False,
        "total_speech_sec": 0.0,
    }
