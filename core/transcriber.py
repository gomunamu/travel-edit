"""faster-whisper 기반 음성 인식 (한국어/영어 자동 감지)"""
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_pool: Optional[queue.Queue] = None
_pool_lock = threading.Lock()
_pool_total = 0          # 체크아웃 중 포함 전체 살아있는 인스턴스 수
_pool_total_lock = threading.Lock()
_shrink_lock = threading.Lock()  # OOM 시 한 번에 1개만 폐기되도록 직렬화
_last_shrink_time = 0.0
_SHRINK_COOLDOWN = 20.0  # 폐기 후 최소 대기 시간(초) — 연속 과잉 폐기 방지

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

    if device == "cuda":
        if ctranslate2.get_cuda_device_count() == 0:
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


def init_model_pool(n: int = 0):
    """
    Whisper 모델 인스턴스를 풀에 적재.
    n=0 (auto): VRAM이 허용하는 한도까지 최대한 로드 (OOM 직전까지).
    n>0: 지정한 수만큼 시도, VRAM 부족 시 자동 축소.
    """
    global _pool
    with _pool_lock:
        if _pool is not None:
            return
        auto = (n == 0)
        target = _MAX_AUTO_WORKERS if auto else n
        label = f"최대 {target}개 (VRAM 자동)" if auto else f"최대 {target}개"
        print(f"  Whisper 모델 로드 중: {WHISPER_MODEL} × {label}")
        _pool = queue.Queue()
        loaded = 0
        for i in range(target):
            try:
                model, device = _load_one_model()
                _pool.put(model)
                loaded += 1
                print(f"  ✓ 인스턴스 {loaded} 로드 완료 ({device.upper()})")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if loaded == 0:
                        raise RuntimeError(
                            "Whisper 모델을 하나도 로드할 수 없습니다. VRAM을 확인하세요."
                        )
                    print(f"  VRAM 한계 도달 → {loaded}개 인스턴스로 확정")
                    break
                raise
        if loaded == 0:
            raise RuntimeError("Whisper 모델을 하나도 로드할 수 없습니다. VRAM을 확인하세요.")
        global _pool_total
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
    subprocess.run(cmd, capture_output=True, timeout=120)


def _is_cuda_oom(e: Exception) -> bool:
    err = str(e).lower()
    return any(k in err for k in ("out of memory", "cublas", "cuda"))


def transcribe(video_path: str, start: float = 0.0, duration: float = None,
               force_lang: str = None) -> dict:
    """
    음성 인식 실행, TranscriptDict 반환.
    CUDA OOM 발생 시 해당 인스턴스를 폐기(풀에 반납 안 함)하고
    다음 가용 인스턴스로 재시도. 인스턴스가 모두 소진되면 예외 발생.
    """
    import time
    global _pool_total, _last_shrink_time
    pool = _get_pool()

    while True:
        model = pool.get()  # 가용 인스턴스 대기
        try:
            result = _transcribe_with(model, video_path, start=start,
                                      duration=duration, force_lang=force_lang)
            pool.put(model)  # 성공: 반납
            return result
        except RuntimeError as e:
            if not _is_cuda_oom(e):
                pool.put(model)
                raise
            # OOM: 이 모델은 일단 풀에 반납
            pool.put(model)
            # 동시에 여러 워커가 OOM 나도 딱 1개만 폐기, 쿨다운 내엔 폐기 안 함
            if _shrink_lock.acquire(blocking=False):
                try:
                    now = time.time()
                    if now - _last_shrink_time >= _SHRINK_COOLDOWN:
                        try:
                            victim = pool.get_nowait()  # 유휴 인스턴스 1개 꺼내 폐기
                            with _pool_total_lock:
                                _pool_total -= 1
                                remaining_total = _pool_total
                            _last_shrink_time = now
                            print(f"  [VRAM] OOM → 인스턴스 1개 폐기, 전체 {remaining_total}개 남음")
                            del victim
                            if remaining_total == 0:
                                raise RuntimeError("CUDA OOM: 인스턴스가 모두 소진됨. VRAM 부족.")
                        except queue.Empty:
                            pass  # 모두 체크아웃 중 → 폐기 건너뜀
                    # else: 쿨다운 중 → 폐기 없이 재시도만
                finally:
                    _shrink_lock.release()
            time.sleep(3)  # 다른 워커들이 현재 추론을 마칠 시간


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
