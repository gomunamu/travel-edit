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


def _load_one_model():
    from faster_whisper import WhisperModel
    import ctranslate2

    device = WHISPER_DEVICE
    compute = WHISPER_COMPUTE_TYPE

    if device == "cuda":
        if ctranslate2.get_cuda_device_count() == 0:
            print("  [경고] CUDA 장치 없음, CPU로 전환")
            device = "cpu"
            compute = "int8"
        elif compute not in ctranslate2.get_supported_compute_types("cuda"):
            compute = "float16"

    return WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute,
        num_workers=1,
        cpu_threads=4,
    ), device


def init_model_pool(n: int = 1):
    """n개의 Whisper 모델 인스턴스를 미리 로드해 풀에 적재."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            return
        print(f"  Whisper 모델 로드 중: {WHISPER_MODEL} × {n}개 인스턴스")
        _pool = queue.Queue()
        for i in range(n):
            model, device = _load_one_model()
            _pool.put(model)
            print(f"  ✓ 인스턴스 {i+1}/{n} 로드 완료 ({device.upper()})")


def _get_pool() -> queue.Queue:
    global _pool
    if _pool is None:
        init_model_pool(1)
    return _pool


def _extract_audio(video_path: str, wav_path: str):
    """비디오에서 오디오 추출 (16kHz mono WAV)"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)


def transcribe(video_path: str, force_lang: str = None) -> dict:
    """음성 인식 실행, TranscriptDict 반환. 풀에서 모델을 빌려 쓰고 반납."""
    pool = _get_pool()
    model = pool.get()
    try:
        return _transcribe_with(model, video_path, force_lang)
    finally:
        pool.put(model)


# Whisper 언어 코드 매핑 (config SUBTITLE_LANG → Whisper language code)
_LANG_MAP = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}


def _transcribe_with(model, video_path: str, force_lang: str = None) -> dict:

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    whisper_lang = _LANG_MAP.get(force_lang)  # None이면 자동 감지

    try:
        _extract_audio(video_path, wav_path)

        if not Path(wav_path).exists() or Path(wav_path).stat().st_size < 1000:
            return _empty_transcript()

        segments_iter, info = model.transcribe(
            wav_path,
            language=whisper_lang,  # None=자동감지, 지정 시 해당 언어로 강제
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=600,
                speech_pad_ms=400,
            ),
            word_timestamps=True,
            condition_on_previous_text=True,
        )

        # auto 모드: 한국어/영어 외 감지되면 한국어로 재시도
        if whisper_lang is None and info.language not in ("ko", "en"):
            segments_iter, info = model.transcribe(
                wav_path,
                language="ko",
                beam_size=5,
                best_of=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=600,
                    speech_pad_ms=400,
                ),
                word_timestamps=True,
                condition_on_previous_text=True,
            )

        segments = []
        for seg in segments_iter:
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
