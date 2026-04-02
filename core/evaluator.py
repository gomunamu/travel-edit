"""AI 기반 클립 평가 - Claude / OpenAI / Gemini 라운드로빈"""
import json
import re
import threading
from typing import Optional, Tuple

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    MIN_SEGMENT_DURATION, PURE_LANDSCAPE_THRESHOLD,
)

SYSTEM_PROMPT = """당신은 10년 경력의 여행 브이로그 편집자이자 방송국 PD입니다.
여행 동영상의 각 클립을 분석하고, 최종 편집본에 포함할지 결정합니다.
시청자가 지루하지 않도록, 재미있고 감동적인 장면만 선별하는 것이 목표입니다."""

EVAL_PROMPT = """다음 클립 정보를 분석하고 편집 결정을 내려주세요.

## 클립 정보
- 길이: {duration:.1f}초
- 해상도: {width}x{height} {orientation}
- 음성 여부: {has_speech}
- 음성 구간 길이: {speech_sec:.1f}초

## 음성 전사
{transcript_text}

## 판단 기준
1. 음성 없이 풍경만 {threshold}초 이상 → 지루하지 않게 잘라서 살리거나, 완전히 버리기
2. 재미있는 혼잣말/대화/반응 → 살리기
3. 같은 장면이 너무 길게 이어지면 → 앞뒤 잘라서 살리기
4. 너무 흔들리거나 무의미한 장면 → 버리기
5. 2초 미만 → 항상 버리기

## 응답 형식 (JSON만 출력)
{{
  "decision": "keep" | "trim" | "discard",
  "reason": "한국어로 간단한 이유",
  "keep_start": 0.0,
  "keep_end": {duration:.1f},
  "interest_score": 1~10
}}

단, trim인 경우 keep_start와 keep_end를 반드시 지정하세요."""

# ─── 라운드로빈 상태 ────────────────────────────────────────────────────────
_rr_lock   = threading.Lock()
_rr_offset = 0   # rate limit 발생 시 +1 → 다음 API부터 시작


def _rotate():
    global _rr_offset
    with _rr_lock:
        _rr_offset += 1


def _get_apis():
    """설정된 API 목록을 현재 오프셋 순서로 반환."""
    all_apis = []
    if ANTHROPIC_API_KEY:
        all_apis.append(("Claude",  _call_claude))
    if OPENAI_API_KEY:
        all_apis.append(("OpenAI",  _call_openai))
    if GEMINI_API_KEY:
        all_apis.append(("Gemini",  _call_gemini))
    if not all_apis:
        return []
    with _rr_lock:
        start = _rr_offset % len(all_apis)
    return all_apis[start:] + all_apis[:start]


# ─── 평가 진입점 ─────────────────────────────────────────────────────────────
def evaluate_clip(clip: dict, transcript: dict) -> dict:
    duration = clip.get("duration", 0)

    if duration < MIN_SEGMENT_DURATION:
        return _decision("discard", "너무 짧음 (2초 미만)", 0, duration, 1)

    has_speech  = transcript.get("has_speech", False)
    speech_sec  = transcript.get("total_speech_sec", 0)
    is_portrait = clip.get("is_portrait", False)
    w = clip.get("display_width",  clip.get("raw_width",  1920))
    h = clip.get("display_height", clip.get("raw_height", 1080))

    transcript_text = _build_transcript_text(transcript)

    apis = _get_apis()
    if not apis:
        return _rule_based_eval(duration, has_speech, speech_sec)

    prompt = EVAL_PROMPT.format(
        duration=duration, width=w, height=h,
        orientation="(세로 영상)" if is_portrait else "(가로 영상)",
        has_speech="있음" if has_speech else "없음",
        speech_sec=speech_sec,
        transcript_text=transcript_text,
        threshold=PURE_LANDSCAPE_THRESHOLD,
    )

    for name, caller in apis:
        result, rate_limited = caller(prompt, duration)
        if result is not None:
            return result
        if rate_limited:
            _rotate()
            print(f"  [rate limit] {name} → 다음 API로 전환")

    return _rule_based_eval(duration, has_speech, speech_sec)


# ─── API 호출 (result, is_rate_limited) 반환 ──────────────────────────────
def _call_claude(prompt: str, duration: float) -> Tuple[Optional[dict], bool]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_response(msg.content[0].text.strip(), duration), False
    except Exception as e:
        if _is_rate_limit(e):
            return None, True
        print(f"  [경고] Claude 평가 실패: {e}")
        return None, False


def _call_openai(prompt: str, duration: float) -> Tuple[Optional[dict], bool]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        msg = client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return _parse_response(msg.choices[0].message.content.strip(), duration), False
    except Exception as e:
        if _is_rate_limit(e):
            return None, True
        print(f"  [경고] OpenAI 평가 실패: {e}")
        return None, False


def _call_gemini(prompt: str, duration: float) -> Tuple[Optional[dict], bool]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        return _parse_response(response.text, duration), False
    except Exception as e:
        if _is_rate_limit(e):
            return None, True
        print(f"  [경고] Gemini 평가 실패: {e}")
        return None, False


# ─── 공통 유틸 ───────────────────────────────────────────────────────────────
def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("429", "rate_limit", "rate limit", "quota", "resource_exhausted"))


def _parse_response(text: str, duration: float) -> Optional[dict]:
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        r = json.loads(match.group())
        return {
            "decision":       r.get("decision", "keep"),
            "reason":         r.get("reason", ""),
            "keep_start":     float(r.get("keep_start", 0)),
            "keep_end":       float(r.get("keep_end", duration)),
            "interest_score": int(r.get("interest_score", 5)),
        }
    except (json.JSONDecodeError, ValueError):
        return None


def _build_transcript_text(transcript: dict) -> str:
    segs = [s for s in transcript.get("segments", [])
            if s.get("no_speech_prob", 1) < 0.5 and s.get("text")]
    if not segs:
        return "(음성 없음)"
    return "\n".join(f"[{s['start']:.1f}s~{s['end']:.1f}s] {s['text']}" for s in segs)


def _rule_based_eval(duration: float, has_speech: bool, speech_sec: float) -> dict:
    if not has_speech and duration > PURE_LANDSCAPE_THRESHOLD:
        return _decision("trim", "음성 없는 긴 풍경 - 앞부분만 유지", 0, min(8.0, duration), 4)
    if has_speech and speech_sec > 2:
        return _decision("keep", "음성 포함", 0, duration, 7)
    if duration <= 10:
        return _decision("keep", "짧은 클립 유지", 0, duration, 5)
    return _decision("keep", "기본 유지", 0, duration, 5)


def _decision(decision: str, reason: str, start: float, end: float, score: int) -> dict:
    return {
        "decision": decision, "reason": reason,
        "keep_start": start,  "keep_end": end,
        "interest_score": score,
    }
