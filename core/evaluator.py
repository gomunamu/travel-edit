"""Claude AI 기반 클립 평가 - 브이로그 편집자/PD 시점"""
import json
import re
from typing import Optional

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
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


def _build_transcript_text(transcript: dict) -> str:
    segments = transcript.get("segments", [])
    speech_segs = [s for s in segments if s.get("no_speech_prob", 1) < 0.5 and s.get("text")]
    if not speech_segs:
        return "(음성 없음)"
    lines = []
    for s in speech_segs:
        lines.append(f"[{s['start']:.1f}s~{s['end']:.1f}s] {s['text']}")
    return "\n".join(lines)


def evaluate_clip(clip: dict, transcript: dict) -> dict:
    """클립 평가 실행"""
    duration = clip.get("duration", 0)

    # 즉시 버리기: 너무 짧음
    if duration < MIN_SEGMENT_DURATION:
        return _decision("discard", "너무 짧음 (2초 미만)", 0, duration, 1)

    has_speech = transcript.get("has_speech", False)
    speech_sec = transcript.get("total_speech_sec", 0)
    is_portrait = clip.get("is_portrait", False)
    w = clip.get("display_width", clip.get("raw_width", 1920))
    h = clip.get("display_height", clip.get("raw_height", 1080))

    transcript_text = _build_transcript_text(transcript)

    # API 키 없으면 규칙 기반 fallback
    if not ANTHROPIC_API_KEY:
        return _rule_based_eval(duration, has_speech, speech_sec)

    prompt = EVAL_PROMPT.format(
        duration=duration,
        width=w,
        height=h,
        orientation="(세로 영상)" if is_portrait else "(가로 영상)",
        has_speech="있음" if has_speech else "없음",
        speech_sec=speech_sec,
        transcript_text=transcript_text,
        threshold=PURE_LANDSCAPE_THRESHOLD,
    )

    result = _call_claude(prompt, duration)
    if result is not None:
        return result

    # Claude 실패(rate limit 등) → OpenAI 폴백
    if OPENAI_API_KEY:
        result = _call_openai(prompt, duration)
        if result is not None:
            return result

    return _rule_based_eval(duration, has_speech, speech_sec)


def _parse_response(text: str, duration: float) -> Optional[dict]:
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        return {
            "decision": result.get("decision", "keep"),
            "reason": result.get("reason", ""),
            "keep_start": float(result.get("keep_start", 0)),
            "keep_end": float(result.get("keep_end", duration)),
            "interest_score": int(result.get("interest_score", 5)),
        }
    except (json.JSONDecodeError, ValueError):
        return None


def _call_claude(prompt: str, duration: float) -> Optional[dict]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse_response(message.content[0].text.strip(), duration)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err:
            print(f"  [rate limit] Claude → OpenAI 폴백")
        else:
            print(f"  [경고] Claude 평가 실패: {e}")
        return None


def _call_openai(prompt: str, duration: float) -> Optional[dict]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        message = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return _parse_response(message.choices[0].message.content.strip(), duration)
    except Exception as e:
        print(f"  [경고] OpenAI 평가 실패: {e}")
        return None


def _rule_based_eval(duration: float, has_speech: bool, speech_sec: float) -> dict:
    """API 없을 때 규칙 기반 평가"""
    if not has_speech and duration > PURE_LANDSCAPE_THRESHOLD:
        # 풍경 긴 클립: 앞 8초만 유지
        keep_end = min(8.0, duration)
        return _decision("trim", "음성 없는 긴 풍경 - 앞부분만 유지", 0, keep_end, 4)
    if has_speech and speech_sec > 2:
        return _decision("keep", "음성 포함", 0, duration, 7)
    if duration <= 10:
        return _decision("keep", "짧은 클립 유지", 0, duration, 5)
    return _decision("keep", "기본 유지", 0, duration, 5)


def _decision(decision: str, reason: str, start: float, end: float, score: int) -> dict:
    return {
        "decision": decision,
        "reason": reason,
        "keep_start": start,
        "keep_end": end,
        "interest_score": score,
    }
