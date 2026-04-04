"""AI 기반 클립 평가 - Claude / OpenAI / Gemini 라운드로빈"""
import json
import re
import threading
from contextlib import contextmanager
from typing import Optional, Tuple

from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    MIN_SEGMENT_DURATION, PURE_LANDSCAPE_THRESHOLD,
)
from core.token_tracker import tracker as _tracker

SYSTEM_PROMPT = """당신은 10년 경력의 여행 브이로그 편집자이자 방송국 PD입니다.
여행 동영상의 각 클립을 분석하고, 최종 편집본에 포함할지 결정합니다.
시청자가 지루하지 않도록, 재미있고 감동적인 장면만 선별하는 것이 목표입니다."""

EVAL_PROMPT = """다음 클립 정보를 분석하고 편집 결정과 품질 점수를 출력하세요.

## 클립 정보
- 길이: {duration:.1f}초
- 해상도: {width}x{height} {orientation}
- 음성 여부: {has_speech}
- 음성 구간 길이: {speech_sec:.1f}초

## 음성 전사
{transcript_text}

## 편집 판단 기준
1. 음성 없이 풍경만 {threshold}초 이상 → 지루하지 않게 잘라서 살리거나, 완전히 버리기
2. 재미있는 혼잣말/대화/반응 → 살리기
3. 같은 장면이 너무 길게 이어지면 → 앞뒤 잘라서 살리기
4. 너무 흔들리거나 무의미한 장면 → 버리기
5. 2초 미만 → 항상 버리기

## 품질 점수 기준 (편집 결정과 무관하게 독립적으로 평가)
각 항목을 0~25점으로 채점하세요.
- visual  (시각 품질): 화면 안정성, 노출·구도 완성도 (흔들림·역광·흐림 감점)
- speech  (음성·대화): 대화·혼잣말·감정 표현의 흥미도 (무음이면 0점 가능)
- scene   (장면 가치): 여행 장소·활동·경험의 희소성·감동 (평범한 이동 장면 감점)
- flow    (편집 흐름): 독립 장면으로의 완결성, 앞뒤 클립과 이어지기 쉬운 정도

## 응답 형식 (JSON만 출력)
{{
  "decision": "keep" | "trim" | "discard",
  "reason": "한국어로 상세한 이유 (장면 내용, 판단 근거, 편집 관점을 2~3문장으로)",
  "keep_start": 0.0,
  "keep_end": {duration:.1f},
  "score": {{
    "visual": 0~25,
    "speech": 0~25,
    "scene": 0~25,
    "flow": 0~25
  }}
}}

trim인 경우 keep_start와 keep_end를 반드시 지정하세요."""

# ─── 적응형 동시 요청 제어 (AIMD) ──────────────────────────────────────────
class AdaptiveConcurrency:
    """
    Rate limit에 반응해 동시 요청 수를 자동 조정.
    - 성공 increase_after번 연속 → +1 (max까지)
    - rate limit 발생 → 현재값 // 2 (min 이상)
    Condition variable 기반이라 스레드 수와 무관하게 동작.
    """
    def __init__(self, initial: int = 3, min_w: int = 1, max_w: int = 50,
                 increase_after: int = 5):
        self._limit = initial
        self._active = 0
        self._min = min_w
        self._max = max_w
        self._increase_after = increase_after
        self._consecutive_ok = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    @contextmanager
    def slot(self):
        with self._cv:
            while self._active >= self._limit:
                self._cv.wait()
            self._active += 1
        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def on_success(self):
        with self._cv:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self._increase_after and self._limit < self._max:
                self._limit += 1
                self._consecutive_ok = 0
                self._cv.notify_all()
                print(f"  [adaptive] 동시 요청 ↑ {self._limit}")

    def on_rate_limit(self):
        with self._cv:
            self._consecutive_ok = 0
            new = max(self._min, self._limit // 2)
            if new < self._limit:
                self._limit = new
                print(f"  [adaptive] rate limit → 동시 요청 ↓ {self._limit}")

    @property
    def current_limit(self) -> int:
        return self._limit


_adaptive = AdaptiveConcurrency(initial=3, min_w=1, max_w=50, increase_after=5)


# ─── API 가용성 관리 (rate limit 시 일시 비활성화) ─────────────────────────
_api_lock    = threading.Lock()
_disabled_until: dict = {}   # {"Claude": timestamp, ...}
_COOLDOWN = 60.0             # rate limit 후 재시도까지 대기 시간(초)


def _disable_api(name: str):
    import time
    with _api_lock:
        _disabled_until[name] = time.time() + _COOLDOWN


def _get_apis():
    """현재 활성화된 API 목록을 우선순위 순으로 반환."""
    import time
    now = time.time()
    all_apis = []
    if ANTHROPIC_API_KEY:
        all_apis.append(("Claude",  _call_claude))
    if OPENAI_API_KEY:
        all_apis.append(("OpenAI",  _call_openai))
    if GEMINI_API_KEY:
        all_apis.append(("Gemini",  _call_gemini))
    with _api_lock:
        available = [(n, fn) for n, fn in all_apis
                     if now >= _disabled_until.get(n, 0)]
    return available


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

    with _adaptive.slot():
        for name, caller in apis:
            result, rate_limited = caller(prompt, duration)
            if result is not None:
                _adaptive.on_success()
                return result
            if rate_limited:
                _adaptive.on_rate_limit()
                _disable_api(name)
                remaining = _get_apis()
                next_name = remaining[0][0] if remaining else "규칙 기반"
                print(f"  [rate limit] {name} → {next_name} 으로 전환 ({name} {_COOLDOWN:.0f}초 비활성화)")

        _adaptive.on_success()
        return _rule_based_eval(duration, has_speech, speech_sec)


# ─── API 호출 (result, is_rate_limited) 반환 ──────────────────────────────
def _call_claude(prompt: str, duration: float) -> Tuple[Optional[dict], bool]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        _tracker.record("Anthropic", CLAUDE_MODEL,
                        msg.usage.input_tokens, msg.usage.output_tokens)
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
            model=OPENAI_MODEL, max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        _tracker.record("OpenAI", OPENAI_MODEL,
                        msg.usage.prompt_tokens, msg.usage.completion_tokens)
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
        meta = response.usage_metadata
        _tracker.record("Gemini", GEMINI_MODEL,
                        meta.prompt_token_count, meta.candidates_token_count)
        return _parse_response(response.text, duration), False
    except Exception as e:
        if _is_rate_limit(e):
            return None, True
        print(f"  [경고] Gemini 평가 실패: {e}")
        return None, False


# ─── 공통 유틸 ───────────────────────────────────────────────────────────────
def _is_rate_limit(e: Exception) -> bool:
    """rate limit 또는 크레딧 소진 등 다음 API로 폴백해야 하는 에러."""
    s = str(e).lower()
    return any(k in s for k in (
        "429", "rate_limit", "rate limit", "quota", "resource_exhausted",
        "credit balance", "too low", "billing",
    ))


def _parse_response(text: str, duration: float) -> Optional[dict]:
    # 중첩 JSON을 포함한 전체 객체 추출
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        r = json.loads(match.group())
        raw_score = r.get("score", {})
        if isinstance(raw_score, dict):
            visual = max(0, min(25, int(raw_score.get("visual", 12))))
            speech = max(0, min(25, int(raw_score.get("speech", 12))))
            scene  = max(0, min(25, int(raw_score.get("scene",  12))))
            flow   = max(0, min(25, int(raw_score.get("flow",   12))))
        else:
            visual = speech = scene = flow = 12
        score = {
            "visual": visual, "speech": speech,
            "scene": scene, "flow": flow,
            "total": visual + speech + scene + flow,
        }
        return {
            "decision":   r.get("decision", "keep"),
            "reason":     r.get("reason", ""),
            "keep_start": float(r.get("keep_start", 0)),
            "keep_end":   float(r.get("keep_end", duration)),
            "score":      score,
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
        return _decision("trim", "음성 없는 긴 풍경 - 앞부분만 유지", 0, min(8.0, duration), 35)
    if has_speech and speech_sec > 2:
        return _decision("keep", "음성 포함", 0, duration, 60)
    if duration <= 10:
        return _decision("keep", "짧은 클립 유지", 0, duration, 50)
    return _decision("keep", "기본 유지", 0, duration, 50)


def _decision(decision: str, reason: str, start: float, end: float, total: int) -> dict:
    # total: 규칙 기반 추정 점수 (0-100), 세부 항목은 균등 분배
    q, r = divmod(max(0, min(100, total)), 4)
    score = {
        "visual": q + (1 if r > 0 else 0),
        "speech": q + (1 if r > 1 else 0),
        "scene":  q + (1 if r > 2 else 0),
        "flow":   q,
        "total":  total,
    }
    return {
        "decision": decision, "reason": reason,
        "keep_start": start,  "keep_end": end,
        "score": score,
    }
