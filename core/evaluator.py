"""AI 기반 클립 평가
폴백 체인: Claude (Anthropic) → OpenAI → Gemini → 규칙 기반 평가
(rate limit·실패 시 다음 API로 순차 강등, refiner.py와 동일한 패턴)
"""
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
    EDIT_STYLE, STYLE_MAX_LANDSCAPE, STYLE_DISCARD_SILENT,
)
from core.token_tracker import tracker as _tracker
from core.refiner import remove_repetitions

# 스타일별 system prompt — scene 계열은 다큐멘터리 관점으로 전환
_SYSTEM_PROMPTS = {
    "scene-long": (
        "당신은 여행 다큐멘터리 촬영감독입니다.\n"
        "여행지의 풍경·자연·분위기를 충분히 담는 것이 목표입니다.\n"
        "무음 풍경은 결함이 아니라 핵심 콘텐츠입니다. "
        "음성이 없어도 시각적으로 아름답거나 장소감이 느껴지는 클립은 적극적으로 살립니다.\n"
        "버림 기준은 오직 심한 흔들림·초점 불량·완전히 무의미한 장면에만 한정합니다."
    ),
    "scene-short": (
        "당신은 여행 영상 편집자입니다.\n"
        "풍경과 분위기를 간결하게 담는 것이 목표입니다.\n"
        "무음 풍경도 핵심 장면이면 살리되, 너무 길면 앞뒤를 잘라 핵심만 남깁니다.\n"
        "음성이 없다는 이유만으로 버리지 마세요."
    ),
    "default": (
        "당신은 10년 경력의 여행 브이로그 편집자이자 방송국 PD입니다.\n"
        "여행 동영상의 각 클립을 분석하고, 최종 편집본에 포함할지 결정합니다.\n"
        "시청자가 지루하지 않도록, 재미있고 감동적인 장면만 선별하는 것이 목표입니다."
    ),
}

# 기본 system prompt (하위 호환)
SYSTEM_PROMPT = _SYSTEM_PROMPTS["default"]

# 스타일별 AI 지침
_STYLE_GUIDE = {
    "balanced": (
        "음성과 풍경을 균형 있게 선별하세요. "
        "재미있는 대화·반응과 감동적인 풍경을 동등하게 평가합니다."
    ),
    "voice": (
        "음성·대화가 있는 클립만 남기세요. "
        "무음 풍경은 예외 없이 버리세요. "
        "말하는 장면·감정 표현·현장 반응만으로 편집본을 구성합니다."
    ),
    "scene-short": (
        "풍경을 포함하되 간결하게 유지하세요. "
        "무음 풍경은 최대 {max_keep}초 이내로 트림하고, 가장 핵심 장면만 남기세요. "
        "음성 없음은 감점 사유가 아닙니다."
    ),
    "scene-long": (
        "풍경·자연·분위기 클립을 최대한 살리는 것이 이 스타일의 목적입니다. "
        "무음 풍경은 기본적으로 keep 또는 trim이며, "
        "discard는 오직 심한 흔들림·초점 불량·렌즈 가림처럼 화질 자체가 불량한 경우에만 사용합니다. "
        "【중요】음성이 없다는 이유만으로 절대 discard를 내리지 마세요. "
        "{max_keep}초가 넘는 클립은 앞뒤를 트림해 {max_keep}초로 맞추되, 버리지는 마세요. "
        "음성 유무는 결정에 영향을 주지 않습니다."
    ),
    "highlight": (
        "최고 품질의 클립만 엄선하세요. "
        "visual·scene 점수가 낮거나 흔들리는 장면은 과감히 버리세요. "
        "소셜 미디어 하이라이트 릴 수준의 완성도를 기준으로 삼으세요."
    ),
    "vlog": (
        "말하는 장면 중심이지만 짧은 장면 전환 컷은 살립니다. "
        "카메라를 향해 말하거나 감정을 표현하는 장면을 우선하고, "
        "무음 풍경은 최대 {max_keep}초 이내로 트림해 전환 컷으로 활용하세요. "
        "voice와 달리 무음 클립을 버리지 않고 편집 흐름에 녹여넣는 것이 핵심입니다."
    ),
}

# 스타일별 규칙 1 (무음 풍경 처리 방침이 스타일마다 반대이므로 분리)
_RULE1 = {
    "scene-long": (
        "음성 없는 풍경·자연·분위기 클립 → 기본적으로 **keep** (음성 없음은 discard 사유 아님). "
        "{max_keep}초 초과 시 {max_keep}초로 트림. "
        "discard 허용 사유: 심한 흔들림 / 심각한 초점 불량 / 렌즈 가림 / 완전히 무의미한 장면(주차장 바닥·발바닥 등). "
        "단순히 '음성이 없다', '지루할 수 있다', '장소감이 부족하다'는 discard 사유로 인정하지 않습니다."
    ),
    "scene-short": (
        "음성 없는 풍경 → 핵심 구간만 남겨 최대 {max_keep}초로 트림. "
        "장소감·분위기가 전혀 없는 장면만 버림."
    ),
    "voice": (
        "음성 없는 클립 → 예외 없이 버림."
    ),
    "default": (
        "음성 없이 풍경만 {threshold}초 이상 → 지루하지 않게 트림하거나, "
        "장소감·분위기가 없으면 버리기."
    ),
}

EVAL_PROMPT = """다음 클립 정보를 분석하고 편집 결정과 품질 점수를 출력하세요.

## 편집 스타일
{style_guide}

## 클립 정보
- 길이: {duration:.1f}초
- 해상도: {width}x{height} {orientation}
- 음성 여부: {has_speech}
- 음성 구간 길이: {speech_sec:.1f}초

## 음성 전사
{transcript_text}

## 편집 판단 기준
1. {rule1}
2. 재미있는 혼잣말/대화/반응 → 살리기
3. 같은 장면이 너무 길게 이어지면 → 앞뒤 잘라서 살리기
4. 너무 흔들리거나 무의미한 장면 → 버리기
5. 2초 미만 → 항상 버리기

## 품질 점수 기준 (편집 결정과 무관하게 독립적으로 평가)
각 항목을 0~25점으로 채점하세요.
- visual  (시각 품질): 화면 안정성, 노출·구도 완성도 (흔들림·역광·흐림 감점)
- speech  (음성·대화): 대화·혼잣말·감정 표현의 흥미도 (무음이면 0점 가능)
- scene   (장면 가치): 여행 장소·활동·경험의 희소성·감동 ({scene_note})
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

# scene 점수 안내 (스타일별)
_SCENE_NOTE = {
    "scene-long":  "무음 풍경도 장소감·자연미가 있으면 높은 점수 부여",
    "scene-short": "무음 풍경도 장소감이 있으면 정상 점수 부여",
    "default":     "평범한 이동 장면 감점",
}

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

    def on_rate_limit(self):
        with self._cv:
            self._consecutive_ok = 0
            new = max(self._min, self._limit // 2)
            if new < self._limit:
                self._limit = new

    @property
    def current_limit(self) -> int:
        return self._limit


_adaptive = AdaptiveConcurrency(initial=3, min_w=1, max_w=50, increase_after=5)


# ─── API 가용성 관리 ─────────────────────────────────────────────────────────
# 우선순위: Claude → OpenAI → Gemini (순차 강등)
# - rate limit / 일시 실패: 쿨다운 후 재시도, 3회 누적 시 세션 내 비활성화.
#   파일 단위로 조건부 리셋 (rate limit 회복 가정).
# - 인증 오류(잘못된 키): 대기해도 회복 불가 → 즉시 세션 영구 비활성화.
#   파일 리셋에도 되살리지 않아 클립·파일마다 무의미한 재시도를 막는다.
_api_lock       = threading.Lock()
_disabled_until: dict = {}   # {"Claude": timestamp or inf}  (쿨다운/누적 실패)
_fail_count:     dict = {}   # {"Claude": int}
_auth_disabled:  set  = set()  # 잘못된 키 — 세션 영구 비활성화 (리셋에도 유지)
_COOLDOWN        = 5.0       # 실패 후 재시도 대기(초)
_MAX_FAILS       = 3         # 이 횟수 도달 시 현재 파일 내 비활성화


def reset_api_state():
    """파일 경계에서 API 상태를 조건부 리셋한다.

    - 활성 API가 하나라도 남아있으면 리셋하지 않는다.
      (예: Claude 실패 후 OpenAI가 작동 중이면 Claude를 재시도하지 않음)
    - 모든 API가 비활성화된 경우에만 리셋한다.
      (rate limit이 회복됐을 가능성이 있으므로 다음 파일에서 재시도)
    - 인증 오류로 비활성화된 키(_auth_disabled)는 리셋 대상에서 제외한다.
      (잘못된 키는 다음 파일에서도 회복되지 않으므로 재시도하지 않음)
    """
    import time
    now = time.time()
    configured = []
    if ANTHROPIC_API_KEY:
        configured.append("Claude")
    if OPENAI_API_KEY:
        configured.append("OpenAI")
    if GEMINI_API_KEY:
        configured.append("Gemini")

    with _api_lock:
        live = [n for n in configured if n not in _auth_disabled]
        has_active = any(now >= _disabled_until.get(n, 0) for n in live)
        if live and not has_active:
            # 쿨다운/누적 실패만 리셋, 인증 오류 비활성화는 유지
            _disabled_until.clear()
            _fail_count.clear()
            print("  [API 리셋] 모든 API 비활성화 상태 → 다음 파일에서 재시도")


def _on_api_fail(name: str):
    import time
    with _api_lock:
        count = _fail_count.get(name, 0) + 1
        _fail_count[name] = count
        if count >= _MAX_FAILS:
            _disabled_until[name] = float("inf")
            print(f"  [API 비활성화] {name} {count}회 실패 → 이번 세션 사용 중단")
        else:
            _disabled_until[name] = time.time() + _COOLDOWN
            print(f"  [API 쿨다운] {name} {_COOLDOWN:.0f}초 대기 (실패 {count}/{_MAX_FAILS})")


def _on_auth_fail(name: str):
    """잘못된 키(인증 오류) → 세션 동안 영구 비활성화. 파일 리셋에도 유지."""
    with _api_lock:
        already = name in _auth_disabled
        _auth_disabled.add(name)
    if not already:
        print(f"  [API 비활성화] {name} 인증 오류(잘못된 키) → 이번 세션 사용 중단")


def _get_apis():
    """우선순위(Claude→OpenAI→Gemini) 순으로 현재 활성화된 API 목록 반환."""
    import time
    now = time.time()
    candidates = []
    if ANTHROPIC_API_KEY:
        candidates.append(("Claude", _call_claude))
    if OPENAI_API_KEY:
        candidates.append(("OpenAI", _call_openai))
    if GEMINI_API_KEY:
        candidates.append(("Gemini", _call_gemini))
    with _api_lock:
        return [(n, fn) for n, fn in candidates
                if n not in _auth_disabled and now >= _disabled_until.get(n, 0)]


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

    style_guide = _STYLE_GUIDE.get(EDIT_STYLE, _STYLE_GUIDE["balanced"]).format(
        max_keep=STYLE_MAX_LANDSCAPE
    )
    rule1 = _RULE1.get(EDIT_STYLE, _RULE1["default"]).format(
        threshold=PURE_LANDSCAPE_THRESHOLD,
        max_keep=STYLE_MAX_LANDSCAPE,
    )
    scene_note = _SCENE_NOTE.get(EDIT_STYLE, _SCENE_NOTE["default"])
    system = _SYSTEM_PROMPTS.get(EDIT_STYLE, _SYSTEM_PROMPTS["default"])

    prompt = EVAL_PROMPT.format(
        duration=duration, width=w, height=h,
        orientation="(세로 영상)" if is_portrait else "(가로 영상)",
        has_speech="있음" if has_speech else "없음",
        speech_sec=speech_sec,
        transcript_text=transcript_text,
        threshold=PURE_LANDSCAPE_THRESHOLD,
        style_guide=style_guide,
        rule1=rule1,
        scene_note=scene_note,
    )

    with _adaptive.slot():
        # 슬롯 진입 직후 최신 가용 API 목록 조회 (비활성화 반영)
        for name, caller in _get_apis():
            result, fail = caller(prompt, duration, system=system)
            if result is not None:
                _adaptive.on_success()
                return _scene_override(result, has_speech, duration)
            if fail == "rate_limit":
                _adaptive.on_rate_limit()
                _on_api_fail(name)
                remaining = _get_apis()
                next_name = remaining[0][0] if remaining else "규칙 기반"
                print(f"  [rate limit] {name} → {next_name} 으로 전환")
            elif fail == "auth":
                _on_auth_fail(name)
                remaining = _get_apis()
                next_name = remaining[0][0] if remaining else "규칙 기반"
                print(f"  [인증 오류] {name} → {next_name} 으로 전환")

        _adaptive.on_success()
        return _rule_based_eval(duration, has_speech, speech_sec)


def _scene_override(result: dict, has_speech: bool, duration: float) -> dict:
    """scene 스타일에서 LLM이 무음 클립에 discard를 내리면 trim으로 교정한다.
    프롬프트 지시를 LLM이 무시하는 경우에 대한 안전망."""
    if EDIT_STYLE not in ("scene-long", "scene-short"):
        return result
    if result.get("decision") != "discard":
        return result
    if has_speech:
        return result  # 음성 있는 클립은 LLM 판단 존중
    # 무음 클립을 discard → trim으로 교정
    max_keep = float(STYLE_MAX_LANDSCAPE)
    keep_end = min(max_keep, duration)
    return {
        **result,
        "decision":   "trim",
        "keep_start": 0.0,
        "keep_end":   keep_end,
        "reason":     "(scene 스타일 보정) " + result.get("reason", ""),
    }


# ─── 공통: 프롬프트 sanitize ─────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    """JSON/HTTP body를 깨뜨리는 제어 문자 제거. 탭(\\t)·줄바꿈(\\n\\r)은 유지."""
    import re
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)


# ─── API 호출 (result, fail_kind) 반환 ────────────────────────────────────
# fail_kind: None(성공/일시오류) | "rate_limit"(쿨다운) | "auth"(잘못된 키, 영구 비활성화)
def _call_claude(prompt: str, duration: float, system: str = SYSTEM_PROMPT) -> Tuple[Optional[dict], Optional[str]]:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=512,
            system=_sanitize(system),
            messages=[{"role": "user", "content": _sanitize(prompt)}],
        )
        _tracker.record("Anthropic", CLAUDE_MODEL,
                        msg.usage.input_tokens, msg.usage.output_tokens)
        return _parse_response(msg.content[0].text.strip(), duration), None
    except Exception as e:
        return None, _classify_error(e, "Claude")


def _call_openai(prompt: str, duration: float, system: str = SYSTEM_PROMPT) -> Tuple[Optional[dict], Optional[str]]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        msg = client.chat.completions.create(
            model=OPENAI_MODEL, max_tokens=512,
            messages=[
                {"role": "system", "content": _sanitize(system)},
                {"role": "user",   "content": _sanitize(prompt)},
            ],
        )
        _tracker.record("OpenAI", OPENAI_MODEL,
                        msg.usage.prompt_tokens, msg.usage.completion_tokens)
        return _parse_response(msg.choices[0].message.content.strip(), duration), None
    except Exception as e:
        return None, _classify_error(e, "OpenAI")


def _call_gemini(prompt: str, duration: float, system: str = SYSTEM_PROMPT) -> Tuple[Optional[dict], Optional[str]]:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_sanitize(prompt),
            config=types.GenerateContentConfig(
                system_instruction=_sanitize(system),
                max_output_tokens=512,
            ),
        )
        meta = response.usage_metadata
        _tracker.record("Gemini", GEMINI_MODEL,
                        meta.prompt_token_count, meta.candidates_token_count)
        return _parse_response(response.text, duration), None
    except Exception as e:
        return None, _classify_error(e, "Gemini")


# ─── 공통 유틸 ───────────────────────────────────────────────────────────────
def _classify_error(e: Exception, name: str) -> Optional[str]:
    """예외를 폴백 종류로 분류. rate_limit > auth 우선, 그 외는 일시 오류로 경고만."""
    if _is_rate_limit(e):
        return "rate_limit"
    if _is_auth_error(e):
        return "auth"
    print(f"  [경고] {name} 평가 실패: {e}")
    return None


def _is_rate_limit(e: Exception) -> bool:
    """rate limit 또는 크레딧 소진 등 다음 API로 폴백해야 하는 에러."""
    s = str(e).lower()
    return any(k in s for k in (
        "429", "rate_limit", "rate limit", "quota", "resource_exhausted",
        "credit balance", "too low", "billing",
    ))


def _is_auth_error(e: Exception) -> bool:
    """잘못된/만료된 키 등 대기해도 회복되지 않는 인증 오류.
    rate limit 판별 이후에 호출되므로 quota성 403은 이미 걸러진 상태."""
    s = str(e).lower()
    return any(k in s for k in (
        "401", "403", "unauthorized", "authentication", "invalid_api_key",
        "invalid api key", "incorrect api key", "api key not valid",
        "invalid x-api-key", "permission_denied", "permission denied",
    ))


def _parse_response(text: str, duration: float) -> Optional[dict]:
    # 중첩 JSON을 포함한 전체 객체 추출
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        r = json.loads(match.group())
        def _safe_int(val, default: int) -> int:
            try:
                return int(val) if val is not None else default
            except (ValueError, TypeError):
                return default

        def _safe_float(val, default: float) -> float:
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default

        raw_score = r.get("score", {})
        if isinstance(raw_score, dict):
            visual = max(0, min(25, _safe_int(raw_score.get("visual"), 12)))
            speech = max(0, min(25, _safe_int(raw_score.get("speech"), 12)))
            scene  = max(0, min(25, _safe_int(raw_score.get("scene"),  12)))
            flow   = max(0, min(25, _safe_int(raw_score.get("flow"),   12)))
        else:
            visual = speech = scene = flow = 12
        score = {
            "visual": visual, "speech": speech,
            "scene": scene, "flow": flow,
            "total": visual + speech + scene + flow,
        }
        return {
            "decision":   r.get("decision") or "keep",
            "reason":     r.get("reason") or "",
            "keep_start": _safe_float(r.get("keep_start"), 0.0),
            "keep_end":   _safe_float(r.get("keep_end"),   duration),
            "score":      score,
        }
    except (json.JSONDecodeError, ValueError):
        return None


def _build_transcript_text(transcript: dict) -> str:
    segs = [s for s in transcript.get("segments", [])
            if s.get("no_speech_prob", 1) < 0.5 and s.get("text")]
    if not segs:
        return "(음성 없음)"
    return "\n".join(
        f"[{s['start']:.1f}s~{s['end']:.1f}s] {remove_repetitions(s['text'])}" for s in segs
    )


def _rule_based_eval(duration: float, has_speech: bool, speech_sec: float) -> dict:
    if not has_speech:
        if STYLE_DISCARD_SILENT and duration > PURE_LANDSCAPE_THRESHOLD:
            return _decision("discard", f"무음 풍경 ({EDIT_STYLE} 스타일: 무음 클립 제거)", 0, duration, 10)
        if duration > PURE_LANDSCAPE_THRESHOLD:
            keep = min(float(STYLE_MAX_LANDSCAPE), duration)
            return _decision("trim", f"무음 풍경 트림 ({EDIT_STYLE}: 최대 {STYLE_MAX_LANDSCAPE}초)", 0, keep, 35)
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
