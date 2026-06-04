"""LLM 기반 STT 결과 정제 (한국어 오인식·외부 소음 보정)
폴백 체인: Claude (Anthropic) → OpenAI → Gemini → 원본 반환
"""
import json
import re
from typing import Optional

from config import (
    ANTHROPIC_API_KEY, STT_REFINE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
)
from core.token_tracker import tracker as _tracker


_SYSTEM_PROMPT = """\
당신은 한국어 STT(음성 인식) 교정 전문가입니다.
Whisper로 인식된 한국어 텍스트에는 다음과 같은 오류가 자주 발생합니다:
- 동음이의어 혼동 (예: "되" → "돼", "안 됩니다" → "않습니다")
- 외부 소음으로 인한 단어 삽입·누락
- 붙여쓰기/띄어쓰기 오류
- 문맥과 맞지 않는 단어 대체
- 고유명사(지명·브랜드) 오인식: 유사한 발음의 한국어 단어로 대체되는 경우
- Whisper hallucination: 단어·구절이 수십 회 반복 (예: "살금 살금 살금 살금...")
  → 반복은 자연스러운 횟수(1~2회)로 줄이거나 완전히 제거하세요

주어진 JSON 배열의 각 텍스트를 자연스러운 한국어로 교정하여 반환하세요.

규칙:
1. 텍스트 수를 변경하지 마세요 (입력과 동일한 개수 반환)
2. 명백한 오인식만 수정하고, 불확실하면 원문 유지
3. 말투(반말/존댓말)를 바꾸지 마세요
4. 의미를 추가하거나 요약하지 마세요
5. 결과는 반드시 JSON 배열로만 반환하세요 (설명 없이)
"""

_LOCATION_HINT_TEMPLATE = """\

## 촬영 장소 힌트
이 클립은 다음 지역에서 촬영되었습니다: {locations}
Whisper가 이 지명들을 발음이 비슷한 한국어 단어로 잘못 인식했을 수 있습니다.
텍스트에서 이 지명의 한국어 외래어 표기(예: Queenstown→퀸스타운)와 유사한 오인식이 보이면 올바른 지명으로 교정하세요.
지명이 등장하지 않는 경우 이 힌트는 무시하세요.\
"""


def remove_repetitions(text: str, threshold: int = 4) -> str:
    """
    Whisper hallucination 제거: 단어·구절이 threshold회 이상 연속 반복되면 1회로 축소.

    "살금 살금 살금 살금 살금..." → "살금"
    "레스토랑 레스토랑 레스토랑..." → "레스토랑"

    LLM 전에 적용해 토큰 낭비와 불안정한 응답을 방지.
    """
    changed = True
    while changed:
        changed = False
        # n-gram 길이 4→1 순서로 (긴 패턴 우선)
        for n in range(4, 0, -1):
            if n == 1:
                pat = r'(\S+)(?:\s+\1){' + str(threshold - 1) + r',}'
            else:
                pat = r'(' + r'\s+'.join([r'\S+'] * n) + r')(?:\s+\1){' + str(threshold - 1) + r',}'
            new = re.sub(pat, r'\1', text)
            if new != text:
                text = new
                changed = True
    return text.strip()


def _build_messages(texts: list, system: str) -> tuple:
    """(system_prompt, user_content) 반환."""
    user_content = f"다음 STT 텍스트를 교정하세요:\n{json.dumps(texts, ensure_ascii=False)}"
    return system, user_content


def _parse_corrected(raw: str, expected_len: int) -> Optional[list]:
    """LLM 응답에서 JSON 배열 파싱. 개수 불일치 시 None."""
    try:
        corrected = json.loads(raw)
        if isinstance(corrected, list) and len(corrected) == expected_len:
            return corrected
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# 잘못된 키(인증 오류)로 비활성화된 provider — 세션 동안 재시도하지 않음.
# (대기해도 회복되지 않으므로 클립마다 무의미한 호출을 막는다)
_auth_disabled: set = set()


def _is_auth_error(e: Exception) -> bool:
    """잘못된/만료된 키 등 대기해도 회복되지 않는 인증 오류."""
    s = str(e).lower()
    return any(k in s for k in (
        "401", "403", "unauthorized", "authentication", "invalid_api_key",
        "invalid api key", "incorrect api key", "api key not valid",
        "invalid x-api-key", "permission_denied", "permission denied",
    ))


def _note_auth_fail(name: str):
    if name not in _auth_disabled:
        _auth_disabled.add(name)
        print(f"  [자막교정] {name} 인증 오류(잘못된 키) → 이번 세션 사용 중단")


def _call_claude(texts: list, system: str) -> Optional[list]:
    if not ANTHROPIC_API_KEY or "Claude" in _auth_disabled:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        _, user_content = _build_messages(texts, system)
        message = client.messages.create(
            model=STT_REFINE_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        _tracker.record("Anthropic", STT_REFINE_MODEL,
                        message.usage.input_tokens, message.usage.output_tokens)
        return _parse_corrected(message.content[0].text.strip(), len(texts))
    except Exception as e:
        if _is_auth_error(e):
            _note_auth_fail("Claude")
        return None


def _call_openai(texts: list, system: str) -> Optional[list]:
    if not OPENAI_API_KEY or "OpenAI" in _auth_disabled:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        _, user_content = _build_messages(texts, system)
        msg = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
        )
        _tracker.record("OpenAI", OPENAI_MODEL,
                        msg.usage.prompt_tokens, msg.usage.completion_tokens)
        return _parse_corrected(msg.choices[0].message.content.strip(), len(texts))
    except Exception as e:
        if _is_auth_error(e):
            _note_auth_fail("OpenAI")
        return None


def _call_gemini(texts: list, system: str) -> Optional[list]:
    if not GEMINI_API_KEY or "Gemini" in _auth_disabled:
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        _, user_content = _build_messages(texts, system)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=4096,
            ),
        )
        meta = response.usage_metadata
        _tracker.record("Gemini", GEMINI_MODEL,
                        meta.prompt_token_count, meta.candidates_token_count)
        return _parse_corrected(response.text, len(texts))
    except Exception as e:
        if _is_auth_error(e):
            _note_auth_fail("Gemini")
        return None


def refine_transcript(
    transcript: dict,
    location_hints: Optional[list] = None,
) -> dict:
    """
    transcript의 segments 텍스트를 LLM으로 정제한다.
    폴백 체인: Claude → OpenAI → Gemini. 모두 실패 시 원본 반환.
    has_speech=False 또는 segments가 없으면 원본 그대로 반환.
    location_hints: GPS에서 추출한 지역명 목록 (국가 제외). 지명 오인식 교정에 활용.
    """
    if not transcript.get("has_speech") or not transcript.get("segments"):
        return transcript

    segments = transcript["segments"]
    texts = [remove_repetitions(s.get("text", "").strip()) for s in segments]

    if not any(texts):
        return transcript

    system = _SYSTEM_PROMPT
    if location_hints:
        system += _LOCATION_HINT_TEMPLATE.format(
            locations=", ".join(location_hints)
        )

    corrected = (
        _call_claude(texts, system)
        or _call_openai(texts, system)
        or _call_gemini(texts, system)
    )

    if corrected is None:
        # CPU 폴백: LLM 없이 반복 제거만 적용
        corrected = texts  # remove_repetitions는 이미 texts 생성 시 적용됨

    refined = dict(transcript)
    refined["segments"] = [
        dict(seg, text=corrected[i]) for i, seg in enumerate(segments)
    ]
    return refined
