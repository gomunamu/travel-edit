"""LLM 기반 STT 결과 정제 (한국어 오인식·외부 소음 보정)"""
import json
import re
from typing import Optional

from core.token_tracker import tracker as _tracker


_SYSTEM_PROMPT = """\
당신은 한국어 STT(음성 인식) 교정 전문가입니다.
Whisper로 인식된 한국어 텍스트에는 다음과 같은 오류가 자주 발생합니다:
- 동음이의어 혼동 (예: "되" → "돼", "안 됩니다" → "않습니다")
- 외부 소음으로 인한 단어 삽입·누락
- 붙여쓰기/띄어쓰기 오류
- 문맥과 맞지 않는 단어 대체
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


def refine_transcript(transcript: dict, api_key: str, model: str) -> dict:
    """
    transcript의 segments 텍스트를 LLM으로 정제한다.
    has_speech=False 또는 segments가 없으면 원본 그대로 반환.
    """
    if not transcript.get("has_speech") or not transcript.get("segments"):
        return transcript

    segments = transcript["segments"]
    texts = [remove_repetitions(s.get("text", "").strip()) for s in segments]

    if not any(texts):
        return transcript

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"다음 STT 텍스트를 교정하세요:\n{json.dumps(texts, ensure_ascii=False)}",
                }
            ],
        )
        _tracker.record("Anthropic", model,
                        message.usage.input_tokens, message.usage.output_tokens)
        raw = message.content[0].text.strip()

        # JSON 배열 파싱
        corrected = json.loads(raw)
        if not isinstance(corrected, list) or len(corrected) != len(segments):
            return transcript  # 파싱 실패 시 원본 반환

        refined = dict(transcript)
        refined["segments"] = [
            dict(seg, text=corrected[i]) for i, seg in enumerate(segments)
        ]
        return refined

    except Exception:
        return transcript  # 오류 시 원본 반환
