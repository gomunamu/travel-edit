"""API 토큰 사용량 추적 (Anthropic / OpenAI / Gemini)"""
import threading
from dataclasses import dataclass, field
from typing import Dict

# 모델별 1M 토큰당 USD 단가 (input, output)
# 가격은 변동될 수 있으므로 참고용 추정치입니다.
_PRICE_PER_M: Dict[str, tuple] = {
    # Anthropic
    "claude-opus-4-6":             (15.00, 75.00),
    "claude-sonnet-4-6":           (3.00,  15.00),
    "claude-haiku-4-5-20251001":   (0.80,   4.00),
    "claude-haiku-4-5":            (0.80,   4.00),
    "claude-3-5-haiku-20241022":   (0.80,   4.00),
    "claude-3-haiku-20240307":     (0.25,   1.25),
    # OpenAI
    "gpt-4o":                      (5.00,  15.00),
    "gpt-4o-mini":                 (0.15,   0.60),
    "gpt-4-turbo":                 (10.00, 30.00),
    "gpt-3.5-turbo":               (0.50,   1.50),
    # Gemini
    "gemini-1.5-flash":            (0.075,  0.30),
    "gemini-1.5-pro":              (3.50,  10.50),
    "gemini-2.0-flash":            (0.10,   0.40),
}


def _cost_usd(model: str, input_tok: int, output_tok: int) -> float:
    key = model.lower()
    # 정확히 일치하는 키 먼저, 없으면 prefix 매칭
    rates = _PRICE_PER_M.get(key)
    if rates is None:
        for k, v in _PRICE_PER_M.items():
            if key.startswith(k) or k.startswith(key):
                rates = v
                break
    if rates is None:
        return 0.0
    in_price, out_price = rates
    return (input_tok * in_price + output_tok * out_price) / 1_000_000


@dataclass
class _ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class TokenTracker:
    """전역 싱글턴 토큰 사용량 추적기."""

    def __init__(self):
        self._lock = threading.Lock()
        # { provider: { model: _ModelUsage } }
        self._data: Dict[str, Dict[str, _ModelUsage]] = {}

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        with self._lock:
            if provider not in self._data:
                self._data[provider] = {}
            if model not in self._data[provider]:
                self._data[provider][model] = _ModelUsage()
            u = self._data[provider][model]
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.calls += 1

    def _snapshot(self):
        with self._lock:
            return {p: {m: _ModelUsage(u.input_tokens, u.output_tokens, u.calls)
                        for m, u in models.items()}
                    for p, models in self._data.items()}

    def print_current(self, stage: str = ""):
        snap = self._snapshot()
        if not snap:
            return
        label = f"토큰 사용량 ({stage})" if stage else "현재까지 토큰 사용량"
        total_cost = 0.0
        lines = []
        for provider, models in sorted(snap.items()):
            for model, u in sorted(models.items()):
                cost = _cost_usd(model, u.input_tokens, u.output_tokens)
                total_cost += cost
                lines.append(
                    f"    {provider:<10} {model:<35} "
                    f"in={u.input_tokens:>8,}  out={u.output_tokens:>7,}  "
                    f"calls={u.calls:>4}  ~${cost:.4f}"
                )
        print(f"\n  ┌─ {label} {'─'*(50-len(label))}")
        for l in lines:
            print(f"  │{l}")
        print(f"  └─ 합계 예상 비용: ~${total_cost:.4f} USD\n")

    def format_summary(self) -> str:
        snap = self._snapshot()
        if not snap:
            return ""
        total_in = total_out = total_cost = 0
        lines = []
        for provider, models in sorted(snap.items()):
            p_in = p_out = p_cost = 0
            for model, u in sorted(models.items()):
                cost = _cost_usd(model, u.input_tokens, u.output_tokens)
                p_in  += u.input_tokens
                p_out += u.output_tokens
                p_cost += cost
                lines.append(
                    f"    {provider:<10} {model:<35} "
                    f"in={u.input_tokens:>8,}  out={u.output_tokens:>7,}  "
                    f"calls={u.calls:>4}  ~${cost:.4f}"
                )
            total_in   += p_in
            total_out  += p_out
            total_cost += p_cost

        width = 70
        parts = [
            f"\n  {'═'*width}",
            f"  {'토큰 사용량 최종 요약':^{width}}",
            f"  {'═'*width}",
            *lines,
            f"  {'─'*width}",
            f"    {'합계':<46}in={total_in:>8,}  out={total_out:>7,}",
            f"    예상 비용: ~${total_cost:.4f} USD  (참고용 추정치)",
            f"  {'═'*width}",
        ]
        return "\n".join(parts) + "\n"

    def print_summary(self):
        text = self.format_summary()
        if text:
            print(text)


# 전역 싱글턴
tracker = TokenTracker()
