import os
from pathlib import Path

# .env 파일 로드 (python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# === 파이프라인 설정 ===
MAX_SEGMENT_DURATION = 30          # 이 시간(초) 이상인 클립은 자동 분할
MIN_SEGMENT_DURATION = 2           # 이 시간(초) 미만은 자동 버림
PURE_LANDSCAPE_THRESHOLD = 10      # 음성 없이 이 시간(초) 이상이면 AI가 컷 평가

# === 출력 설정 ===
OUTPUT_RESOLUTION = (720, 480)   # 출력 해상도
OUTPUT_FPS = 30
CRF = 9                           # 화질 (낮을수록 좋음, 18 = 거의 무손실)
FFMPEG_PRESET = "fast"

# === 자막 설정 ===
SUBTITLE_FONT = "Arial"            # 한글 지원 폰트: NanumGothic, Malgun Gothic 등
SUBTITLE_FONT_SIZE = 52
SUBTITLE_MARGIN_V = 40             # 하단 여백(px)

# === 장소 표시 설정 ===
LOCATION_DISPLAY_DURATION = 3.5    # 장소 텍스트 표시 시간(초)
LOCATION_FADE_DURATION = 0.4       # 페이드 인/아웃 시간(초)
LOCATION_FONT_SIZE = 34
LOCATION_MARGIN = 20               # 우하단 여백(px)

# === Whisper 설정 ===
# .env 또는 환경변수로 덮어쓸 수 있음. 미설정 시 "large-v3" 사용.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

# === 자막 설정 (언어/방식) ===
SUBTITLE_LANG = os.environ.get("SUBTITLE_LANG", "auto")    # auto|ko|en|ja|zh|off
SUBTITLE_MODE = os.environ.get("SUBTITLE_MODE", "overlay") # overlay|srt

# === Claude AI 설정 ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_CONCURRENT = 5          # 동시 API 호출 수

# === STT 정제 설정 ===
# Whisper 결과를 LLM으로 한 번 더 정제 (외부 소음/한국어 오인식 보정)
STT_REFINE = os.environ.get("STT_REFINE", "true").lower() not in ("0", "false", "off")
STT_REFINE_MODEL = os.environ.get("STT_REFINE_MODEL", "claude-haiku-4-5-20251001")

# === OpenAI / Gemini 설정 (rate limit 시 라운드로빈 폴백) ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL",   "gpt-4o-mini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL",   "gemini-1.5-flash")

# === 병렬 처리 설정 ===
METADATA_WORKERS = 32          # NAS 환경: I/O 대기가 대부분 → 많을수록 유리
SEGMENT_WORKERS = None             # None = max(4, cpu_count) — 전체 세그먼트 단일 풀 병렬
TRANSCRIBE_WORKERS = 8             # 실제 로드 가능한 수는 VRAM에 따라 자동 제한됨
                                   # OOM 발생 시 로드된 인스턴스 수로 자동 조정
RENDER_WORKERS = None              # None = cpu_count // 2
RENDER_BATCH_SIZE = int(os.environ.get("RENDER_BATCH_SIZE", "8"))  # 배치당 클립 수
