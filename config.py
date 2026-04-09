import os
from pathlib import Path

# .env 파일 로드 (python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            # 인라인 주석 제거: 따옴표 없는 값에서 # 이후 삭제
            _v = _v.strip()
            if not (_v.startswith('"') or _v.startswith("'")):
                _v = _v.split("#")[0].strip()
            os.environ.setdefault(_k.strip(), _v)

# === 파이프라인 설정 ===
MAX_SEGMENT_DURATION = 30          # 이 시간(초) 이상인 클립은 자동 분할
MIN_SEGMENT_DURATION = 2           # 이 시간(초) 미만은 자동 버림

# === 편집 스타일 ===
# balanced    : 음성과 풍경 균형 (기본)
# voice       : 음성/대화 우선, 무음 풍경 최소화
# scene-short : 풍경 포함하되 10초 이내로 트림
# scene-long  : 풍경 우선, 최대 30초 허용
# highlight   : 최고 점수 클립만 엄선 (소셜 미디어 하이라이트)
# vlog        : 말하는 장면 중심, 일상 브이로그 스타일
# .env: EDIT_STYLE=voice
EDIT_STYLE = os.environ.get("EDIT_STYLE", "balanced").strip().lower()

# 스타일별 파라미터:
#   landscape_sec  - 이 시간 이상의 무음 풍경을 AI가 컷/트림 평가
#   max_keep_sec   - 무음 풍경 클립의 최대 유지 길이 (rule-based trim)
#   discard_silent - True면 무음 클립을 rule-based에서 바로 버림
_STYLE_TABLE = {
    #                    landscape  max_keep  discard_silent
    "balanced":         (10,        10,       False),
    "voice":            ( 5,         5,       True ),
    "scene-short":      (10,        10,       False),
    "scene-long":       (30,        30,       False),
    "highlight":        ( 5,         8,       False),
    "vlog":             ( 5,         5,       False),  # voice와 달리 5초 이내 전환 컷 허용
}
_st = _STYLE_TABLE.get(EDIT_STYLE, _STYLE_TABLE["balanced"])
PURE_LANDSCAPE_THRESHOLD = _st[0]   # AI 평가 트리거 임계값
STYLE_MAX_LANDSCAPE      = _st[1]   # rule-based 트림 상한
STYLE_DISCARD_SILENT     = _st[2]   # 무음 클립 즉시 버림 여부

# === 출력 설정 ===
# None = 원본 영상 최고 해상도 기준 자동 선택 (4K / 1440p / FHD / 720p 중 가장 가까운 상위 계단)
# .env / 환경변수: OUTPUT_RESOLUTION=auto | 4k | 1440p | fhd | 720p | 3840x2160
_res_env = os.environ.get("OUTPUT_RESOLUTION", "").strip().lower()
_RES_PRESETS = {
    "4k":    (3840, 2160),
    "1440p": (2560, 1440),
    "fhd":   (1920, 1080),
    "1080p": (1920, 1080),
    "720p":  (1280, 720),
}
if _res_env and _res_env not in ("", "auto"):
    if _res_env in _RES_PRESETS:
        OUTPUT_RESOLUTION = _RES_PRESETS[_res_env]
    elif "x" in _res_env:
        try:
            _rw, _rh = _res_env.split("x")
            OUTPUT_RESOLUTION = (int(_rw), int(_rh))
        except ValueError:
            OUTPUT_RESOLUTION = None
    else:
        OUTPUT_RESOLUTION = None
else:
    OUTPUT_RESOLUTION = None  # auto

OUTPUT_FPS = 30  # 미사용: 출력 FPS는 클립 원본 FPS를 기준으로 자동 결정 (모두 60fps+ 이면 60, 아니면 30)
# CRF: H.264 기준 18 = 시각적 무손실, 9 = 거의 완전 무손실(용량 매우 큼)
#      H.265 기준 22 ≈ H.264 CRF 18과 동등 화질, 용량은 H.264의 약 절반
# .env: CRF=18
CRF = int(os.environ.get("CRF", "24"))
# VIDEO_CODEC: h264 (호환성 최대) | h265 (동일 화질에서 용량 약 50% 절감, 인코딩 느림)
# .env: VIDEO_CODEC=h265
_codec_env = os.environ.get("VIDEO_CODEC", "h265").strip().lower()
VIDEO_CODEC = "libx265" if _codec_env in ("h265", "hevc", "libx265") else "libx264"
FFMPEG_PRESET = os.environ.get("FFMPEG_PRESET", "medium")

# NVENC: GPU 하드웨어 인코딩 (NVIDIA GPU 필요)
# .env: USE_NVENC=true  → h264→h264_nvenc, h265→hevc_nvenc 로 자동 전환
# NVENC_PRESET: p1(최속)~p7(최고품질), 기본 p4 (medium 상당)
USE_NVENC = os.environ.get("USE_NVENC", "auto").strip().lower()
NVENC_PRESET = os.environ.get("NVENC_PRESET", "p4")
# GeForce 소비자용 GPU는 NVENC 동시 세션을 3개로 제한 (하드웨어 제약)
# Quadro / RTX A시리즈 / Data Center GPU는 무제한 → 높게 설정 가능
# .env: NVENC_MAX_SESSIONS=3
NVENC_MAX_SESSIONS = int(os.environ.get("NVENC_MAX_SESSIONS", "3"))

# === 자막 설정 ===
SUBTITLE_FONT = "Arial"            # 한글 지원 폰트: NanumGothic, Malgun Gothic 등
SUBTITLE_FONT_SIZE = 52
SUBTITLE_MARGIN_V = 40             # 하단 여백(px)

# === 장소 표시 설정 ===
LOCATION_DISPLAY_DURATION = 5.25   # 장소 텍스트 표시 시간(초)
LOCATION_FADE_DURATION = 0.4       # 페이드 인/아웃 시간(초)
LOCATION_FONT_SIZE = 34
LOCATION_MARGIN = 20               # 우하단 여백(px)

# === Whisper 설정 ===
# 표준 모델:    tiny | base | small | medium | large-v2 | large-v3
# Distil 모델: distil-large-v3 | distil-large-v2 | distil-medium.en
#              (영어 전용, 표준 대비 약 2배 빠름, 정확도 소폭 낮음)
# .env: WHISPER_MODEL=distil-large-v3
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# === 자막 설정 (언어/방식) ===
SUBTITLE_LANG = os.environ.get("SUBTITLE_LANG", "auto")    # auto|ko|en|ja|zh|off
SUBTITLE_MODE = os.environ.get("SUBTITLE_MODE", "overlay") # overlay|srt

# === Claude AI 설정 ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_CONCURRENT = int(os.environ.get("EVAL_WORKERS", "5"))  # 동시 AI 평가 수

# === STT 정제 설정 ===
# Whisper 결과를 LLM으로 한 번 더 정제 (외부 소음/한국어 오인식 보정)
STT_REFINE = os.environ.get("STT_REFINE", "true").lower() not in ("0", "false", "off")
STT_REFINE_MODEL = os.environ.get("STT_REFINE_MODEL", "claude-haiku-4-5-20251001")

# === OpenAI / Gemini 설정 (rate limit 시 라운드로빈 폴백) ===
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL",   "gpt-4o-mini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL",   "gemini-2.5-flash")

# === 방향 분리 설정 ===
# 가로/세로 영상을 별도 파일로 출력 (세로: travel_YYYY-MM-DD_vertical.mp4)
# .env: SPLIT_ORIENTATION=true
SPLIT_ORIENTATION = os.environ.get("SPLIT_ORIENTATION", "false").lower() in ("1", "true", "on")

# === 최소 분량 설정 ===
# 하루 영상이 이 시간(초) 미만이면, 버린 클립 중 고득점 순으로 채움
# 그래도 부족하면 해당 날의 모든 클립을 포함
# 0 = 비활성 (기본)
# .env: MIN_DAY_DURATION=300  (5분)
MIN_DAY_DURATION = int(os.environ.get("MIN_DAY_DURATION", "0"))

# === 아카이브 설정 ===
# 하루치 렌더링 완료 후 mp4(+srt)를 이 폴더로 이동 (None = 이동 안 함)
# .env: ARCHIVE_DIR=/mnt/nas/travel_archive
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "") or None

# === 병렬 처리 설정 ===
METADATA_WORKERS = 32          # NAS 환경: I/O 대기가 대부분 → 많을수록 유리
SEGMENT_WORKERS = None             # None = max(4, cpu_count) — 전체 세그먼트 단일 풀 병렬
_tw = os.environ.get("TRANSCRIBE_WORKERS", "0")
TRANSCRIBE_WORKERS = int(_tw) if _tw.isdigit() else 0
# WHISPER_VRAM_OVERHEAD: 추론 시 idle VRAM 대비 추가 사용 비율 (헤드룸 예약)
# 낮출수록 더 많은 인스턴스 허용 (OOM 위험 증가), 높일수록 보수적
# .env: WHISPER_VRAM_OVERHEAD=0.40  (기본, large-v3 기준 실측치)
# OOM 발생 시 0.5~0.7로 높이고, 여유 있으면 0.2~0.3으로 낮춰볼 것
# 0 = VRAM이 허용하는 한도까지 자동으로 최대한 로드 (OOM 직전까지)
RENDER_WORKERS = int(os.environ.get("RENDER_WORKERS", "0")) or None
# None → cpu_count // 2 로 자동 결정 (renderer.py 참고)
