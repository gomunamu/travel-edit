# Travel Video Editor

여행 동영상을 AI가 VLOG 형식으로 자동으로 편집해주는 도구입니다.  
입력 폴더의 동영상을 분석해 날짜별로 컷 편집, (음성인식)자막 생성, 장소 표시까지 자동으로 처리합니다.

## 주요 기능

- **자동 컷 편집** — 짧거나 불필요한 클립 자동 제거, 긴 클립 자동 분할
- **AI 클립 평가** — Claude AI로 0~100점 채점 (시각·음성·장면·흐름 4개 세부 항목)
- **자막 자동 생성** — Whisper로 음성 인식, 1줄씩 표시, 영상 번인 또는 SRT 파일 출력
- **LLM 자막 정제** — Whisper 결과를 LLM으로 한 번 더 보정 (외래어·소음 오인식 수정)
- **다국어 자막** — 한국어/영어 자동감지, 일본어·중국어 지정 가능
- **장소 표시** — GPS 메타데이터 기반 촬영 장소 자동 표시 우하단 (도시, 국가 형식)
- **자동 해상도** — 원본 클립 최고 해상도 기준 자동 선택 (4K / 1440p / FHD / 720p)
- **날짜별 출력** — 촬영 날짜 기준으로 영상 자동 분류 및 합본 생성
- **병렬 렌더링** — 멀티코어 병렬 렌더링으로 빠른 처리
- **AI 폴백** — Claude → OpenAI → Gemini 순서로 rate limit 시 자동 전환, 소스 파일 단위로 API 상태 리셋
- **토큰 사용량 추적** — Anthropic / OpenAI / Gemini 각각 토큰·비용 집계

## 요구사항

- Python 3.8+
- ffmpeg / ffprobe
- CUDA GPU (권장, CPU도 가능)

```bash
# ffmpeg 설치
sudo apt install ffmpeg        # Ubuntu
brew install ffmpeg            # macOS

# Python 패키지 설치
pip install -r requirements.txt
```

### Python 가상환경 (권장)

CUDA / PyTorch 관련 패키지 충돌 방지를 위해 venv 사용을 권장합니다.

```bash
python3 -m venv ~/venvs/torch
source ~/venvs/torch/bin/activate
pip install -r requirements.txt
```

이후 실행 시에도 항상 같은 venv를 활성화한 상태에서 실행합니다.

### Whisper 의존성 (faster-whisper)

faster-whisper는 **PyTorch 기반** 패키지입니다 (TensorFlow 아님).  
내부적으로 CTranslate2 엔진을 사용하며, CUDA + cuDNN이 필요합니다.

| CUDA 버전 | cuDNN 버전 | faster-whisper |
|-----------|-----------|----------------|
| CUDA 12.x | cuDNN 9.x | ≥ 1.0.0 (권장) |
| CUDA 11.x | cuDNN 8.x | 0.x 계열 |

```bash
# CUDA 12 + cuDNN 9 기준 설치 예시 (Ubuntu)
pip install faster-whisper>=1.0.0
```

### Gemini API 패키지

코드는 신버전 `google-genai` SDK를 사용합니다 (`from google import genai`).  
구버전 `google-generativeai`와 혼용하면 ImportError가 발생합니다.

```bash
pip uninstall google-generativeai -y
pip install google-genai>=1.0.0
```

## 사용법

```bash
python main.py <입력폴더> <출력폴더> [옵션]
```

### 예시

```bash
# 기본 실행 (한/영 자동감지, 자동 해상도, 영상 번인)
python main.py ~/여행사진 ~/편집결과

# AI 없이 규칙 기반으로만 처리
python main.py /media/usb/DCIM ./output --no-ai

# 일본어 자막, 영상 번인
python main.py ./videos ./output --subtitle-lang ja

# 한국어 자막, 별도 SRT 파일로 저장
python main.py ./videos ./output --subtitle-lang ko --subtitle-mode srt

# 자막 없이 빠르게 처리
python main.py ./videos ./output --subtitle-lang off

# 출력 해상도 고정
python main.py ./videos ./output --resolution fhd
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--no-ai` | Claude AI 평가 없이 규칙 기반으로만 처리 |
| `--whisper-model` | Whisper 모델 크기 (`tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`) |
| `--subtitle-lang` | 자막 언어 (`auto` / `ko` / `en` / `ja` / `zh` / `off`, 기본: `auto`) |
| `--subtitle-mode` | 자막 방식 (`overlay`=영상 번인 / `srt`=별도 파일, 기본: `overlay`) |
| `--resolution` | 출력 해상도 (`auto` / `4k` / `1440p` / `fhd` / `720p`, 기본: `auto`) |
| `--max-segment N` | 클립 최대 길이(초), 이보다 길면 자동 분할 (기본: 30) |
| `--workers N` | 렌더링 병렬 워커 수 |
| `--skip-transcribe` | 음성 인식 건너뜀 (`--subtitle-lang off`과 동일) |

## 환경변수 / .env

프로젝트 루트에 `.env` 파일을 만들어 설정할 수 있습니다.

```env
# AI API 키
ANTHROPIC_API_KEY=sk-ant-...   # Claude AI 사용 시 필요
OPENAI_API_KEY=sk-...           # OpenAI 폴백 사용 시 (선택)
GEMINI_API_KEY=...              # Gemini 폴백 사용 시 (선택)

# Whisper 설정
WHISPER_MODEL=large-v3         # tiny | base | small | medium | large-v2 | large-v3
WHISPER_DEVICE=cuda            # cuda | cpu
WHISPER_COMPUTE_TYPE=float16   # float16 | int8

# 자막 설정
SUBTITLE_LANG=auto             # auto | ko | en | ja | zh | off
SUBTITLE_MODE=overlay          # overlay | srt

# STT 정제 (Whisper 결과를 LLM으로 보정)
STT_REFINE=true                # true | false
STT_REFINE_MODEL=claude-haiku-4-5-20251001

# 출력 설정
OUTPUT_RESOLUTION=auto         # auto | 4k | 1440p | fhd | 720p | 1920x1080
CRF=18                         # 화질/용량 균형 (낮을수록 화질↑ 용량↑, 18=시각적 무손실)
VIDEO_CODEC=h264               # h264 (호환성 최대) | h265 (동일 화질에서 용량 약 50% 절감)
FFMPEG_PRESET=medium           # ultrafast | fast | medium | slow (느릴수록 압축률↑)
RENDER_WORKERS=0               # 0 = 해상도·GPU 여부에 따라 자동 결정

# GPU 하드웨어 인코딩 (NVENC)
USE_NVENC=auto                 # auto(감지) | true(강제) | false(비활성)
NVENC_PRESET=p4                # p1(최속)~p7(최고품질), 기본 p4 (medium 상당)
```

- `ANTHROPIC_API_KEY` 없으면 규칙 기반 클립 평가로 자동 대체
- OpenAI / Gemini API 키 등록 시 Claude rate limit 발생 시 자동 폴백
- `.env`는 `.gitignore`에 포함되어 있어 git에 업로드되지 않음

## AI 클립 평가

Claude AI가 각 클립을 0~100점으로 채점합니다.

| 항목 | 설명 |
|------|------|
| **시각 (visual)** | 화질, 흔들림, 노출 등 촬영 품질 |
| **음성 (speech)** | 음성 명료도, 배경 소음 |
| **장면 (scene)** | 풍경·내용의 흥미도 |
| **흐름 (flow)** | 편집 흐름상 필요 여부 |

종합 점수가 낮은 클립은 자동으로 제거되며, 각 클립에 대한 2~3문장 평가 이유도 함께 출력됩니다.

## 출력 구조

```
output/
├── travel_2024-07-15.mp4     # 날짜별 편집 완료 영상
├── travel_2024-07-15.srt     # SRT 모드일 때 자막 파일
├── .cache/                   # 중간 작업 파일 (재실행 시 재사용)
└── rendered/                 # 날짜별 렌더링된 클립들
```

## 설정

`config.py` 또는 `.env`에서 세부 설정을 변경할 수 있습니다.

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `OUTPUT_RESOLUTION` | auto | 출력 해상도 (원본 최고 해상도 자동 선택) |
| `OUTPUT_FPS` | 30 | 출력 프레임레이트 |
| `CRF` | 18 | 화질/용량 균형 (H.264 기준: 18=시각적 무손실, 낮을수록 용량↑) |
| `VIDEO_CODEC` | h264 | 인코딩 코덱 (`h264` / `h265`) — H.265는 동일 화질에서 용량 약 50% 절감 |
| `FFMPEG_PRESET` | medium | 인코딩 속도/압축률 트레이드오프 (`ultrafast` ~ `veryslow`) |
| `USE_NVENC` | auto | NVIDIA GPU 하드웨어 인코딩 (`auto` / `true` / `false`) |
| `NVENC_PRESET` | p4 | NVENC 품질/속도 (`p1`=최속 ~ `p7`=최고품질) |
| `WHISPER_MODEL` | large-v3 | 음성 인식 모델 |
| `SUBTITLE_LANG` | auto | 자막 언어 |
| `SUBTITLE_MODE` | overlay | 자막 방식 |
| `STT_REFINE` | true | LLM 자막 정제 사용 여부 |
| `MAX_SEGMENT_DURATION` | 30초 | 클립 최대 길이 |
| `MIN_SEGMENT_DURATION` | 2초 | 클립 최소 길이 |
| `TRANSCRIBE_WORKERS` | 0(자동) | 음성 인식 병렬 수 (VRAM에 따라 자동 제한) |
| `RENDER_WORKERS` | auto | 렌더링 병렬 수 (해상도·GPU 여부에 따라 자동 결정) |
| `METADATA_WORKERS` | 32 | 메타데이터 추출 병렬 수 |

### NVENC (GPU 하드웨어 인코딩)

NVIDIA GPU가 있으면 자동으로 감지해 하드웨어 인코딩을 사용합니다.  
CPU 인코딩(libx264) 대비 4K 기준 **10~30배 빠른** 인코딩 속도를 제공합니다.

| 인코더 | 4K 속도 | CRF 호환 | 비고 |
|--------|---------|----------|------|
| libx264 (CPU) | 5~15 fps | CRF 사용 | 느리지만 호환성 최대 |
| h264_nvenc (GPU) | 150~200 fps | `-cq` 사용 | CUDA 설치 필요 |
| hevc_nvenc (GPU) | 120~180 fps | `-cq` 사용 | `VIDEO_CODEC=h265`일 때 |

NVENC가 활성화되면 `RENDER_WORKERS`는 `cpu_count // 4`로 자동 조정됩니다.  
(CPU 인코딩 시 4K는 `cpu_count // 8`로 제한하여 메모리 포화 방지)

> **참고**: NVENC 최소 해상도는 145×145px입니다. 그 이하 해상도는 CPU 인코딩으로 자동 전환됩니다.

### 입력 폴더 구조 및 날짜 인식

입력 폴더를 재귀 탐색하므로 하위 디렉토리에 있는 영상도 모두 수집합니다.  
날짜는 다음 순서로 인식합니다.

1. **영상 메타데이터** — `creation_time` EXIF 태그 (가장 정확)
2. **디렉토리명** — `2024-07-15/`, `20240715/`, `2024_07_15/` 형식 인식
3. **파일명** — `VID_20240715_...`, `DJI_20240715...` 등 날짜 포함 파일명
4. **파일 수정시간(mtime)** — 위 세 가지 모두 없을 때 최후 fallback

메타데이터 없이 fallback으로 날짜를 추정한 파일은 실행 시 목록과 함께 안내됩니다.
