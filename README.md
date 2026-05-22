# Travel Video Editor

[한국어](#한국어) | [English](#english)

---

<a name="한국어"></a>

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

## 하드웨어 권장 사양

이 도구는 **GPU 유무에 따라 처리 속도 차이가 매우 큽니다.**  
GPU 없이도 동작하지만, 4K 영상 기준 인코딩 속도가 10~30배 차이납니다.

### GPU (NVIDIA CUDA) — 권장

NVIDIA GPU는 두 가지 작업을 가속합니다.

| 작업 | GPU 가속 방식 | 효과 |
|------|-------------|------|
| **음성 인식 (Whisper)** | CUDA 추론 | CPU 대비 5~10배 빠름 |
| **영상 인코딩 (FFmpeg)** | NVENC 하드웨어 인코더 | 4K 기준 CPU 대비 **10~30배** 빠름 |

**4K 영상 인코딩 속도 비교:**

| 모드 | 처리 속도 | 28코어 기준 병렬 워커 |
|------|----------|-------------------|
| GPU (NVENC+NVDEC) | 150~200 fps | 14개 |
| CPU (libx265) | 5~15 fps | 3개 |

- NVIDIA RTX / GTX 계열이면 모두 NVENC 지원
- `USE_NVENC=auto` (기본값)로 설정하면 GPU가 있을 때 자동으로 활성화됨
- GeForce 소비자용 GPU는 NVENC 동시 세션이 **3개로 하드웨어 제한** (Quadro / RTX A시리즈는 무제한)

### CPU 전용 — 가능하지만 느림

- `./install.sh --cpu-only` 로 설치
- Whisper는 CPU 추론으로 동작 (`WHISPER_DEVICE=cpu`)
- 인코딩은 libx264/libx265 소프트웨어 인코더 사용
- 짧은 여행 영상(FHD 이하)은 CPU로도 현실적인 시간 내 처리 가능

### 최소/권장 사양 요약

| | 최소 | 권장 |
|-|------|------|
| **GPU** | 없음 (CPU 전용) | NVIDIA RTX 3060 이상 |
| **VRAM** | — | 8GB 이상 (large-v3 Whisper 모델 기준) |
| **RAM** | 8GB | 16GB 이상 |
| **CPU** | 4코어 | 8코어 이상 |
| **OS** | macOS / Ubuntu 22.04 | Ubuntu 22.04 (NVENC 지원) |

> macOS는 NVENC를 지원하지 않아 CPU 인코딩만 가능합니다. Whisper는 MPS(Apple Silicon) 또는 CPU로 동작합니다.

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

### 설치 스크립트

```bash
./install.sh
```

`install.sh`가 아래 모든 항목을 자동으로 처리합니다.  
옵션:

```bash
./install.sh                     # 기본 (PyTorch CUDA 12.8 wheel)
./install.sh --torch-cuda=cu124  # CUDA 12.4 wheel 사용
./install.sh --cpu-only          # GPU 없는 환경
./install.sh --skip-torch        # PyTorch 이미 설치된 경우 건너뜀
```

### CUDA 아키텍처

이 앱은 CUDA를 **두 레이어**로 사용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│  시스템 레이어 (apt)                                             │
│    CUDA Toolkit 12.x  ──→  ffmpeg NVENC/NVDEC (GPU 인코딩·디코딩)│
│    NVIDIA Driver       ──→  nvidia-smi, GPU 관리                │
├─────────────────────────────────────────────────────────────────┤
│  venv 레이어 (pip)                                               │
│    PyTorch +cu12x      ──→  nvidia-cudnn-cu12 내장 (cuDNN 자동) │
│    ctranslate2         ──→  자체 CUDA 라이브러리 내장            │
│    faster-whisper      ──→  ctranslate2 사용 (음성 인식)         │
└─────────────────────────────────────────────────────────────────┘
```

**핵심**: pip wheel (`torch+cu128` 등)은 cuDNN을 내장하므로 **시스템에 cuDNN을 별도 설치할 필요가 없습니다.**  
시스템 CUDA Toolkit은 ffmpeg NVENC/NVDEC 전용입니다.

#### 시스템 CUDA Toolkit 설치 (ffmpeg NVENC/NVDEC용)

```bash
# 1) NVIDIA apt 저장소 등록
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 2) CUDA Toolkit 설치 (드라이버 버전에 맞게 선택)
sudo apt-get install -y cuda-toolkit-12-4
```

드라이버별 지원 CUDA 최대 버전:

| NVIDIA 드라이버 | 지원 CUDA 최대 |
|-----------------|---------------|
| 535.x | 12.2 |
| 550.x | 12.4 |
| 560.x | 12.6 |
| 570.x | 12.8 |

Ubuntu 22.04 기본 ffmpeg 패키지는 NVENC/NVDEC를 포함하므로 별도 ffmpeg 빌드가 불필요합니다.

#### venv PyTorch 설치 (Python CUDA용)

```bash
# CUDA 12.8 (드라이버 570.x 이상)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4 (드라이버 550.x 이상)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU 전용
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

faster-whisper는 내부적으로 **CTranslate2** 엔진을 사용합니다 (PyTorch 직접 사용 안 함).  
CTranslate2도 자체 CUDA 라이브러리를 내장하므로 `pip install faster-whisper` 만으로 GPU 가속이 동작합니다.

#### LD_LIBRARY_PATH 정리 (권장)

여러 버전의 CUDA를 설치한 경우 `~/.bashrc`에서 경로를 통일하세요.

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

구버전 CUDA 경로(`/usr/local/cuda-11.3/lib64` 등)가 남아 있으면 충돌이 발생할 수 있습니다.

### Python 가상환경

CUDA / PyTorch 관련 패키지 충돌 방지를 위해 venv 사용을 필수로 권장합니다.

```bash
python3 -m venv ~/venvs/torch
source ~/venvs/torch/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

이후 실행 시에도 항상 같은 venv를 활성화한 상태에서 실행합니다.

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
| `--style` | 편집 스타일 (아래 참고, 기본: `balanced`) |
| `--min-day-duration N` | 하루 영상 최소 길이(분). 미달 시 버린 클립 중 고득점 순으로 채움 (기본: 비활성) |
| `--split-orientation` | 가로/세로 영상을 별도 파일로 출력 (`travel_YYYY-MM-DD_vertical.mp4`) |
| `--max-segment N` | 클립 최대 길이(초), 이보다 길면 자동 분할 (기본: 30) |
| `--workers N` | 렌더링 병렬 워커 수 |
| `--skip-transcribe` | 음성 인식 건너뜀 (`--subtitle-lang off`과 동일) |
| `--no-stt-refine` | STT 결과 LLM 정제 비활성화 |
| `--archive-dir DIR` | 완료된 mp4를 지정 폴더로 이동 (NVMe 용량 절약) |

### 편집 스타일 (`--style`)

하루 영상 전체의 편집 기준을 한 번에 설정합니다.  
스타일에 따라 AI에게 주어지는 **페르소나와 평가 기준이 완전히 달라집니다.**

| 스타일 | AI 페르소나 | 무음 풍경 처리 | 최대 유지 |
|--------|------------|--------------|----------|
| `balanced` | 여행 브이로그 PD | 길면 트림, 짧으면 유지 | 10초 |
| `voice` | 여행 브이로그 PD | **전부 제거** | — |
| `vlog` | 여행 브이로그 PD | 5초 이내 전환 컷은 유지 | 5초 |
| `scene-short` | 여행 영상 편집자 | 핵심 구간만 트림해 유지 | 10초 |
| `scene-long` | **여행 다큐멘터리 감독** | **기본적으로 유지** (화질 불량만 제거) | 30초 |
| `highlight` | 여행 브이로그 PD | 고득점 클립만 엄선 | 8초 |

```bash
python main.py ./videos ./output --style scene-long   # 풍경·자연 위주
python main.py ./videos ./output --style vlog         # 말하는 장면 위주
python main.py ./videos ./output --style highlight    # 하이라이트 릴
```

**`scene-long` / `scene-short` 주의사항**

`scene-long`과 `scene-short`는 AI 페르소나 자체가 다릅니다.

- 다른 스타일: *"시청자가 지루하지 않도록 재미있는 장면만 선별하라"* (브이로그 PD)
- `scene-long`: *"무음 풍경은 결함이 아니라 핵심 콘텐츠. 버림은 심한 흔들림·초점 불량에만 한정"* (다큐 감독)
- `scene-short`: *"음성 없음은 감점 사유가 아님. 핵심 구간만 남겨 트림"* (여행 편집자)

같은 무음 풍경 클립이라도 `scene-long`에서는 기본 keep/trim, 다른 스타일에서는 discard 후보가 됩니다.

> **`voice` vs `vlog`**: `voice`는 무음 클립을 전부 제거합니다. `vlog`는 5초 이내 무음 전환 컷은 편집 흐름상 살립니다.

### 최소 분량 보장 (`--min-day-duration`)

AI 평가 결과 특정 날에 살아남은 클립이 없거나 분량이 부족할 때 자동으로 채웁니다.

1. **1단계**: 버린 클립 중 점수 높은 순으로 추가해 목표 분량을 채움
2. **2단계**: 그래도 부족하면 해당 날의 모든 클립을 포함

```bash
# 하루 최소 3분 보장
python main.py ./videos ./output --min-day-duration 3

# .env로 설정 (초 단위)
MIN_DAY_DURATION=300   # 5분
```

## 환경변수 / .env

프로젝트 루트에 `.env` 파일을 만들어 설정할 수 있습니다.

```env
# AI API 키
ANTHROPIC_API_KEY=sk-ant-...   # Claude AI 사용 시 필요
OPENAI_API_KEY=sk-...          # OpenAI 폴백 사용 시 (선택)
GEMINI_API_KEY=...             # Gemini 폴백 사용 시 (선택)

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
CRF=24                         # 화질/용량 균형 (낮을수록 화질↑ 용량↑, H.265 기준 24=여행 영상 실질 무손실)
VIDEO_CODEC=h265               # h264 (호환성 최대) | h265 (동일 화질에서 용량 약 50% 절감)
FFMPEG_PRESET=medium           # ultrafast | fast | medium | slow (느릴수록 압축률↑)
RENDER_WORKERS=0               # 0 = 해상도·GPU 여부에 따라 자동 결정

# GPU 하드웨어 인코딩 (NVENC)
USE_NVENC=auto                 # auto(감지) | true(강제) | false(비활성)
NVENC_PRESET=p4                # p1(최속)~p7(최고품질), 기본 p4 (medium 상당)
NVENC_MAX_SESSIONS=3           # GeForce 최대 3 (하드웨어 제한) | Quadro/A시리즈는 높게 설정 가능

# 편집 스타일
EDIT_STYLE=balanced            # balanced | voice | vlog | scene-short | scene-long | highlight

# 최소 분량 (초 단위, 0=비활성)
MIN_DAY_DURATION=0             # 예: 300 = 하루 최소 5분

# 방향 분리
SPLIT_ORIENTATION=false        # true = 가로/세로 영상을 별도 파일로 출력
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
| `CRF` | 24 | 화질/용량 균형 (H.265 기준: 24=여행 영상 실질 무손실, 낮을수록 용량↑) |
| `VIDEO_CODEC` | h265 | 인코딩 코덱 (`h264` / `h265`) — H.265는 동일 화질에서 용량 약 50% 절감 |
| `FFMPEG_PRESET` | medium | 인코딩 속도/압축률 트레이드오프 (`ultrafast` ~ `veryslow`) |
| `USE_NVENC` | auto | NVIDIA GPU 하드웨어 인코딩 (`auto` / `true` / `false`) |
| `NVENC_PRESET` | p4 | NVENC 품질/속도 (`p1`=최속 ~ `p7`=최고품질) |
| `NVENC_MAX_SESSIONS` | 3 | NVENC 동시 세션 수 (GeForce 최대 3, Quadro/A시리즈는 높게 설정 가능) |
| `WHISPER_MODEL` | large-v3 | 음성 인식 모델 |
| `SUBTITLE_LANG` | auto | 자막 언어 |
| `SUBTITLE_MODE` | overlay | 자막 방식 |
| `STT_REFINE` | true | LLM 자막 정제 사용 여부 |
| `MAX_SEGMENT_DURATION` | 30초 | 클립 최대 길이 |
| `MIN_SEGMENT_DURATION` | 2초 | 클립 최소 길이 |
| `TRANSCRIBE_WORKERS` | 0(자동) | 음성 인식 병렬 수 (VRAM에 따라 자동 제한) |
| `RENDER_WORKERS` | auto | 렌더링 병렬 수 (GPU 모드: cpu//2, CPU 4K: cpu//8) |
| `METADATA_WORKERS` | 32 | 메타데이터 추출 병렬 수 |
| `EDIT_STYLE` | balanced | 편집 스타일 |
| `MIN_DAY_DURATION` | 0 | 하루 영상 최소 길이(초). 미달 시 버린 클립으로 채움 |
| `SPLIT_ORIENTATION` | false | 가로/세로 영상을 별도 파일로 분리 출력 |

### NVENC + NVDEC (GPU 하드웨어 인코딩·디코딩)

NVIDIA GPU가 있으면 자동으로 감지해 하드웨어 인코딩·디코딩을 모두 사용합니다.  
CPU 인코딩(libx264) 대비 4K 기준 **10~30배 빠른** 처리 속도를 제공합니다.

| 구간 | CPU 모드 | GPU 모드 (NVENC+NVDEC) |
|------|---------|----------------------|
| 디코딩 | CPU (무거움) | NVDEC 전용 하드웨어 |
| 필터 (scale/pad/자막) | CPU | CPU (경량) |
| 인코딩 | CPU libx264/libx265 | NVENC 전용 하드웨어 |
| 4K 처리 속도 | 5~15 fps | 150~200 fps |
| 병렬 워커 수 (28코어 기준) | 3개 (4K) | 14개 |

GPU 모드에서는 디코딩·인코딩이 모두 GPU 전용 하드웨어(NVDEC/NVENC)에서 처리되어  
CPU는 scale/pad/자막 합성 등 가벼운 필터 작업만 담당합니다.

**워커 수 자동 결정 기준:**

| 모드 | 4K | 1440p | 1080p | 720p |
|------|----|-------|-------|------|
| GPU (NVENC+NVDEC) | cpu//2 | cpu//2 | cpu//2 | cpu//2 |
| CPU only | cpu//8 | cpu//6 | cpu//4 | cpu//2 |

#### GeForce GPU의 NVENC 동시 세션 제한

NVIDIA는 **GeForce 소비자용 GPU**(RTX/GTX)의 NVENC 동시 세션 수를 **3개로 하드웨어 제한**합니다.  
Quadro, RTX A시리즈, Data Center GPU(A100 등)는 이 제한이 없습니다.

이 제한을 초과하면 ffmpeg가 다음 에러를 출력하며 실패합니다.

```
[h264_nvenc] InitializeEncoder failed: out of memory (10)
[h264_nvenc] OpenEncodeSessionEx failed: incompatible client key (21)
```

이를 처리하기 위해 다음과 같이 동작합니다.

- **세마포어**: 최대 `NVENC_MAX_SESSIONS`개의 NVENC 세션만 동시에 열림 (기본값 3)
- **나머지 워커**: 세마포어 대기 중에도 NVDEC 디코딩·필터·오디오 처리는 계속 진행
- **자동 fallback**: 세션 한도 초과 에러 발생 시 해당 클립만 CPU(libx264)로 재시도

```
워커 14개 동시 실행 (28코어 기준)
  ├─ NVDEC 디코딩: 14개 병렬 (제한 없음)
  └─ NVENC 인코딩: 세마포어로 최대 3개만 동시 진입
       └─ 세션 한도 초과 시 → CPU 인코딩으로 자동 재시도
```

Quadro 등 제한 없는 GPU를 사용하는 경우 `.env`에서 설정합니다.

```env
NVENC_MAX_SESSIONS=16   # Quadro / RTX A시리즈
```

> **참고**: NVENC 최소 해상도는 145×145px입니다. 그 이하 해상도는 CPU 인코딩으로 자동 전환됩니다.  
> ffmpeg가 `h264_cuvid` / `hevc_cuvid` 디코더를 포함하지 않으면 NVDEC은 비활성화되고 CPU 디코딩으로 폴백합니다.

### 입력 폴더 구조 및 날짜 인식

입력 폴더를 재귀 탐색하므로 하위 디렉토리에 있는 영상도 모두 수집합니다.  
날짜는 다음 순서로 인식합니다.

1. **아이폰 QuickTime 태그** — `com.apple.quicktime.creationdate` (현지 시각 + 타임존 오프셋 포함, 가장 정확)
2. **일반 EXIF 태그** — `creation_time` (UTC 기준, 타임존 오프셋 없음)
3. **디렉토리명** — `2024-07-15/`, `20240715/`, `2024_07_15/` 형식 인식
4. **파일명** — `VID_20240715_...`, `DJI_20240715...` 등 날짜 포함 파일명
5. **파일 수정시간(mtime)** — 위 모두 없을 때 최후 fallback

메타데이터 없이 fallback으로 날짜를 추정한 파일은 실행 시 목록과 함께 안내됩니다.

> **아이폰 MOV 주의**: 일반 `creation_time` 태그는 UTC 기준이라, 한국(UTC+9)·뉴질랜드(UTC+13) 등 UTC와 차이가 큰 지역에서 자정 전후 촬영 시 날짜가 하루 어긋날 수 있습니다. 아이폰 고유 태그(`com.apple.quicktime.creationdate`)를 우선 사용하므로 이 문제를 방지합니다.

> **여러 사람 영상 혼합**: 파일명 순서는 정렬에 사용하지 않습니다. 각 파일의 실제 촬영 시각(`creation_time`) 기준으로 정렬하므로 여러 사람의 영상이 섞여도 시간순으로 올바르게 배치됩니다.

### 실행 시 주의사항

**반드시 venv를 활성화한 상태에서 실행**해야 모든 기능이 정상 동작합니다.  
특히 Gemini API는 `google-genai` 패키지가 필요하며, 시스템 Python에는 설치되지 않습니다.

```bash
# 권장 실행 방법
source ~/venvs/torch/bin/activate
python main.py <입력폴더> <출력폴더>

# 또는 venv Python 직접 지정
~/venvs/torch/bin/python main.py <입력폴더> <출력폴더>
```

시스템 Python(`/usr/bin/python3`)으로 실행하면 Gemini import 오류가 발생합니다.

```
[경고] Gemini 평가 실패: cannot import name 'genai' from 'google'
```

---

<a name="english"></a>

# Travel Video Editor — English Guide

An AI-powered tool that automatically edits travel videos into a vlog format.  
It analyzes videos in an input folder and handles cut editing by date, subtitle generation via speech recognition, and location overlays — all automatically.

## Features

- **Automatic Cut Editing** — Removes short/unnecessary clips; splits long clips automatically
- **AI Clip Scoring** — Claude AI scores each clip 0–100 across 4 sub-categories (visual, speech, scene, flow)
- **Subtitle Generation** — Whisper speech recognition with 1-line display; burned into video or exported as SRT
- **LLM Subtitle Refinement** — Whisper output post-corrected by an LLM to fix foreign words and noise misrecognitions
- **Multi-language Subtitles** — Auto-detects Korean/English; Japanese and Chinese can be specified manually
- **Location Overlay** — Displays shooting location (City, Country) in the bottom-right corner from GPS metadata
- **Auto Resolution** — Selects output resolution from the highest-resolution source clip (4K / 1440p / FHD / 720p)
- **Date-based Output** — Classifies and merges videos by shooting date
- **Parallel Rendering** — Multi-core parallel rendering for fast processing
- **AI Fallback** — Automatically switches Claude → OpenAI → Gemini on rate limits
- **Token Usage Tracking** — Aggregates token counts and costs for Anthropic / OpenAI / Gemini

## Hardware Requirements

This tool is **heavily GPU-dependent**. It works without a GPU, but encoding speed can be 10–30× slower for 4K footage.

### GPU (NVIDIA CUDA) — Recommended

An NVIDIA GPU accelerates two key workloads:

| Workload | GPU acceleration | Speedup |
|----------|-----------------|---------|
| **Speech recognition (Whisper)** | CUDA inference | 5–10× faster than CPU |
| **Video encoding (FFmpeg)** | NVENC hardware encoder | **10–30× faster** than CPU (4K) |

**4K encoding speed comparison:**

| Mode | Speed | Parallel workers (28-core CPU) |
|------|-------|-------------------------------|
| GPU (NVENC + NVDEC) | 150–200 fps | 14 |
| CPU (libx265) | 5–15 fps | 3 |

- Any NVIDIA RTX / GTX GPU supports NVENC
- `USE_NVENC=auto` (default) enables GPU encoding automatically when a GPU is detected
- Consumer GeForce GPUs are **hardware-limited to 3 concurrent NVENC sessions** (Quadro / RTX A-series: unlimited)

### CPU-only — Supported but Slow

- Install with `./install.sh --cpu-only`
- Whisper runs on CPU (`WHISPER_DEVICE=cpu`)
- Encoding uses software libx264/libx265
- Practical for short trips or FHD footage; 4K will be significantly slower

### Minimum / Recommended Specs

| | Minimum | Recommended |
|-|---------|-------------|
| **GPU** | None (CPU-only) | NVIDIA RTX 3060 or better |
| **VRAM** | — | 8 GB+ (for Whisper large-v3) |
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 cores | 8 cores+ |
| **OS** | macOS / Ubuntu 22.04 | Ubuntu 22.04 (full NVENC support) |

> macOS does not support NVENC. Whisper runs on MPS (Apple Silicon) or CPU.

## Installation

```bash
# Clone the repo and run the install script
./install.sh                     # Default (PyTorch CUDA 12.8)
./install.sh --torch-cuda=cu124  # CUDA 12.4 wheel
./install.sh --cpu-only          # No GPU
./install.sh --skip-torch        # PyTorch already installed
```

**Copy and fill in the environment file:**

```bash
cp core/.env.example core/.env
# Edit core/.env and set at least ANTHROPIC_API_KEY
```

## Usage

```bash
python main.py <input_folder> <output_folder> [options]
```

### Examples

```bash
# Basic run (auto language detection, auto resolution, burned-in subtitles)
python main.py ~/travel ~/output

# Rule-based only, no AI
python main.py /media/usb/DCIM ./output --no-ai

# Japanese subtitles, burned into video
python main.py ./videos ./output --subtitle-lang ja

# Korean subtitles as separate SRT file
python main.py ./videos ./output --subtitle-lang ko --subtitle-mode srt

# No subtitles (fastest)
python main.py ./videos ./output --subtitle-lang off

# Force FHD output
python main.py ./videos ./output --resolution fhd
```

### Options

| Option | Description |
|--------|-------------|
| `--no-ai` | Rule-based clip evaluation only, skip Claude AI |
| `--whisper-model` | Whisper model (`tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`) |
| `--subtitle-lang` | Subtitle language (`auto` / `ko` / `en` / `ja` / `zh` / `off`, default: `auto`) |
| `--subtitle-mode` | Subtitle mode (`overlay`=burned in / `srt`=separate file, default: `overlay`) |
| `--resolution` | Output resolution (`auto` / `4k` / `1440p` / `fhd` / `720p`, default: `auto`) |
| `--style` | Editing style (see below, default: `balanced`) |
| `--min-day-duration N` | Minimum minutes per day; fills from discarded clips if under target |
| `--split-orientation` | Output landscape and portrait as separate files |
| `--max-segment N` | Max clip length in seconds before auto-split (default: 30) |
| `--workers N` | Number of parallel rendering workers |
| `--skip-transcribe` | Skip speech recognition (same as `--subtitle-lang off`) |
| `--no-stt-refine` | Disable LLM subtitle refinement |
| `--archive-dir DIR` | Move finished mp4 files to this directory after rendering |

### Editing Styles (`--style`)

Styles change both the AI persona and the evaluation criteria for silent/landscape clips.

| Style | AI persona | Silent landscape clips | Max kept |
|-------|-----------|----------------------|----------|
| `balanced` | Travel vlog producer | Trim if long, keep if short | 10s |
| `voice` | Travel vlog producer | **Remove all** | — |
| `vlog` | Travel vlog producer | Keep short transition cuts (≤5s) | 5s |
| `scene-short` | Travel video editor | Keep only key segments (trimmed) | 10s |
| `scene-long` | **Travel documentary director** | **Keep by default** (remove only bad quality) | 30s |
| `highlight` | Travel vlog producer | Keep only top-scoring clips | 8s |

```bash
python main.py ./videos ./output --style scene-long   # landscape / nature focus
python main.py ./videos ./output --style vlog         # talking-head focus
python main.py ./videos ./output --style highlight    # highlight reel
```

## Environment Variables / `.env`

Copy `core/.env.example` to `core/.env` and configure:

```env
# API keys (at least one required for AI features)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...          # Optional — fallback when Claude rate-limits
GEMINI_API_KEY=...             # Optional — fallback when OpenAI rate-limits

# Whisper
WHISPER_MODEL=large-v3         # tiny | base | small | medium | large-v3
WHISPER_DEVICE=cuda            # cuda | cpu
WHISPER_COMPUTE_TYPE=int8

# Subtitles
SUBTITLE_LANG=auto             # auto | ko | en | ja | zh | off
SUBTITLE_MODE=overlay          # overlay | srt

# Output
OUTPUT_RESOLUTION=auto         # auto | 4k | 1440p | fhd | 720p
VIDEO_CODEC=h265               # h264 | h265
CRF=24
FFMPEG_PRESET=medium           # ultrafast | fast | medium | slow

# GPU encoding (NVENC)
USE_NVENC=auto                 # auto | true | false
NVENC_PRESET=p4                # p1 (fastest) ~ p7 (best quality)
NVENC_MAX_SESSIONS=3           # 3 for GeForce; increase for Quadro / RTX A-series

# Editing
EDIT_STYLE=balanced
```

Without `ANTHROPIC_API_KEY`, clip evaluation falls back to rule-based scoring automatically.  
`.env` is listed in `.gitignore` and will never be committed.

## AI Clip Scoring

Claude AI scores each clip from 0 to 100 across four categories:

| Category | Description |
|----------|-------------|
| **Visual** | Recording quality — sharpness, shake, exposure |
| **Speech** | Voice clarity and background noise |
| **Scene** | Interest level of the scenery or content |
| **Flow** | Whether the clip is necessary for editing continuity |

Clips with low scores are automatically removed. A 2–3 sentence reason is printed for each clip.

## Output Structure

```
output/
├── travel_2024-07-15.mp4     # Final edited video per day
├── travel_2024-07-15.srt     # Subtitle file (SRT mode only)
├── .cache/                   # Intermediate files (reused on re-run)
└── rendered/                 # Rendered clips per day
```
