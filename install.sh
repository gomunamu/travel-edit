#!/bin/bash
# =============================================================================
# Travel Video Editor - 설치 스크립트
# Ubuntu 22.04 LTS / macOS (Apple Silicon & Intel) 지원
#
# CUDA 아키텍처 (Linux):
#   [시스템] CUDA Toolkit (apt)  → ffmpeg NVENC/NVDEC, nvidia-smi
#   [venv]   PyTorch+cuXX (pip)  → PyTorch, faster-whisper/ctranslate2
#            각 pip 패키지가 cuDNN을 내장하므로 시스템 cuDNN 불필요
#
# macOS:
#   VideoToolbox (내장)          → ffmpeg GPU 인코딩·디코딩
#   MPS는 ctranslate2 미지원     → Whisper는 CPU로 동작
# =============================================================================
set -e

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✗]${NC} $1"; }
section() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"    # Darwin | Linux
ARCH="$(uname -m)"  # arm64 | x86_64

# =============================================================================
# 0. 인수 파싱
# =============================================================================
TORCH_CUDA="cu128"           # pip wheel CUDA 버전 (cu118 / cu121 / cu124 / cu128)
SKIP_TORCH=false
SKIP_CUDA_CHECK=false

for arg in "$@"; do
    case $arg in
        --torch-cuda=*) TORCH_CUDA="${arg#*=}" ;;
        --skip-torch)   SKIP_TORCH=true ;;
        --cpu-only)     SKIP_TORCH=false; TORCH_CUDA="cpu" ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo "  --torch-cuda=VER   pip wheel CUDA 버전 (기본: cu128, Linux 전용)"
            echo "                     선택: cu118 | cu121 | cu124 | cu128"
            echo "  --skip-torch       PyTorch 설치 건너뜀 (이미 설치된 경우)"
            echo "  --cpu-only         CPU 전용 PyTorch 설치"
            echo ""
            echo "  macOS에서는 --torch-cuda 옵션이 무시됩니다."
            echo "  VideoToolbox는 자동 감지됩니다 (별도 설치 불필요)."
            exit 0
            ;;
    esac
done

# =============================================================================
# 1. 환경 확인
# =============================================================================
section "환경 확인"

info "OS: $OS ($ARCH)"
info "Python: $(python3 --version 2>&1 | awk '{print $2}')"

if [ "$OS" = "Darwin" ]; then
    info "macOS 감지 → VideoToolbox 인코딩 경로로 설치"
    if [ "$ARCH" = "arm64" ]; then
        info "Apple Silicon (arm64) — MPS 감지됨 (Whisper는 ctranslate2 제약으로 CPU 사용)"
    else
        info "Intel Mac (x86_64)"
    fi
elif [ "$OS" = "Linux" ]; then
    UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
    if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
        warn "Ubuntu 22.04 기준 스크립트입니다. 다른 Linux 배포판에서는 패키지명이 다를 수 있습니다."
    fi
    info "Ubuntu $UBUNTU_VER"
else
    warn "지원하지 않는 OS: $OS"
fi

# =============================================================================
# 2. GPU 확인
# =============================================================================
section "GPU 확인"

if [ "$OS" = "Darwin" ]; then
    # macOS: VideoToolbox는 항상 사용 가능 (시스템 프레임워크)
    info "VideoToolbox: macOS 내장 — 별도 설치 불필요"
    GPU_MODEL=$(system_profiler SPDisplaysDataType 2>/dev/null | awk '/Chipset Model/{print $NF; exit}')
    [ -n "$GPU_MODEL" ] && info "GPU: $GPU_MODEL"
else
    # Linux: NVIDIA 드라이버 확인
    if command -v nvidia-smi &>/dev/null; then
        DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        info "GPU: $GPU_NAME"
        info "드라이버: $DRIVER_VER"
    else
        warn "NVIDIA 드라이버 미설치 → GPU 기능 전부 비활성화됩니다."
        warn ""
        warn "설치 방법 (권장: 최신 production 드라이버):"
        warn "  sudo apt install nvidia-driver-570"
        warn "  sudo reboot"
        warn ""
        warn "설치 후 이 스크립트를 다시 실행하세요."
    fi
fi

# =============================================================================
# 3. 시스템 패키지 (OS별)
# =============================================================================

if [ "$OS" = "Darwin" ]; then
    section "Homebrew 패키지 설치"

    if ! command -v brew &>/dev/null; then
        error "Homebrew가 설치되어 있지 않습니다."
        error "설치: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    info "Homebrew: $(brew --version | head -1)"

    brew install ffmpeg python3
    info "ffmpeg / python3 설치 완료"

    # ffmpeg VideoToolbox 지원 확인
    if ffmpeg -encoders 2>/dev/null | grep -q "h264_videotoolbox"; then
        info "ffmpeg VideoToolbox (h264/hevc): 지원됨"
    else
        warn "ffmpeg VideoToolbox 미지원 → CPU 인코딩으로 동작"
    fi

    # 한글 자막 폰트: Apple SD Gothic Neo는 macOS 내장이므로 별도 설치 불필요
    info "폰트: 'Apple SD Gothic Neo' (macOS 내장) — 추가 설치 불필요"
    # 원한다면: brew install --cask font-nanum-gothic

else
    # Linux (Ubuntu)
    section "시스템 CUDA Toolkit (ffmpeg NVENC/NVDEC용)"

    echo "  시스템 CUDA는 ffmpeg의 GPU 인코딩(NVENC)·디코딩(NVDEC)에 필요합니다."
    echo "  Python 패키지(PyTorch/faster-whisper)는 pip로 설치되는 CUDA를 따로 사용합니다."

    if dpkg -l 2>/dev/null | grep -q "cuda-toolkit-[0-9]"; then
        CUDA_VER=$(dpkg -l | grep "cuda-toolkit-[0-9]-[0-9]" | awk '{print $2}' | head -1)
        CUDA_PATH=$(ls -d /usr/local/cuda-1[2-9]* 2>/dev/null | sort -V | tail -1)
        info "시스템 CUDA Toolkit: $CUDA_VER"
        [ -n "$CUDA_PATH" ] && info "경로: $CUDA_PATH (심볼릭: /usr/local/cuda)"
    else
        warn "시스템 CUDA Toolkit 미설치. ffmpeg NVENC/NVDEC 사용 불가."
        warn ""
        warn "설치 방법 (CUDA 12.x, Ubuntu 22.04):"
        warn "  # 1) NVIDIA apt 저장소 키 등록"
        warn "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
        warn "  sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        warn "  sudo apt-get update"
        warn "  # 2) CUDA Toolkit 설치 (버전은 드라이버 지원 범위 내에서 선택)"
        warn "  sudo apt-get install -y cuda-toolkit-12-4"
        warn ""
        warn "  드라이버별 지원 CUDA 최대 버전:"
        warn "    드라이버 535.x → CUDA 12.2"
        warn "    드라이버 550.x → CUDA 12.4"
        warn "    드라이버 560.x → CUDA 12.6"
        warn "    드라이버 570.x → CUDA 12.8"
    fi

    # LD_LIBRARY_PATH에 구버전 CUDA 경로 경고
    if echo "$LD_LIBRARY_PATH" | grep -qE "cuda-[0-9]+\.[0-9]+"; then
        OLD_CUDA=$(echo "$LD_LIBRARY_PATH" | grep -oE "cuda-[0-9]+\.[0-9]+" | head -1)
        warn "LD_LIBRARY_PATH에 구버전 $OLD_CUDA 경로가 포함되어 있습니다."
        warn "최신 CUDA와 충돌할 수 있으니 ~/.bashrc에서 제거하거나 최신 버전으로 교체하세요:"
        warn "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    fi
fi

# =============================================================================
# 4. 시스템 패키지 (Linux apt)
# =============================================================================

if [ "$OS" = "Linux" ]; then
    section "시스템 패키지 설치 (apt)"

    sudo apt-get update -qq

    APT_PACKAGES=(
        ffmpeg                  # 영상 인코딩·디코딩 (Ubuntu 22.04 기본 패키지에 NVENC/NVDEC 포함)
        python3-venv            # Python 가상환경
        python3-pip             # pip 기본
        python3-dev             # Python C 확장 빌드용 헤더
        git                     # 소스 관리
        fonts-nanum             # 한글 자막: 나눔 폰트
        fonts-noto-cjk          # 한글/중국어/일본어 자막: Noto CJK
        libass9                 # ASS/SSA 자막 렌더링 (ffmpeg 자막 번인용)
    )

    sudo apt-get install -y "${APT_PACKAGES[@]}"
    info "시스템 패키지 설치 완료"

    # ffmpeg GPU 지원 확인
    if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
        info "ffmpeg NVENC (h264/hevc): 지원됨"
    else
        warn "ffmpeg NVENC 미지원 → CPU 인코딩으로 동작 (시스템 CUDA 미설치 또는 드라이버 없음)"
    fi
    if ffmpeg -decoders 2>/dev/null | grep -q "h264_cuvid"; then
        info "ffmpeg NVDEC (h264_cuvid/hevc_cuvid): 지원됨"
    else
        warn "ffmpeg NVDEC 미지원 → CPU 디코딩으로 동작"
    fi
fi

# =============================================================================
# 5. Python 가상환경 생성
# =============================================================================
section "Python 가상환경 설정 (~/venvs/torch)"

VENV_DIR="$HOME/venvs/torch"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ -d "$VENV_DIR" ]; then
    info "가상환경 이미 존재: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    info "가상환경 생성: $VENV_DIR"
fi

"$VENV_PIP" install --upgrade pip -q
info "pip 업그레이드 완료"

# =============================================================================
# 6. PyTorch (pip) — Python 패키지용
# =============================================================================

if [ "$OS" = "Darwin" ]; then
    section "PyTorch 설치 (macOS)"
    echo "  macOS 표준 wheel: MPS 포함 (Apple Silicon 자동 감지)"
    echo "  ※ ctranslate2(faster-whisper)는 MPS 미지원 → Whisper는 CPU 동작"
else
    section "PyTorch 설치 (venv CUDA: $TORCH_CUDA)"
    echo "  PyTorch pip wheel은 cuDNN을 내장하므로 시스템 cuDNN 설치가 불필요합니다."
    echo "  faster-whisper/ctranslate2도 동일하게 자체 CUDA 라이브러리를 사용합니다."
fi

if [ "$SKIP_TORCH" = true ]; then
    info "PyTorch 설치 건너뜀 (--skip-torch)"
elif "$VENV_PYTHON" -c "import torch; print('already installed')" 2>/dev/null | grep -q "already installed"; then
    TORCH_VER=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null)
    info "PyTorch 이미 설치됨: $TORCH_VER (건너뜀)"
else
    if [ "$OS" = "Darwin" ]; then
        # macOS: 표준 wheel이 MPS를 포함 (별도 인덱스 불필요)
        "$VENV_PIP" install torch torchvision torchaudio
    elif [ "$TORCH_CUDA" = "cpu" ]; then
        "$VENV_PIP" install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
    else
        echo "  설치 인덱스: https://download.pytorch.org/whl/${TORCH_CUDA}"
        "$VENV_PIP" install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    fi
    info "PyTorch 설치 완료"
fi

# =============================================================================
# 7. Python 패키지 설치 (requirements.txt)
# =============================================================================
section "Python 패키지 설치 (requirements.txt)"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    error "requirements.txt를 찾을 수 없습니다: $SCRIPT_DIR/requirements.txt"
    exit 1
fi

"$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
info "requirements.txt 설치 완료"

# =============================================================================
# 8. 설치 결과 검증
# =============================================================================
section "설치 결과 검증"

# ffmpeg
if command -v ffmpeg &>/dev/null; then
    info "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    error "ffmpeg 설치 실패"
fi

# PyTorch
if "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
    TORCH_VER=$("$VENV_PYTHON" -c "import torch; print(torch.__version__)")
    if [ "$OS" = "Darwin" ]; then
        MPS_AVAIL=$("$VENV_PYTHON" -c "import torch; print(torch.backends.mps.is_available())")
        info "PyTorch $TORCH_VER  /  MPS available: $MPS_AVAIL"
        info "  ※ Whisper는 ctranslate2 제약으로 CPU 동작 (MPS 미지원)"
    else
        CUDA_AVAIL=$("$VENV_PYTHON" -c "import torch; print(torch.cuda.is_available())")
        CUDA_VER_TORCH=$("$VENV_PYTHON" -c "import torch; print(torch.version.cuda or 'N/A')")
        info "PyTorch $TORCH_VER  /  CUDA available: $CUDA_AVAIL  /  CUDA version: $CUDA_VER_TORCH"
    fi
else
    warn "PyTorch 미설치 (CPU 전용 모드로 동작)"
fi

# CTranslate2 (faster-whisper 엔진)
if "$VENV_PYTHON" -c "import ctranslate2" 2>/dev/null; then
    CT2_VER=$("$VENV_PYTHON" -c "import ctranslate2; print(ctranslate2.__version__)")
    if [ "$OS" = "Darwin" ]; then
        info "CTranslate2 $CT2_VER  /  device: cpu (macOS)"
    else
        CUDA_DEVS=$("$VENV_PYTHON" -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())")
        info "CTranslate2 $CT2_VER  /  CUDA devices: $CUDA_DEVS"
    fi
fi

# Python 패키지
PACKAGES=("anthropic" "openai" "google.genai" "faster_whisper" "reverse_geocoder" "tqdm")
PKGNAMES=("anthropic" "openai" "google-genai" "faster-whisper" "reverse_geocoder" "tqdm")
for i in "${!PACKAGES[@]}"; do
    if "$VENV_PYTHON" -c "import ${PACKAGES[$i]}" 2>/dev/null; then
        info "${PKGNAMES[$i]}: OK"
    else
        error "${PKGNAMES[$i]}: 설치 실패"
    fi
done

# =============================================================================
# 9. 실행 안내
# =============================================================================
section "설치 완료"

cat <<'GUIDE'

─── 실행 방법 ───────────────────────────────────────────
  # 방법 1: venv 활성화 후 실행
  source ~/venvs/torch/bin/activate
  python main.py <입력폴더> <출력폴더>

  # 방법 2: venv Python 직접 지정
  ~/venvs/torch/bin/python main.py <입력폴더> <출력폴더>

─── API 키 설정 (.env) ──────────────────────────────────
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=...

GUIDE

if [ "$OS" = "Darwin" ]; then
    cat <<'MACOS_GUIDE'
─── macOS 추가 정보 ─────────────────────────────────────
  인코딩:  VideoToolbox 하드웨어 인코더 자동 사용
  Whisper: CPU 동작 (ctranslate2 MPS 미지원)
           M2/M3 기준 large-v3 모델도 실용적인 속도로 동작

  편의 alias 등록 (zsh):
    echo "alias venv='source ~/venvs/torch/bin/activate'" >> ~/.zshrc
    source ~/.zshrc

MACOS_GUIDE
else
    cat <<'LINUX_GUIDE'
─── LD_LIBRARY_PATH 정리 (권장) ─────────────────────────
  # 구버전 CUDA 경로 제거, 현재 시스템 CUDA 경로로 통일
  # ~/.bashrc 에 추가:
  export CUDA_HOME=/usr/local/cuda
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

  편의 alias 등록 (bash):
    echo "alias venv='source ~/venvs/torch/bin/activate'" >> ~/.bashrc
    source ~/.bashrc

LINUX_GUIDE
fi
