#!/bin/bash
# =============================================================================
# Travel Video Editor - 설치 스크립트
# Ubuntu 22.04 LTS 기준
# =============================================================================
set -e

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
section() { echo -e "\n${GREEN}━━━ $1 ━━━${NC}"; }

# =============================================================================
# 1. OS 확인
# =============================================================================
section "환경 확인"

if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    warn "Ubuntu 22.04 기준으로 작성된 스크립트입니다. 다른 OS에서는 패키지명이 다를 수 있습니다."
fi

UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "unknown")
info "OS: Ubuntu $UBUNTU_VER / Python $(python3 --version 2>&1 | awk '{print $2}')"

# =============================================================================
# 2. 시스템 패키지 (apt)
# =============================================================================
section "시스템 패키지 설치 (apt)"

sudo apt-get update -qq

APT_PACKAGES=(
    ffmpeg                  # 영상 인코딩·디코딩 (NVENC/NVDEC 포함)
    python3-venv            # Python 가상환경
    python3-pip             # pip 기본
    python3-dev             # Python C 확장 빌드용 헤더
    git                     # 소스 관리
    fonts-nanum             # 한글 자막: 나눔 폰트
    fonts-noto-cjk          # 한글/중국어/일본어 자막: Noto CJK
    libass9                 # ASS/SSA 자막 렌더링 라이브러리 (ffmpeg 자막 번인용)
)

sudo apt-get install -y "${APT_PACKAGES[@]}"
info "시스템 패키지 설치 완료"

# ffmpeg NVENC/NVDEC 지원 확인
if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    info "ffmpeg: NVENC(h264/hevc) 지원 확인"
else
    warn "ffmpeg: NVENC 미지원. GPU 인코딩 비활성화됨 (CPU 인코딩으로 동작)"
fi
if ffmpeg -decoders 2>/dev/null | grep -q "h264_cuvid"; then
    info "ffmpeg: NVDEC(h264_cuvid/hevc_cuvid) 지원 확인"
else
    warn "ffmpeg: NVDEC 미지원 (CPU 디코딩으로 동작)"
fi

# =============================================================================
# 3. NVIDIA GPU / CUDA / cuDNN 확인
# =============================================================================
section "NVIDIA GPU / CUDA / cuDNN 확인"

if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "GPU: $GPU_NAME  /  드라이버: $DRIVER_VER"
else
    warn "nvidia-smi 없음 → NVIDIA 드라이버 미설치. GPU 기능 비활성화됨."
    warn "설치: sudo apt install nvidia-driver-570  (또는 https://www.nvidia.com/drivers)"
fi

if dpkg -l | grep -q "cuda-toolkit"; then
    CUDA_VER=$(dpkg -l | grep "cuda-toolkit-[0-9]" | awk '{print $3}' | head -1)
    info "CUDA Toolkit: $CUDA_VER"
else
    warn "CUDA Toolkit 미설치."
    warn "설치 방법: https://developer.nvidia.com/cuda-downloads"
    warn "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    warn "  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update"
    warn "  sudo apt install cuda-toolkit-12-4"
fi

if dpkg -l | grep -q "libcudnn"; then
    CUDNN_VER=$(dpkg -l | grep "libcudnn" | awk '{print $3}' | head -1)
    info "cuDNN: $CUDNN_VER"
else
    warn "cuDNN 미설치. faster-whisper GPU 가속 불가."
    warn "설치 방법: https://developer.nvidia.com/cudnn"
    warn "  (CUDA Toolkit 설치 후: sudo apt install libcudnn9-cuda-12)"
fi

# =============================================================================
# 4. Python 가상환경 생성
# =============================================================================
section "Python 가상환경 설정 (~/venvs/torch)"

VENV_DIR="$HOME/venvs/torch"

if [ -d "$VENV_DIR" ]; then
    info "가상환경 이미 존재: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    info "가상환경 생성 완료: $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# pip 최신화
"$VENV_PIP" install --upgrade pip -q
info "pip 업그레이드 완료"

# =============================================================================
# 5. Python 패키지 설치
# =============================================================================
section "Python 패키지 설치 (requirements.txt)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    error "requirements.txt를 찾을 수 없습니다: $SCRIPT_DIR/requirements.txt"
    exit 1
fi

"$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
info "Python 패키지 설치 완료"

# =============================================================================
# 6. 설치 결과 검증
# =============================================================================
section "설치 결과 검증"

# ffmpeg
if command -v ffmpeg &>/dev/null; then
    info "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    error "ffmpeg 설치 실패"
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
# 7. 실행 안내
# =============================================================================
section "설치 완료"

cat <<EOF

사용 전 반드시 가상환경을 활성화하세요:

  source ~/venvs/torch/bin/activate
  python main.py <입력폴더> <출력폴더>

또는 매번 활성화 없이 직접 실행:

  ~/venvs/torch/bin/python main.py <입력폴더> <출력폴더>

편의를 위해 ~/.bashrc에 alias 등록 (선택):

  echo "alias venv='source ~/venvs/torch/bin/activate'" >> ~/.bashrc
  source ~/.bashrc

API 키 설정 (.env 파일):

  cp .env.example .env   # 없으면 직접 생성
  # .env 파일에 ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY 입력

EOF
