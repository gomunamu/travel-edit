# Travel Editor

여행 동영상을 AI가 자동으로 편집해주는 도구입니다.  
입력 폴더의 동영상을 분석해 날짜별로 컷 편집, (음성인식)자막 생성, 장소 표시까지 자동으로 처리합니다.

## 주요 기능

- **자동 컷 편집** — 짧거나 불필요한 클립 자동 제거, 긴 클립 자동 분할
- **AI 클립 평가** — Claude AI로 풍경/내용 품질 평가 후 컷 선별
- **자막 자동 생성** — Whisper로 음성 인식, 영상 번인 또는 SRT 파일 출력
- **다국어 자막** — 한국어/영어 자동감지, 일본어·중국어 지정 가능
- **장소 표시** — GPS 메타데이터 기반 촬영 장소 자동 표시 (도시, 국가 형식)
- **날짜별 출력** — 촬영 날짜 기준으로 영상 자동 분류 및 합본 생성

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

## 사용법

```bash
python main.py <입력폴더> <출력폴더> [옵션]
```

### 예시

```bash
# 기본 실행 (한/영 자동감지, 영상 번인)
python main.py ~/여행사진 ~/편집결과

# AI 없이 규칙 기반으로만 처리
python main.py /media/usb/DCIM ./output --no-ai

# 일본어 자막, 영상 번인
python main.py ./videos ./output --subtitle-lang ja

# 한국어 자막, 별도 SRT 파일로 저장
python main.py ./videos ./output --subtitle-lang ko --subtitle-mode srt

# 자막 없이 빠르게 처리
python main.py ./videos ./output --subtitle-lang off
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--no-ai` | Claude AI 평가 없이 규칙 기반으로만 처리 |
| `--whisper-model` | Whisper 모델 크기 (`tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`) |
| `--subtitle-lang` | 자막 언어 (`auto` / `ko` / `en` / `ja` / `zh` / `off`, 기본: `auto`) |
| `--subtitle-mode` | 자막 방식 (`overlay`=영상 번인 / `srt`=별도 파일, 기본: `overlay`) |
| `--max-segment N` | 클립 최대 길이(초), 이보다 길면 자동 분할 (기본: 30) |
| `--workers N` | 렌더링 병렬 워커 수 |
| `--skip-transcribe` | 음성 인식 건너뜀 (`--subtitle-lang off`과 동일) |

## 환경변수 / .env

프로젝트 루트에 `.env` 파일을 만들어 설정할 수 있습니다.

```env
ANTHROPIC_API_KEY=sk-ant-...   # Claude AI 사용 시 필요
WHISPER_MODEL=large-v3         # tiny | base | small | medium | large-v2 | large-v3
WHISPER_DEVICE=cuda            # cuda | cpu
WHISPER_COMPUTE_TYPE=float16   # float16 | int8
SUBTITLE_LANG=auto             # auto | ko | en | ja | zh | off
SUBTITLE_MODE=overlay          # overlay | srt
```

- `ANTHROPIC_API_KEY` 없으면 규칙 기반 클립 평가로 자동 대체
- `.env`는 `.gitignore`에 포함되어 있어 git에 업로드되지 않음

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
| `OUTPUT_RESOLUTION` | 1920x1080 | 출력 해상도 |
| `OUTPUT_FPS` | 30 | 출력 프레임레이트 |
| `CRF` | 18 | 화질 (낮을수록 좋음) |
| `WHISPER_MODEL` | large-v3 | 음성 인식 모델 |
| `SUBTITLE_LANG` | auto | 자막 언어 |
| `SUBTITLE_MODE` | overlay | 자막 방식 |
| `MAX_SEGMENT_DURATION` | 30초 | 클립 최대 길이 |
| `MIN_SEGMENT_DURATION` | 2초 | 클립 최소 길이 |
