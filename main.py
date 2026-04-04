#!/usr/bin/env python3
"""
여행 동영상 자동 편집기
Usage: python main.py <입력폴더> <출력폴더> [옵션]
"""
import argparse
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """필수 의존성 확인"""
    import subprocess

    # ffmpeg/ffprobe 확인
    for tool in ["ffmpeg", "ffprobe"]:
        result = subprocess.run([tool, "-version"], capture_output=True)
        if result.returncode != 0:
            print(f"[오류] {tool}가 설치되어 있지 않습니다.")
            print(f"  Ubuntu: sudo apt install ffmpeg")
            print(f"  macOS:  brew install ffmpeg")
            sys.exit(1)

    # Python 패키지 확인
    missing = []
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    try:
        import reverse_geocoder
    except ImportError:
        missing.append("reverse_geocoder")
    try:
        import anthropic
    except ImportError:
        missing.append("anthropic")
    try:
        import tqdm
    except ImportError:
        missing.append("tqdm")

    if missing:
        print(f"[경고] 다음 패키지가 설치되어 있지 않습니다: {', '.join(missing)}")
        print(f"  설치: pip install {' '.join(missing)}")
        if "faster-whisper" in missing:
            print("[오류] faster-whisper는 필수입니다.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="여행 동영상 자동 편집기 - AI 기반 컷 편집 및 자막 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py ~/여행사진 ~/편집결과
  python main.py /media/usb/DCIM ./output --no-ai
  python main.py ./videos ./output --whisper-model large-v3

환경변수:
  ANTHROPIC_API_KEY  Claude AI API 키 (없으면 규칙 기반 평가 사용)

출력:
  ./output/travel_2024-07-15.mp4   날짜별 편집 완료 영상
  ./output/.cache/                 중간 작업 파일 (재실행시 재사용)
  ./output/rendered/               날짜별 렌더링된 클립들
        """
    )

    parser.add_argument("input", help="입력 폴더 (동영상/사진 혼합 가능)")
    parser.add_argument("output", help="출력 폴더")
    parser.add_argument(
        "--no-ai", action="store_true",
        help="Claude AI 평가 없이 규칙 기반으로만 처리"
    )
    parser.add_argument(
        "--whisper-model", default=None,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper 모델 크기 (기본: config.py 설정값)"
    )
    parser.add_argument(
        "--max-segment", type=int, default=None,
        help="클립 최대 길이(초), 이보다 길면 자동 분할 (기본: 30)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="렌더링 병렬 워커 수, 곧 NAS 동시 연결 수 상한 (기본: CPU코어수//2)"
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="음성 인식 건너뜀 (자막 없음, 빠름)"
    )
    parser.add_argument(
        "--subtitle-lang", default=None,
        choices=["auto", "ko", "en", "ja", "zh", "off"],
        help="자막 언어 (auto=한/영 자동감지, ko=한국어, en=영어, ja=일본어, zh=중국어, off=없음 / 기본: auto)"
    )
    parser.add_argument(
        "--subtitle-mode", default=None,
        choices=["overlay", "srt"],
        help="자막 방식 (overlay=영상 번인, srt=별도 .srt 파일 / 기본: overlay)"
    )
    parser.add_argument(
        "--no-stt-refine", action="store_true",
        help="STT 결과 LLM 정제 비활성화 (기본: 활성화)"
    )

    args = parser.parse_args()

    # 설정 오버라이드
    import config
    if args.no_ai:
        config.ANTHROPIC_API_KEY = ""
        print("[알림] AI 평가 비활성화 - 규칙 기반으로 처리합니다.")
    if args.whisper_model:
        config.WHISPER_MODEL = args.whisper_model
    if args.max_segment:
        config.MAX_SEGMENT_DURATION = args.max_segment
    if args.workers:
        config.RENDER_WORKERS = args.workers
    if args.subtitle_lang:
        config.SUBTITLE_LANG = args.subtitle_lang
    if args.subtitle_mode:
        config.SUBTITLE_MODE = args.subtitle_mode
    if args.skip_transcribe:
        config.SUBTITLE_LANG = "off"
    if args.no_stt_refine:
        config.STT_REFINE = False

    # 의존성 확인
    check_dependencies()

    # 입력 폴더 확인
    if not os.path.isdir(args.input):
        print(f"[오류] 입력 폴더가 존재하지 않습니다: {args.input}")
        sys.exit(1)

    # API 키 안내
    if not config.ANTHROPIC_API_KEY:
        print("[알림] ANTHROPIC_API_KEY 없음 - 규칙 기반 클립 평가를 사용합니다.")
        print("       AI 평가를 사용하려면 .env에 ANTHROPIC_API_KEY=sk-ant-... 추가\n")

    # Whisper / 자막 설정 명시
    if config.SUBTITLE_LANG == "off":
        print("[음성인식] 건너뜀 (자막 없음)")
    else:
        whisper_src = "CLI 인수" if args.whisper_model else ".env / 기본값"
        lang_src    = "CLI 인수" if args.subtitle_lang else ".env / 기본값"
        mode_src    = "CLI 인수" if args.subtitle_mode else ".env / 기본값"
        lang_label  = {"auto": "자동(한/영)", "ko": "한국어", "en": "영어",
                       "ja": "일본어", "zh": "중국어"}.get(config.SUBTITLE_LANG, config.SUBTITLE_LANG)
        mode_label  = {"overlay": "영상 번인", "srt": "별도 SRT 파일"}.get(config.SUBTITLE_MODE, config.SUBTITLE_MODE)
        refine_label = "ON" if config.STT_REFINE else "OFF"
        refine_src   = "CLI 인수" if args.no_stt_refine else ".env / 기본값"
        print(f"[음성인식] Whisper 모델: {config.WHISPER_MODEL}  ({whisper_src})")
        print(f"           디바이스: {config.WHISPER_DEVICE.upper()}  |  연산타입: {config.WHISPER_COMPUTE_TYPE}")
        print(f"[자막]     언어: {lang_label} ({lang_src})  |  방식: {mode_label} ({mode_src})")
        print(f"[STT 정제] {refine_label}  모델: {config.STT_REFINE_MODEL}  ({refine_src})\n")

    from core.pipeline import run
    run(args.input, args.output)


if __name__ == "__main__":
    main()
