"""ASS 자막 파일 생성"""
from typing import List, Tuple, Optional


def _fmt_time(seconds: float) -> str:
    """ASS 시간 형식: H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def make_subtitle_ass(
    segments: List[dict],
    output_path: str,
    resolution: Tuple[int, int] = (1920, 1080),
    font: str = "Arial",
    font_size: int = 42,
    margin_v: int = 40,
    trim_offset: float = 0.0,
):
    """
    Whisper segments에서 ASS 자막 파일 생성.
    trim_offset: 세그먼트 시작 오프셋 (잘린 클립의 경우 자막 시간 보정)
    """
    W, H = resolution
    # 출력 해상도에 비례해서 폰트/여백 스케일 (기준: 1080p)
    scaled_font = max(1, int(font_size * H / 1080))
    scaled_margin_v = max(1, int(margin_v * H / 1080))
    speech_segs = [
        s for s in segments
        if s.get("text", "").strip()
        and s.get("no_speech_prob", 1.0) < 0.5
    ]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{font},{scaled_font},&H00FFFFFF,&H000000FF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,{scaled_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for i, seg in enumerate(speech_segs):
        start = max(0.0, seg["start"] - trim_offset)
        end   = max(seg["end"] - trim_offset, start + 0.3)
        # 다음 세그먼트 시작 직전에 종료 → 자막 겹침 방지
        if i + 1 < len(speech_segs):
            next_start = max(0.0, speech_segs[i + 1]["start"] - trim_offset)
            end = min(end, next_start - 0.05)
        end = max(start + 0.3, end)  # 최소 0.3초 표시
        text = seg["text"].replace("\n", "\\N").strip()
        if not text:
            continue
        # 페이드 효과
        events.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Sub,,0,0,0,,{{\\fad(150,150)}}{text}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        if events:
            f.write("\n")


def make_subtitle_srt(segments: List[dict], output_path: str):
    """
    segments 리스트에서 SRT 자막 파일 생성.
    각 segment: {"start": float, "end": float, "text": str}
    """
    def _srt_time(seconds: float) -> str:
        seconds = max(0.0, seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds % 1) * 1000))
        if ms >= 1000:
            ms = 999
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def make_location_ass(
    location_name: str,
    clip_duration: float,
    output_path: str,
    resolution: Tuple[int, int] = (1920, 1080),
    display_duration: float = 3.5,
    fade_duration: float = 0.4,
    font: str = "Arial",
    font_size: int = 26,
    margin: int = 20,
):
    """장소명 오버레이 ASS 파일 생성 (우하단, 페이드 인/아웃)"""
    W, H = resolution
    end_time = min(display_duration, clip_duration)
    # 출력 해상도에 비례해서 폰트/여백 스케일 (기준: 1080p)
    scaled_font = max(1, int(font_size * H / 1080))
    scaled_margin = max(1, int(margin * H / 1080))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Loc,{font},{scaled_font},&H00FFFFFF,&H000000FF,&H00000000,&HAA000000,0,0,0,0,100,100,0,0,1,2,1,1,{scaled_margin},{scaled_margin},{scaled_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Alignment 1 = 좌하단 (iPhone/Synology 스타일)
    fade_ms = int(fade_duration * 1000)
    text = location_name.replace(",", "\\,")
    event = (
        f"Dialogue: 0,{_fmt_time(0)},{_fmt_time(end_time)},Loc,,0,0,0,,"
        f"{{\\fad({fade_ms},{fade_ms})}}{text}\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(event)
