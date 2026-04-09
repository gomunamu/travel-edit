"""ffprobe 기반 비디오 메타데이터 추출"""
import subprocess
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v",
    ".mts", ".m2ts", ".3gp", ".wmv", ".hevc",
    ".ts", ".mxf", ".flv", ".webm"
}


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _run_ffprobe(filepath: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def _run_ffprobe_recover(filepath: str) -> dict:
    """손상된 파일 복구 시도: analyze_duration/probesize 확장 + fflags +igndts"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-fflags", "+igndts+genpts",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe recover failed: {result.stderr[:200]}")
    return json.loads(result.stdout)


def _parse_creation_time(tags: dict) -> Optional[datetime]:
    candidates = [
        tags.get("creation_time"),
        tags.get("com.apple.quicktime.creationdate"),
        tags.get("date_time_original"),
        tags.get("date"),
    ]
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
    ]
    for val in candidates:
        if not val:
            continue
        # Python 3.10 fromisoformat: Z suffix 미지원 → +00:00으로 변환
        val_norm = val.strip()
        if val_norm.endswith("Z"):
            val_norm = val_norm[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(val_norm)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
        # 폴백: 수동 포맷 파싱
        for fmt in formats:
            try:
                dt = datetime.strptime(val[:26].strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue
    return None


# 날짜 패턴: YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD
_DATE_PATTERNS = [
    re.compile(r'(\d{4})[_\-](\d{2})[_\-](\d{2})'),  # 2024-07-15 / 2024_07_15
    re.compile(r'(\d{4})(\d{2})(\d{2})'),              # 20240715
]


def _parse_date_from_text(text: str) -> Optional[datetime]:
    """문자열(파일명·경로)에서 날짜를 파싱한다."""
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2000 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return datetime(y, mo, d, tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _fallback_creation_time(filepath: str) -> datetime:
    """
    메타데이터에 날짜가 없을 때 3단계 fallback.
    1) 상위 디렉토리 경로명에서 날짜 파싱 (부모 → 조부모 순)
    2) 파일명에서 날짜 파싱
    3) 파일 mtime
    """
    p = Path(filepath)

    # 1) 디렉토리 경로 (하위 → 상위 순)
    for part in reversed(p.parts[:-1]):
        dt = _parse_date_from_text(part)
        if dt:
            return dt

    # 2) 파일명
    dt = _parse_date_from_text(p.stem)
    if dt:
        return dt

    # 3) mtime
    mtime = p.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def _parse_gps(tags: dict) -> Optional[tuple]:
    """ISO 6709 형식 또는 개별 GPS 태그에서 좌표 추출"""
    # Apple QuickTime / ISO 6709: "+37.5665+126.9780/"
    for key in ["location", "com.apple.quicktime.location.ISO6709"]:
        val = tags.get(key, "")
        if val:
            match = re.match(r'([+-]\d+\.?\d*)([+-]\d+\.?\d*)', val)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (round(lat, 6), round(lon, 6))

    # EXIF 스타일 개별 태그
    try:
        lat_ref = tags.get("GPSLatitudeRef", "N")
        lon_ref = tags.get("GPSLongitudeRef", "E")
        lat_str = tags.get("GPSLatitude")
        lon_str = tags.get("GPSLongitude")
        if lat_str and lon_str:
            def parse_dms(s):
                parts = re.findall(r'[\d.]+', s)
                d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
                return d + m / 60 + sec / 3600
            lat = parse_dms(lat_str) * (-1 if lat_ref == "S" else 1)
            lon = parse_dms(lon_str) * (-1 if lon_ref == "W" else 1)
            return (round(lat, 6), round(lon, 6))
    except Exception:
        pass

    return None


def _check_file_integrity(filepath: str) -> Optional[str]:
    """사전 검사: 빈 파일만 필터링. moov atom 검사는 NAS에서 너무 느려 ffprobe에 위임."""
    if Path(filepath).stat().st_size == 0:
        return "빈 파일 (0 bytes) — 전송/기록 실패"
    return None


def get_video_info(filepath: str) -> Optional[dict]:
    """비디오 파일의 모든 메타정보 추출. 실패 시 복구 옵션으로 재시도."""
    reason = _check_file_integrity(filepath)
    if reason:
        print(f"  [건너뜀] {Path(filepath).name}: {reason}")
        return None

    try:
        data = _run_ffprobe(filepath)
    except Exception as e:
        print(f"  [경고] ffprobe 실패, 복구 시도 중: {Path(filepath).name} - {e}")
        try:
            data = _run_ffprobe_recover(filepath)
            print(f"  [복구 성공] {Path(filepath).name}")
        except Exception as e2:
            print(f"  [오류] 복구 실패, 건너뜀: {Path(filepath).name} - {e2}")
            return None

    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        return None

    # 원본 해상도
    raw_w = int(video_stream.get("width", 0))
    raw_h = int(video_stream.get("height", 0))

    # 회전 정보 (side_data_list에서 추출)
    rotation = 0
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = int(sd["rotation"]) % 360

    # 실제 표시 해상도 (회전 적용)
    if rotation in (90, 270):
        display_w, display_h = raw_h, raw_w
    else:
        display_w, display_h = raw_w, raw_h

    is_portrait = display_h > display_w

    # FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except Exception:
        fps = 30.0

    # 비트레이트
    fmt = data.get("format", {})
    total_bitrate = int(fmt.get("bit_rate", 0)) // 1000  # kbps
    video_bitrate = int(video_stream.get("bit_rate", 0)) // 1000
    audio_bitrate = int(audio_stream.get("bit_rate", 0)) // 1000 if audio_stream else 0

    # 태그 수집 (format + video stream)
    all_tags = {}
    all_tags.update(fmt.get("tags", {}))
    all_tags.update(video_stream.get("tags", {}))

    creation_time = _parse_creation_time(all_tags)
    # 메타데이터에 날짜 없으면 경로/파일명/mtime으로 fallback
    date_from_meta = creation_time is not None
    if creation_time is None:
        creation_time = _fallback_creation_time(filepath)

    gps = _parse_gps(all_tags)

    duration = float(fmt.get("duration", 0))
    if duration == 0:
        duration = float(video_stream.get("duration", 0))

    return {
        "filepath": filepath,
        "filename": Path(filepath).name,
        "duration": duration,
        "raw_width": raw_w,
        "raw_height": raw_h,
        "display_width": display_w,
        "display_height": display_h,
        "is_portrait": is_portrait,
        "rotation": rotation,
        "fps": round(fps, 3),
        "video_codec": video_stream.get("codec_name", "h264"),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "has_audio": audio_stream is not None,
        "total_bitrate_kbps": total_bitrate,
        "video_bitrate_kbps": video_bitrate,
        "audio_bitrate_kbps": audio_bitrate if audio_bitrate else 192,
        "creation_time": creation_time.isoformat(),
        "creation_time_source": "metadata" if date_from_meta else "path/filename/mtime",
        "day_key": creation_time.astimezone().strftime("%Y-%m-%d"),
        "gps": list(gps) if gps else None,
    }
