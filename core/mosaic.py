"""얼굴 모자이크 처리 모듈

InsightFace(SCRFD/buffalo_sc) + GPU 우선, OpenCV DNN CPU 폴백.
- 스레드 로컬 검출기 인스턴스: 병렬 클립 처리 시 각 스레드가 독립된 세션 유지
- detect_interval 프레임마다 검출, 중간 프레임은 이전 결과 재사용
- FFmpeg 파이프로 재인코딩 — trim 구간 오디오 mux
"""
import os
import threading
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# ─── 스레드 로컬 검출기 ──────────────────────────────────────────────────────
_tls        = threading.local()   # 스레드별 detector 인스턴스
_init_lock  = threading.Lock()    # 초기화 직렬화 (다운로드·print)
_app_failed = False               # InsightFace 영구 실패 플래그
_dnn_failed = False               # OpenCV DNN 영구 실패 플래그
_dnn_model_ready = False          # DNN 모델 파일 준비 완료 플래그


def _get_app(use_gpu: bool):
    """스레드 로컬 InsightFace 인스턴스 반환. 실패 시 None."""
    global _app_failed
    if _app_failed:
        return None
    # 이 스레드에 이미 올바른 인스턴스가 있으면 즉시 반환
    if getattr(_tls, 'app_gpu', None) == use_gpu and getattr(_tls, 'app', None) is not None:
        return _tls.app
    try:
        from insightface.app import FaceAnalysis
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if use_gpu else
            ['CPUExecutionProvider']
        )
        a = FaceAnalysis(name='buffalo_sc', providers=providers)
        a.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        _tls.app = a
        _tls.app_gpu = use_gpu
        return _tls.app
    except Exception as e:
        with _init_lock:
            if not _app_failed:
                print(f"  [모자이크] InsightFace 초기화 실패: {e}")
                print(f"  [모자이크] 힌트: ~/venvs/torch/bin/pip install insightface onnxruntime-gpu")
                _app_failed = True
        return None


def _ensure_dnn_models() -> tuple[str, str] | None:
    """DNN 모델 파일 경로 반환. 필요 시 다운로드 (lock 내에서 1회만)."""
    global _dnn_failed, _dnn_model_ready
    if _dnn_failed:
        return None
    model_dir = os.path.join(os.path.dirname(__file__), "..", ".cache_models")
    proto      = os.path.join(model_dir, "deploy.prototxt")
    caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if _dnn_model_ready:
        return proto, caffemodel
    with _init_lock:
        if _dnn_failed:
            return None
        if _dnn_model_ready:
            return proto, caffemodel
        try:
            import urllib.request
            os.makedirs(model_dir, exist_ok=True)
            if not os.path.exists(proto):
                print("  [모자이크] OpenCV DNN 모델 다운로드 중 (deploy.prototxt)...")
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    proto,
                )
            if not os.path.exists(caffemodel):
                print("  [모자이크] OpenCV DNN 모델 다운로드 중 (res10 caffemodel, ~10MB)...")
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                    caffemodel,
                )
            _dnn_model_ready = True
            return proto, caffemodel
        except Exception as e:
            print(f"  [모자이크] OpenCV DNN 모델 준비 실패: {e}")
            _dnn_failed = True
            return None


def _get_dnn():
    """스레드 로컬 OpenCV DNN 인스턴스 반환 (DNN Net은 스레드 비안전)."""
    global _dnn_failed
    if _dnn_failed:
        return None
    if getattr(_tls, 'dnn_net', None) is not None:
        return _tls.dnn_net
    paths = _ensure_dnn_models()
    if paths is None:
        return None
    proto, caffemodel = paths
    try:
        net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
        _tls.dnn_net = net
        with _init_lock:
            # 첫 번째 스레드만 완료 메시지 출력
            if not getattr(_tls, '_dnn_announced', False):
                print("  [모자이크] OpenCV DNN 폴백 초기화 완료")
                _tls._dnn_announced = True
        return _tls.dnn_net
    except Exception as e:
        print(f"  [모자이크] OpenCV DNN 로드 실패: {e}")
        _dnn_failed = True
        return None


# ─── 얼굴 검출 (DNN) ─────────────────────────────────────────────────────────
def _detect_dnn(frame, net, conf_thr=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                  (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thr:
            continue
        x1 = max(0, int(dets[0, 0, i, 3] * w))
        y1 = max(0, int(dets[0, 0, i, 4] * h))
        x2 = min(w, int(dets[0, 0, i, 5] * w))
        y2 = min(h, int(dets[0, 0, i, 6] * h))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


# ─── 픽셀화 모자이크 ──────────────────────────────────────────────────────────
def _pixelate(frame, x1, y1, x2, y2, strength: int = 15):
    roi = frame[y1:y2, x1:x2]
    pw = max(1, (x2 - x1) // strength)
    ph = max(1, (y2 - y1) // strength)
    small = cv2.resize(roi, (pw, ph), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                       interpolation=cv2.INTER_NEAREST)


# ─── 공개 API ─────────────────────────────────────────────────────────────────
def is_korea(day_segs: list) -> bool:
    """세그먼트 목록 중 하나라도 한국(위도33~39 경도124~132) GPS면 True."""
    for seg in day_segs:
        gps = seg.get("gps")
        if gps and len(gps) >= 2:
            try:
                lat, lon = float(gps[0]), float(gps[1])
                if 33.0 <= lat <= 39.0 and 124.0 <= lon <= 132.0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def apply_face_mosaic(
    input_path: str,
    output_path: str,
    use_gpu: bool = True,
    detect_interval: int = 5,
    codec: str = "libx265",
    crf: int = 23,
    trim_start: float = 0.0,
    trim_end: Optional[float] = None,
) -> bool:
    """
    동영상 trim 구간에 얼굴 모자이크를 적용한다. 스레드 안전.

    Parameters
    ----------
    input_path      : 원본 영상
    output_path     : 출력 MP4 (원본과 다른 경로)
    use_gpu         : InsightFace CUDAExecutionProvider 사용 여부
    detect_interval : N 프레임마다 얼굴 검출 (중간 프레임은 이전 결과 재사용)
    codec           : FFmpeg 비디오 코덱
    crf             : 재인코딩 CRF
    trim_start      : 처리 시작 위치(초)
    trim_end        : 처리 종료 위치(초). None = 파일 끝까지

    Returns True on success.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [모자이크] 영상 열기 실패: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 스레드 로컬 검출기: InsightFace 우선, 실패 시 DNN
    app = _get_app(use_gpu)
    net = None if app else _get_dnn()
    if app is None and net is None:
        print(f"  [모자이크] 검출기 없음 — 건너뜀: {Path(input_path).name}")
        cap.release()
        return False

    # trim 구간 시크
    if trim_start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, trim_start * 1000)

    trim_dur = (trim_end - trim_start) if trim_end is not None else None
    if trim_dur is not None:
        audio_args = ["-ss", f"{trim_start:.3f}", "-t", f"{trim_dur:.3f}", "-i", input_path]
    else:
        audio_args = ["-ss", f"{trim_start:.3f}", "-i", input_path]

    tmp_path = output_path + ".mosaic_tmp.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        *audio_args,
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", codec, "-crf", str(crf),
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", tmp_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_idx  = 0
    last_boxes: list = []

    try:
        while True:
            if trim_end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= trim_end:
                break
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % detect_interval == 0:
                if app:
                    faces = app.get(frame)
                    last_boxes = []
                    for f in faces:
                        x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        if x2 > x1 and y2 > y1:
                            last_boxes.append((x1, y1, x2, y2))
                else:
                    last_boxes = _detect_dnn(frame, net)

            for box in last_boxes:
                _pixelate(frame, *box)

            proc.stdin.write(frame.tobytes())
            frame_idx += 1

    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()

    if proc.returncode != 0:
        print(f"  [모자이크] FFmpeg 재인코딩 실패 (returncode={proc.returncode}): {Path(input_path).name}")
        Path(tmp_path).unlink(missing_ok=True)
        return False

    Path(output_path).unlink(missing_ok=True)
    os.rename(tmp_path, output_path)
    dur_str = f"{trim_start:.1f}~{trim_end:.1f}초" if trim_end else "전체"
    print(f"  [모자이크] ✓ {frame_idx}프레임 ({dur_str}): {Path(output_path).name}")
    return True
