"""얼굴 모자이크 처리 모듈

InsightFace(SCRFD/buffalo_sc) + GPU 우선, CPU 폴백.
detect_interval 프레임마다 검출 후 중간 프레임은 이전 결과 재사용.
FFmpeg 파이프로 재인코딩 — 오디오 스트림 보존.
"""
import os
import subprocess
import cv2
import numpy as np
from typing import Optional

# ─── 얼굴 검출기 (싱글톤) ─────────────────────────────────────────────────────
_app = None
_app_gpu: Optional[bool] = None


def _get_app(use_gpu: bool):
    global _app, _app_gpu
    if _app is not None and _app_gpu == use_gpu:
        return _app
    try:
        from insightface.app import FaceAnalysis
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if use_gpu else
            ['CPUExecutionProvider']
        )
        a = FaceAnalysis(name='buffalo_sc', providers=providers)
        a.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        _app = a
        _app_gpu = use_gpu
        return _app
    except Exception as e:
        print(f"  [모자이크] InsightFace 초기화 실패: {e}")
        return None


# ─── OpenCV DNN 폴백 ──────────────────────────────────────────────────────────
_dnn_net = None

def _get_dnn():
    global _dnn_net
    if _dnn_net is not None:
        return _dnn_net
    try:
        import urllib.request
        model_dir = os.path.join(os.path.dirname(__file__), "..", ".cache_models")
        os.makedirs(model_dir, exist_ok=True)
        proto = os.path.join(model_dir, "deploy.prototxt")
        caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists(proto):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                proto,
            )
        if not os.path.exists(caffemodel):
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                caffemodel,
            )
        _dnn_net = cv2.dnn.readNetFromCaffe(proto, caffemodel)
        return _dnn_net
    except Exception as e:
        print(f"  [모자이크] OpenCV DNN 로드 실패: {e}")
        return None


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
) -> bool:
    """
    동영상에 얼굴 모자이크를 적용한다.

    Parameters
    ----------
    input_path      : 원본 MP4
    output_path     : 출력 MP4 (원본과 달라도 됨, 덮어쓰기 가능)
    use_gpu         : InsightFace CUDAExecutionProvider 사용 여부
    detect_interval : N 프레임마다 얼굴 검출 (중간은 이전 결과 재사용)
    codec           : FFmpeg 비디오 코덱
    crf             : 재인코딩 CRF

    Returns True on success.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [모자이크] 영상 열기 실패: {input_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # InsightFace 우선, 실패 시 DNN
    app = _get_app(use_gpu)
    net = None if app else _get_dnn()
    if app is None and net is None:
        print("  [모자이크] 검출기 없음 — 건너뜀")
        cap.release()
        return False

    # FFmpeg 파이프 열기 (오디오는 원본에서 복사)
    tmp_path = output_path + ".mosaic_tmp.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",          # 모자이크된 영상
        "-i", input_path,        # 오디오 소스
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", codec,
        "-crf", str(crf),
        "-c:a", "copy",
        "-shortest",
        tmp_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_idx  = 0
    last_boxes: list = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 얼굴 검출 (N 프레임마다)
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
        print(f"  [모자이크] FFmpeg 재인코딩 실패 (returncode={proc.returncode})")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    # 성공 → 원본 교체
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)
    print(f"  [모자이크] {frame_idx}프레임 처리 완료: {os.path.basename(output_path)}")
    return True
