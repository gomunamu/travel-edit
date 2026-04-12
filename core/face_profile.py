"""가족 얼굴 프로파일 스캔

InsightFace buffalo_sc의 recognition 임베딩(512차원)으로
입력 폴더 영상에서 가장 자주 등장하는 상위 N명을 클러스터링.

cosine 유사도 > SIMILARITY_THRESHOLD → 동일인
"""
import cv2
import numpy as np
import tempfile
from pathlib import Path
from typing import Optional

SIMILARITY_THRESHOLD = 0.45   # 동일인 판정 기준
SAMPLE_INTERVAL      = 30     # N프레임마다 1회 샘플링


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def scan_top_faces(
    input_dir: str,
    n: int = 5,
    sample_interval: int = SAMPLE_INTERVAL,
    threshold: float = SIMILARITY_THRESHOLD,
    progress_cb=None,          # callable(current, total, msg) — 선택
) -> list[dict]:
    """
    입력 폴더의 모든 영상을 샘플링하여 상위 n명의 얼굴 클러스터 반환.

    Returns
    -------
    list[dict] (count 내림차순):
        count       : 등장 프레임 수
        embedding   : 정규화된 대표 임베딩 (np.ndarray, 512차원)
        image_path  : 대표 얼굴 크롭 JPEG 경로 (임시 파일)
        dominant    : 압도적으로 자주 등장하는지 여부 (최대 count × 60% 이상)
    """
    from core.mosaic import _get_app, _ensure_insightface_importable
    _ensure_insightface_importable()

    app = _get_app(use_gpu=True) or _get_app(use_gpu=False)
    if app is None:
        return []

    exts = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mts', '.ts'}
    video_files = sorted(
        p for p in Path(input_dir).rglob('*')
        if p.suffix.lower() in exts
    )
    if not video_files:
        return []

    # ── 클러스터: list of {emb, count, best_frame, best_bbox, best_area}
    clusters: list[dict] = []

    for vi, vpath in enumerate(video_files):
        if progress_cb:
            progress_cb(vi, len(video_files), f"스캔 중: {vpath.name}")
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        fidx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fidx % sample_interval != 0:
                    fidx += 1
                    continue
                for face in app.get(frame):
                    emb_raw = getattr(face, 'embedding', None)
                    if emb_raw is None:
                        continue
                    emb = emb_raw / (np.linalg.norm(emb_raw) + 1e-8)

                    best_ci, best_sim = -1, 0.0
                    for ci, cl in enumerate(clusters):
                        s = _cosine(emb, cl['emb'])
                        if s > best_sim:
                            best_sim, best_ci = s, ci

                    if best_sim >= threshold:
                        cl = clusters[best_ci]
                        cl['count'] += 1
                        # running average 임베딩
                        cl['emb'] = cl['emb'] * 0.9 + emb * 0.1
                        cl['emb'] /= (np.linalg.norm(cl['emb']) + 1e-8)
                        bbox = [int(v) for v in face.bbox.tolist()]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > cl['best_area']:
                            cl['best_area'] = area
                            cl['best_frame'] = frame.copy()
                            cl['best_bbox'] = bbox
                    else:
                        bbox = [int(v) for v in face.bbox.tolist()]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        clusters.append({
                            'count': 1,
                            'emb': emb.copy(),
                            'best_frame': frame.copy(),
                            'best_bbox': bbox,
                            'best_area': area,
                        })
                fidx += 1
        finally:
            cap.release()

    if not clusters:
        return []

    clusters.sort(key=lambda c: c['count'], reverse=True)
    top = clusters[:n]
    max_count = top[0]['count']
    dominant_thr = max_count * 0.6

    tmp_dir = Path(tempfile.mkdtemp(prefix='tve_faces_'))
    results = []
    for i, cl in enumerate(top):
        frame = cl['best_frame']
        x1, y1, x2, y2 = cl['best_bbox']
        h, w = frame.shape[:2]
        pad = int(max(x2 - x1, y2 - y1) * 0.35)
        crop = frame[max(0, y1-pad):min(h, y2+pad),
                     max(0, x1-pad):min(w, x2+pad)]
        if crop.size == 0:
            crop = frame
        face_img = cv2.resize(crop, (256, 256))
        img_path = str(tmp_dir / f'face_{i:02d}.jpg')
        cv2.imwrite(img_path, face_img)

        results.append({
            'count':      cl['count'],
            'embedding':  cl['emb'],
            'image_path': img_path,
            'dominant':   cl['count'] >= dominant_thr,
        })

    return results
