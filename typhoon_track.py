import os
import re
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from pipeline import run_pipeline, DEFAULT_CONFIG


# =========================================================
# 태풍 경로 추적 모듈
#
# 연속 위성 이미지에서 구름 분리 후
# 가장 큰 cloud segment 중심점을 태풍 위치로 추정
# 중심점 궤적을 path.png에 직접 오버레이해서 비교
# =========================================================


# ─── 파이프라인 설정 ──────────────────────────────────────
TYPHOON_CONFIG: dict = {
    **DEFAULT_CONFIG,
    "threshold":   {**DEFAULT_CONFIG["threshold"],   "method": "both", "adaptive_block": 51, "adaptive_c": -10},
    "hsv_prior":   {**DEFAULT_CONFIG["hsv_prior"],   "v_min": 90},
    "hsv_relaxed": {**DEFAULT_CONFIG["hsv_relaxed"], "v_min": 60},
}



# ─── 중심점 추출 ──────────────────────────────────────────

def find_largest_centroid(binary_mask: np.ndarray) -> Optional[tuple[float, float]]:
    # =====================================================
    # 가장 큰 cloud component 내에서 y³-가중 centroid 반환
    #
    # 문제: 태풍 + 전선 구름대가 하나의 거대 blob으로 합쳐지면
    #       단순 centroid는 픽셀 수가 많은 전선 쪽(북쪽, 낮은 y)으로 당겨짐
    #
    # 해결: 이미지 하단(남쪽 = 태풍 방향)일수록 가중치를 높여
    #       centroid를 태풍 위치(blob 남쪽 밀집 부분)로 끌어당김
    #       weight = (y / H)³  →  상단 픽셀 대비 하단 픽셀 약 30× 강조
    # =====================================================
    H = binary_mask.shape[0]
    mask_255 = (binary_mask > 0).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    merged = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged)
    if num_labels <= 1:
        return None

    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # 가장 큰 component의 픽셀 좌표 추출
    ys, xs = np.where(labels == largest_idx)
    if len(ys) == 0:
        return float(centroids[largest_idx][0]), float(centroids[largest_idx][1])

    # y³ 가중치: 이미지 하단(태풍 위치)을 강조
    weights = (ys.astype(np.float32) / H) ** 3
    total_w = weights.sum()
    if total_w < 1e-6:
        return float(centroids[largest_idx][0]), float(centroids[largest_idx][1])

    cx = float(np.dot(xs.astype(np.float32), weights) / total_w)
    cy = float(np.dot(ys.astype(np.float32), weights) / total_w)
    return cx, cy


# ─── 유틸리티 ────────────────────────────────────────────

def parse_timestamp(filename: str) -> str:
    match = re.search(r'_(\d{12})\.png$', filename)
    return match.group(1) if match else filename


def format_time_label(timestamp: str) -> str:
    if len(timestamp) != 12:
        return timestamp
    return f"{timestamp[4:6]}/{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}"


def _result_subdir(base_result_dir: str) -> str:
    subdir = os.path.join(base_result_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(subdir, exist_ok=True)
    return subdir


# ─── 추적 ────────────────────────────────────────────────

def track_typhoon(src_dir: str, config: dict = TYPHOON_CONFIG) -> list[dict]:
    # =====================================================
    # src_dir 내 위성 PNG를 시간순 처리 (path.png 제외)
    # 각 프레임에서 가장 큰 cloud segment 중심점 추출
    # =====================================================
    pngs = sorted([
        f for f in os.listdir(src_dir)
        if f.endswith('.png') and f not in ('path.png', 'path_crop.png')
    ])

    print(f"\n{'파일명':<50} {'중심 X':>8} {'중심 Y':>8}")
    print("-" * 70)

    results = []
    for fname in pngs:
        img_path  = os.path.join(src_dir, fname)
        timestamp = parse_timestamp(fname)

        binary_mask, _ = run_pipeline(img_path, config)
        center = find_largest_centroid(binary_mask)

        if center:
            results.append({
                "timestamp": timestamp,
                "cx": center[0],
                "cy": center[1],
                "filename": fname,
            })
            print(f"{fname:<50} {center[0]:>8.1f} {center[1]:>8.1f}")
        else:
            print(f"{fname:<50} {'(미검출)':>17}")

    return results


# ─── path_crop.png 로드 ───────────────────────────────────

def _load_path_crop(path_crop_png: str, target_size: tuple[int, int]) -> Optional[np.ndarray]:
    # 사용자가 직접 크롭한 path_crop.png를 위성 이미지 크기로 리사이즈
    img = cv2.imread(path_crop_png)
    if img is None:
        return None
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


# ─── 위성 이미지 위 궤적 그리기 ──────────────────────────

def _draw_track_on_satellite(
    frame: np.ndarray,
    track: list[dict],
    up_to_idx: int,
) -> np.ndarray:
    out    = frame.copy()
    points = [(int(t["cx"]), int(t["cy"])) for t in track[:up_to_idx + 1]]

    for i, pt in enumerate(points):
        if i > 0:
            cv2.line(out, points[i - 1], pt, (0, 255, 0), 2, cv2.LINE_AA)
        if i == up_to_idx:
            cv2.circle(out, pt, 10, (0, 0, 255), -1)
            cv2.circle(out, pt, 10, (255, 255, 255), 2)
        else:
            cv2.circle(out, pt, 5, (0, 220, 0), -1)
            cv2.circle(out, pt, 5, (255, 255, 255), 1)

    label = format_time_label(track[up_to_idx]["timestamp"])
    cv2.putText(out, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ─── path.png 위 궤적 오버레이 ───────────────────────────

def _draw_track_on_path(
    path_canvas: np.ndarray,
    track: list[dict],
) -> np.ndarray:
    # =====================================================
    # 크롭+리사이즈된 path.png 위에 궤적 오버레이
    #
    # 두 이미지가 동일한 지리 범위를 동일한 크기로 나타내므로
    # centroid (cx, cy) 픽셀 좌표를 그대로 사용
    # =====================================================
    out    = path_canvas.copy()
    points = [(int(t["cx"]), int(t["cy"])) for t in track]
    n      = len(points)

    # 연결선
    for i in range(1, n):
        cv2.line(out, points[i - 1], points[i], (0, 230, 0), 3, cv2.LINE_AA)

    # 점 + 번호
    for i, pt in enumerate(points):
        ratio = i / max(n - 1, 1)
        green = int(130 + 125 * ratio)
        cv2.circle(out, pt, 9, (0, green, 0), -1)
        cv2.circle(out, pt, 9, (255, 255, 255), 2)

        label = str(i + 1)
        cv2.putText(out, label, (pt[0] + 11, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (pt[0] + 11, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 시작(주황) / 끝(빨강) 강조
    if points:
        cv2.circle(out, points[0],  12, (0,   140, 255), -1)
        cv2.circle(out, points[-1], 12, (0,   0,   255), -1)
        for pt in [points[0], points[-1]]:
            cv2.circle(out, pt, 12, (255, 255, 255), 2)

    # 범례
    h = out.shape[0]
    for text, color, y_off in [
        ("1  Start", (0, 140, 255), 55),
        (f"{n}  End",   (0,   0, 255), 30),
    ]:
        cv2.putText(out, text, (14, h - y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (14, h - y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color,    1, cv2.LINE_AA)

    return out


# ─── 전체 시각화 ─────────────────────────────────────────

def visualize_track(
    src_dir: str,
    track: list[dict],
    output_dir: str,
    path_crop_png: Optional[str] = None,
) -> None:
    if not track:
        return

    # 위성 이미지 크기 확인 (헤더 크롭 후)
    sample = cv2.imread(os.path.join(src_dir, track[0]["filename"]))
    if sample is None:
        return
    sample       = sample[42:, :]
    sat_h, sat_w = sample.shape[:2]

    # 1) 프레임별 누적 궤적
    for i, entry in enumerate(track):
        frame = cv2.imread(os.path.join(src_dir, entry["filename"]))
        if frame is None:
            continue
        frame = frame[42:, :]
        out   = _draw_track_on_satellite(frame, track, i)
        cv2.imwrite(os.path.join(output_dir, f"track_{i+1:02d}_{entry['timestamp']}.png"), out)

    # 2) 전체 궤적 (마지막 프레임 기반)
    last_frame = cv2.imread(os.path.join(src_dir, track[-1]["filename"]))
    if last_frame is not None:
        last_frame = last_frame[42:, :]
        final      = _draw_track_on_satellite(last_frame, track, len(track) - 1)
        cv2.imwrite(os.path.join(output_dir, "final_track.png"), final)

    # 3) path_crop.png 위에 궤적 오버레이
    if path_crop_png and os.path.exists(path_crop_png):
        path_canvas = _load_path_crop(path_crop_png, (sat_w, sat_h))
        if path_canvas is not None:
            comparison = _draw_track_on_path(path_canvas, track)
            cv2.imwrite(os.path.join(output_dir, "comparison.png"), comparison)


# ─── 진입점 ──────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    TYPHOON_BASE = os.path.join(BASE_DIR, "typhoon")

    # src/ 하위 디렉토리가 있는 폴더를 모두 태풍으로 인식 (이름순)
    # typhoon_names = sorted([
    #     d for d in os.listdir(TYPHOON_BASE)
    #     if os.path.isdir(os.path.join(TYPHOON_BASE, d, "src"))
    # ])
    typhoon_names = ["HINNAMNOR"]  # 임시: 힌남노만 실행

    for typhoon_name in typhoon_names:
        print(f"\n{'=' * 60}")
        print(f"  태풍 {typhoon_name} 경로 추적")
        print(f"{'=' * 60}")

        typhoon_dir = os.path.join(BASE_DIR, "typhoon", typhoon_name)
        src_dir       = os.path.join(typhoon_dir, "src")
        result_dir    = os.path.join(typhoon_dir, "result")
        path_crop_png = os.path.join(src_dir, "path_crop.png")
        output_dir    = _result_subdir(result_dir)

        track = track_typhoon(src_dir, TYPHOON_CONFIG)

        if track:
            visualize_track(src_dir, track, output_dir, path_crop_png)
            print(f"\n결과 저장 완료: {output_dir}/")
            print(f"  track_01 ~ track_{len(track):02d}_*.png")
            print(f"  final_track.png")
            print(f"  comparison.png  ← path_crop.png + 궤적 오버레이")
        else:
            print("추적 실패: 검출된 cloud 없음")
