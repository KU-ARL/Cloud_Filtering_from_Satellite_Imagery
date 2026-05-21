import cv2
import numpy as np

from preprocessing  import bilateral_filter, to_grayscale, histogram_equalize
from segmentation   import apply_threshold, detect_edges, rgb_prior, hsv_prior, combine_masks, region_grow
from postprocessing import morphological_clean, filter_components


# =========================================================
# 기본 설정값
# experiment.py에서 값을 변경하여 성능 비교 가능
# =========================================================
DEFAULT_CONFIG: dict = {
    "bilateral":   {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0},
    "clahe":       {"clip_limit": 2.0, "tile_grid": (8, 8)},
    "threshold":   {"value": 0, "method": "otsu", "adaptive_block": 51, "adaptive_c": -10},
    "canny":       {"low": 100, "high": 200},
    "rgb_prior":   {"enabled": True, "brightness_min": 150, "uniformity_max": 20},
    # v_min 실험 결과: 180→150 완화 시 IoU 0.44→0.55 최대 향상폭
    "hsv_prior":   {"enabled": True, "s_max": 60,  "v_min": 150},
    "hsv_relaxed": {"enabled": True, "s_max": 80,  "v_min": 120},
    "region_grow": {"enabled": True, "iterations": 3},
    "morphology":  {"kernel_size": 5, "open_iter": 2, "close_iter": 3},
    "min_area":    500,
}


def run_pipeline(
    image_path: str,
    config: dict = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    # =====================================================
    # 0. 이미지 로드
    # =====================================================
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # =====================================================
    # 1. 헤더 크롭
    # 상단 타이틀 바 제거 → Otsu 히스토그램 왜곡 방지
    # =====================================================
    img = img[42:, :]
    original = img.copy()

    # =====================================================
    # 2. 전처리
    # Bilateral → Grayscale → CLAHE
    # denoised를 Prior에도 사용: 노이즈가 RGB 균일성 조건을 방해하지 않도록
    # =====================================================
    denoised  = bilateral_filter(img, **config["bilateral"])
    gray      = to_grayscale(denoised)
    equalized = histogram_equalize(gray, **config["clahe"])

    # =====================================================
    # 2. 세그멘테이션
    # Threshold + Edge 검출 후 Prior로 필터링
    # =====================================================
    thresh = apply_threshold(equalized, **config["threshold"])
    edges  = detect_edges(equalized, **config["canny"])

    rgb_cfg  = config["rgb_prior"]
    rgb_mask = (
        rgb_prior(denoised, brightness_min=rgb_cfg["brightness_min"], uniformity_max=rgb_cfg["uniformity_max"])
        if rgb_cfg["enabled"] else None
    )

    hsv_cfg  = config["hsv_prior"]
    hsv_mask = (
        hsv_prior(denoised, s_max=hsv_cfg["s_max"], v_min=hsv_cfg["v_min"])
        if hsv_cfg["enabled"] else None
    )

    combined = combine_masks(thresh, edges, rgb_mask, hsv_mask)

    # =====================================================
    # 2-2. Region Growing (선택적)
    # 엄격한 prior로 잡은 seed에서 완화된 조건 픽셀로 확장
    # → 밝기가 낮은 중하층 구름 경계 복원
    # =====================================================
    rg_cfg = config["region_grow"]
    if rg_cfg["enabled"] and config["hsv_relaxed"]["enabled"]:
        rx_cfg        = config["hsv_relaxed"]
        relaxed_mask  = hsv_prior(denoised, s_max=rx_cfg["s_max"], v_min=rx_cfg["v_min"])
        combined      = region_grow(combined, relaxed_mask, iterations=rg_cfg["iterations"])

    # =====================================================
    # 3. 후처리
    # Morphology → Connected Component 필터
    # =====================================================
    cleaned               = morphological_clean(combined, **config["morphology"])
    final_mask, centroids = filter_components(cleaned, config["min_area"])

    # =====================================================
    # 4. Binary Mask 변환
    # cloud = 1 / clear = 0
    # =====================================================
    binary_mask = (final_mask > 0).astype(np.uint8)

    # =====================================================
    # 5. Overlay 생성
    # 구름 영역 빨간색 반투명 표시
    # =====================================================
    overlay = original.copy()
    overlay[binary_mask == 1] = [0, 0, 255]
    result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)

    return binary_mask, result
