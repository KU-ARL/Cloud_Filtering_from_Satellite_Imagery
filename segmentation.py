from typing import Optional

import cv2
import numpy as np


# =========================================================
# 세그멘테이션 모듈
#
# Threshold → Edge → Prior → Combine
# Prior 적용 여부 / 파라미터 실험을 위해 독립 설계
# =========================================================


def apply_threshold(
    equalized: np.ndarray,
    value: int = 0,
    method: str = "otsu",
    adaptive_block: int = 51,
    adaptive_c: int = -10,
) -> np.ndarray:
    # =====================================================
    # Threshold
    #
    # method="otsu"     : 전역 자동 임계값 (value=0 시 Otsu)
    # method="adaptive" : 국소 영역 기준 임계값
    #                     → 어두운 배경 위 회색 구름도 검출 가능
    # method="both"     : 두 결과 OR → 상호 보완
    #
    # adaptive_block : 국소 영역 크기 (홀수)
    # adaptive_c     : 음수일수록 local_mean보다 밝은 픽셀만 통과
    # =====================================================
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            equalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            adaptive_c,
        )

    if method == "both":
        otsu     = apply_threshold(equalized, value, "otsu")
        adaptive = apply_threshold(equalized, 0, "adaptive", adaptive_block, adaptive_c)
        return cv2.bitwise_or(otsu, adaptive)

    # method="otsu" 또는 고정값
    flags = cv2.THRESH_BINARY
    if value == 0:
        flags |= cv2.THRESH_OTSU
    _, thresh = cv2.threshold(equalized, value, 255, flags)
    return thresh


def detect_edges(
    equalized: np.ndarray,
    low: int = 100,
    high: int = 200,
) -> np.ndarray:
    # =====================================================
    # Canny Edge Detection
    #
    # Gaussian → Sobel → NMS → Double Threshold → Hysteresis
    # Bilateral 전처리 덕분에 더 깨끗한 경계선 검출 가능
    # =====================================================
    return cv2.Canny(equalized, low, high)


def rgb_prior(
    img: np.ndarray,
    brightness_min: int = 150,
    uniformity_max: int = 20,
) -> np.ndarray:
    # =====================================================
    # RGB Prior
    #
    # "밝고 + 세 채널이 비슷하다 = 흰색 계열 = 구름 후보"
    # brightness_min : 어두운 무채색(그림자 등) 제거
    # uniformity_max : R≈G≈B 조건 (채널 간 차이 허용 범위)
    # =====================================================
    b, g, r = cv2.split(img)

    brightness = (r.astype(int) + g.astype(int) + b.astype(int)) / 3

    condition = (
        (brightness > brightness_min) &
        (np.abs(r.astype(int) - g.astype(int)) < uniformity_max) &
        (np.abs(r.astype(int) - b.astype(int)) < uniformity_max) &
        (np.abs(g.astype(int) - b.astype(int)) < uniformity_max)
    )

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[condition] = 255
    return mask


def hsv_prior(
    img: np.ndarray,
    s_max: int = 60,
    v_min: int = 180,
) -> np.ndarray:
    # =====================================================
    # HSV Prior
    #
    # "채도 낮고(무채색) + 밝다 = 흰색 = 구름 후보"
    # S 하나로 유채색/무채색 판단 → RGB uniformity보다 직관적
    # =====================================================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    condition = (s_ch < s_max) & (v_ch > v_min)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[condition] = 255
    return mask


def region_grow(
    seed_mask: np.ndarray,
    candidate_mask: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    # =====================================================
    # Region Growing
    #
    # seed_mask     : 엄격한 조건으로 확정된 구름 픽셀
    # candidate_mask: 완화된 조건을 통과한 후보 픽셀
    #
    # seed 주변 candidate 픽셀을 반복적으로 흡수
    # → 밝기가 낮아 prior를 못 통과한 구름 경계 복원
    # =====================================================
    kernel  = np.ones((3, 3), np.uint8)
    current = seed_mask.copy()

    for _ in range(iterations):
        dilated = cv2.dilate(current, kernel, iterations=1)
        current = cv2.bitwise_and(dilated, candidate_mask)
        current = cv2.bitwise_or(current, seed_mask)

    return current


def combine_masks(
    thresh: np.ndarray,
    edges: np.ndarray,
    rgb_mask: Optional[np.ndarray] = None,
    hsv_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # =====================================================
    # Mask 조합
    #
    # (Threshold OR Edges) AND Prior
    #
    # Prior를 마지막에 AND 적용:
    # → Edge도 반드시 색상 조건을 통과해야 함
    # → 해안선/육지 경계가 Edge로 오검출되는 문제 방지
    # =====================================================
    combined = cv2.bitwise_or(thresh, edges)

    if rgb_mask is not None:
        combined = cv2.bitwise_and(combined, rgb_mask)
    if hsv_mask is not None:
        combined = cv2.bitwise_and(combined, hsv_mask)

    return combined
