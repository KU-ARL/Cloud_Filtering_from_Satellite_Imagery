import cv2
import numpy as np


# =========================================================
# 전처리 모듈
#
# Bilateral Filter → Grayscale → CLAHE
# 각 단계는 독립적으로 호출 가능
# =========================================================


def bilateral_filter(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    # =====================================================
    # Bilateral Filter
    #
    # 가까이 있고 밝기가 비슷한 픽셀만 평균내는 필터
    # 거리 가중치 + 밝기 차이 가중치 동시 적용
    # → 구름 경계를 보존하면서 노이즈 제거
    # =====================================================
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    # =====================================================
    # Grayscale 변환
    #
    # RGB 3채널 → 밝기 정보 1채널
    # 이후 Threshold / Edge 연산 단순화
    # =====================================================
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram_equalize(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    # =====================================================
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #
    # 전역 HE와 달리 타일 단위로 적응형 적용
    # → 구름/바다/육지 밝기 분포를 국소적으로 벌려 경계 강조
    # clipLimit: 과포화 방지
    # =====================================================
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)
