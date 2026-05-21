import cv2
import numpy as np


# =========================================================
# 후처리 모듈
#
# Morphology → Connected Component 분석
# Centroid: 태풍 중심점 추적에 활용
# =========================================================


def morphological_clean(
    mask: np.ndarray,
    kernel_size: int = 5,
    open_iter: int = 2,
    close_iter: int = 3,
) -> np.ndarray:
    # =====================================================
    # Morphological Opening → Closing
    #
    # Opening  (Erosion→Dilation) : 작은 노이즈 blob 제거
    # Closing  (Dilation→Erosion) : 구름 내부 구멍 메우기
    #
    # close_iter > open_iter:
    # 구름 내부 구멍이 잔류 노이즈 점보다 크기 때문
    # =====================================================
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=open_iter)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return closed


def filter_components(
    mask: np.ndarray,
    min_area: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    # =====================================================
    # Connected Component 분석
    #
    # 일정 면적 이하 blob 제거 (잔류 노이즈 정리)
    # centroids: 각 구름 덩어리의 중심점
    #            → 태풍 경로 추적에 활용
    #
    # 반환값:
    #   filtered_mask : 면적 조건 통과한 구름만 표시한 mask
    #   centroids     : 각 blob의 (x, y) 중심점 배열
    # =====================================================
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    filtered_mask = np.zeros_like(mask)
    valid_centroids = []

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
            valid_centroids.append(tuple(centroids[i]))

    return filtered_mask, np.array(valid_centroids)
