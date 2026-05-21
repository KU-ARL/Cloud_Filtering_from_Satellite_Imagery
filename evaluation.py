import cv2
import numpy as np
from netCDF4 import Dataset  # type: ignore


# =========================================================
# 평가 모듈
#
# KMA NetCDF (CTT) Ground Truth 로드 + IoU 계산
# RGB PNG와 NetCDF 해상도 불일치 → align_mask로 자동 해결
# =========================================================


def load_ground_truth(nc_path: str) -> np.ndarray:
    # =====================================================
    # Ground Truth 로드
    #
    # GK2A CLD 변수: 0,1 → cloud / 2 → clear
    # CTT 저온 영역(-40°C 이하) = 높은 구름 = 정답 마스크
    # =====================================================
    nc = Dataset(nc_path)
    cld = nc.variables['CLD'][:]

    ground_truth = np.where(cld <= 1, 1, 0).astype(np.uint8)
    return ground_truth


def align_mask(
    my_mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    # =====================================================
    # Mask 해상도 정렬
    #
    # RGB PNG와 NetCDF CLD 그리드의 해상도 불일치 해결
    # INTER_NEAREST: binary mask 픽셀값(0/1) 보존
    # =====================================================
    h, w = target_shape
    resized = cv2.resize(
        (my_mask > 0).astype(np.uint8) * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def save_ground_truth_image(
    nc_path: str,
    save_path: str,
) -> None:
    # =====================================================
    # Ground Truth 시각화 이미지 저장
    #
    # 흰색  : 구름 (CLD <= 1)
    # 검정  : 맑음 (CLD == 2)
    # 회색  : 결측/마스크 (그 외)
    # =====================================================
    ground_truth = load_ground_truth(nc_path)
    raw = Dataset(nc_path).variables['CLD'][:]

    h, w = ground_truth.shape
    img  = np.zeros((h, w, 3), dtype=np.uint8)

    img[raw == 2]            = [30,  30,  30]   # 맑음 → 검정
    img[ground_truth == 1]   = [255, 255, 255]  # 구름 → 흰색
    img[np.ma.getmaskarray(raw)] = [100, 100, 100]  # 결측 → 회색

    cv2.imwrite(save_path, img)


def calculate_metrics(
    ground_truth: np.ndarray,
    my_mask: np.ndarray,
) -> dict[str, float]:
    # =====================================================
    # 통합 성능 지표 계산
    #
    # TP : 우리도 구름, GT도 구름  (맞게 잡음)
    # FP : 우리는 구름, GT는 맑음  (잘못 잡음)
    # FN : 우리는 맑음, GT는 구름  (놓침)
    #
    # IoU       = TP / (TP + FP + FN)
    # Precision = TP / (TP + FP)  → 잡은 것 중 맞은 비율
    # Recall    = TP / (TP + FN)  → 실제 구름 중 잡은 비율
    # F1        = 조화평균(Precision, Recall)
    # =====================================================
    pred = (my_mask > 0).astype(np.uint8)

    if ground_truth.shape != pred.shape:
        pred = align_mask(pred, ground_truth.shape)

    gt   = ground_truth.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt,  pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt,  ~pred).sum()

    iou       = tp / (tp + fp + fn)         if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)              if (tp + fp)      > 0 else 0.0
    recall    = tp / (tp + fn)              if (tp + fn)      > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall)  > 0 else 0.0

    return {
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def calculate_iou(
    ground_truth: np.ndarray,
    my_mask: np.ndarray,
) -> float:
    # calculate_metrics의 IoU만 반환 (하위 호환)
    return calculate_metrics(ground_truth, my_mask)["iou"]
