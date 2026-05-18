import cv2
import numpy as np


# =========================================================
# 구름 추출 함수
#
# 입력:
#   image_path : RGB 위성영상 경로
#
# 출력:
#   refined_mask : 최종 구름 Binary Mask
#                  (cloud=1, clear=0)
#
#   result_image : Overlay 결과 영상
#
# 특징:
#   - 결과 창 출력 유지
#   - 성능 비교(IoU 등)에 바로 사용 가능
#   - 다른 파일에서 함수 하나만 호출하면 됨
# =========================================================
def extract_cloud(image_path, display_scale=0.2):

    # =====================================================
    # 내부 출력용 resize 함수
    # =====================================================
    def resize_for_display(img, scale=display_scale):

        h, w = img.shape[:2]

        return cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )

    # =====================================================
    # 1. 원본 이미지 불러오기
    # =====================================================
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    original = img.copy()

    # =====================================================
    # 2. Bilateral Filter
    # =====================================================
    bilateral = cv2.bilateralFilter(
        img,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # =====================================================
    # 3. GrayScale
    # =====================================================
    gray = cv2.cvtColor(
        bilateral,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # 4. Histogram Equalization
    # =====================================================
    equalized = cv2.equalizeHist(gray)

    # =====================================================
    # 5. Threshold
    # =====================================================
    _, thresh = cv2.threshold(
        equalized,
        180,
        255,
        cv2.THRESH_BINARY
    )

    # =====================================================
    # 6. Edge Detection
    # =====================================================
    edges = cv2.Canny(
        equalized,
        100,
        200
    )

    # =====================================================
    # 7. RGB Prior
    # =====================================================
    b, g, r = cv2.split(img)

    rgb_condition = (
        (np.abs(r.astype(int) - g.astype(int)) < 20) &
        (np.abs(r.astype(int) - b.astype(int)) < 20) &
        (np.abs(g.astype(int) - b.astype(int)) < 20)
    )

    rgb_mask = np.zeros_like(gray)
    rgb_mask[rgb_condition] = 255

    # =====================================================
    # 8. HSV Prior
    # =====================================================
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    hsv_condition = (
        (s < 60) &
        (v > 180)
    )

    hsv_mask = np.zeros_like(gray)
    hsv_mask[hsv_condition] = 255

    # =====================================================
    # 9. Mask Combination
    # =====================================================
    combined = cv2.bitwise_and(
        thresh,
        rgb_mask
    )

    combined = cv2.bitwise_and(
        combined,
        hsv_mask
    )

    combined = cv2.bitwise_or(
        combined,
        edges
    )

    # =====================================================
    # 10. Morphology
    # =====================================================
    kernel = np.ones((5, 5), np.uint8)

    opened = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel
    )

    refined = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================================
    # 11. Binary Mask 변환
    #
    # 성능 비교(IoU 등)를 위해:
    # cloud = 1
    # clear = 0
    # =====================================================
    refined_mask = (
        refined > 0
    ).astype(np.uint8)

    # =====================================================
    # 12. Overlay 생성
    # =====================================================
    overlay = original.copy()

    overlay[refined_mask == 1] = [0, 0, 255]

    result = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )

    # =====================================================
    # 13. 결과 출력
    # =====================================================
    cv2.imshow(
        "Original",
        original
    )

    cv2.imshow(
        "Threshold",
        thresh
    )

    cv2.imshow(
        "RGB Prior",
        rgb_mask
    )

    cv2.imshow(
        "HSV Prior",
        hsv_mask
    )

    cv2.imshow(
        "Cloud Extraction Result",
        result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # =====================================================
    # 14. Return
    # =====================================================
    return refined_mask, result