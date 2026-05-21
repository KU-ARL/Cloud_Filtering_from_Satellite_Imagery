# Cloud Filtering from Satellite Imagery

GK2A (천리안2A) 가시광선 RGB 위성 이미지에서 구름을 검출하고, CTT NetCDF Ground Truth와 IoU로 성능을 평가하는 파이프라인.

> 핵심 탐구: RGB Prior(도메인 지식)를 적용했을 때 구름 검출 성능이 향상되는가?

---

## Pipeline

```
RGB PNG
  │
  ├─ [Step 1] 헤더 크롭 (상단 타이틀 바 제거)
  ├─ [Step 2] Bilateral Filter          ← 노이즈 제거 + 경계 보존
  ├─ [Step 3] Grayscale + CLAHE         ← 국소 대비 강화
  │
  ├─ [Step 4] Otsu Threshold            ← 구름 내부 영역
  ├─ [Step 5] Canny Edge Detection      ← 구름 경계선
  │
  ├─ [Step 6] RGB Prior (밝기 + 채널 균일성)  ─┐
  ├─ [Step 7] HSV Prior (채도 낮고 + 밝음)     ├─ (Threshold OR Edges) AND Prior
  │                                          ─┘
  ├─ [Step 8] Region Growing            ← 완화된 조건으로 중하층 구름 경계 확장
  ├─ [Step 9] Morphology (Open → Close) ← 노이즈 제거 + 구멍 메우기
  ├─ [Step 10] Connected Component      ← 소형 blob 제거
  │
  └─ IoU 평가 (vs CLD NetCDF Ground Truth)
```

---

## 코드 구조

```
├── src/                        # 입력 데이터 (직접 추가 필요)
│   ├── gk2a_ami_le1b_rgb-s-true_*.png   # 가시광선 RGB 입력
│   └── gk2a_ami_le2_cld_*.nc            # CTT Ground Truth
│
├── preprocessing.py            # bilateral_filter / to_grayscale / histogram_equalize
├── segmentation.py             # apply_threshold / detect_edges / rgb_prior / hsv_prior / combine_masks / region_grow
├── postprocessing.py           # morphological_clean / filter_components
├── evaluation.py               # load_ground_truth / align_mask / calculate_iou
├── pipeline.py                 # DEFAULT_CONFIG + run_pipeline()
├── experiment.py               # Prior 조합별 IoU 비교 실험
└── main.py                     # 진입점
```

---

## 실행

```bash
pip install -r requirements.txt
python main.py
```

Prior 조합 비교 실험 (no_prior / rgb_only / hsv_only / both + 파라미터 변형):

```python
from experiment import compare_configs
compare_configs(rgb_path, nc_path)
```

---

## 파라미터 조정

`pipeline.py`의 `DEFAULT_CONFIG`에서 변경:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `threshold.value` | `0` (Otsu) | 고정값 사용 시 숫자 입력 |
| `hsv_prior.s_max` | `60` | 낮출수록 더 엄격한 무채색 조건 |
| `hsv_prior.v_min` | `180` | 높일수록 더 밝은 구름만 검출 |
| `hsv_relaxed.v_min` | `140` | Region Growing 확장 범위 |
| `region_grow.iterations` | `3` | 확장 반복 횟수 |
