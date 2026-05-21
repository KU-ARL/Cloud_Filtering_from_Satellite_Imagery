# Cloud Filtering from Satellite Imagery

GK2A (천리안2A) 가시광선 RGB 위성 이미지에서 구름을 검출하고, CTT NetCDF Ground Truth와 비교해 성능을 평가하는 파이프라인.

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
  ├─ [Step 4] Threshold (Otsu / Adaptive / 고정값)  ← 구름 내부 영역
  ├─ [Step 5] Canny Edge Detection                  ← 구름 경계선
  │
  ├─ [Step 6] RGB Prior (밝기 + 채널 균일성)  ─┐
  ├─ [Step 7] HSV Prior (채도 낮고 + 밝음)     ├─ (Threshold OR Edges) AND Prior
  │                                          ─┘
  ├─ [Step 8] Region Growing            ← 완화된 조건으로 중하층 구름 경계 확장
  ├─ [Step 9] Morphology (Open → Close) ← 노이즈 제거 + 구멍 메우기
  ├─ [Step 10] Connected Component      ← 소형 blob 제거
  │
  └─ 성능 평가 (IoU / Precision / Recall / F1)
```

---

## 코드 구조

```
├── src/                        # 입력 데이터 (직접 추가 필요)
│   ├── gk2a_ami_le1b_rgb-s-true_*.png   # 가시광선 RGB 입력
│   ├── gk2a_ami_le2_cld_*.nc            # CTT Ground Truth
│   └── result/                          # 실험 결과 이미지 저장 경로
│
├── preprocessing.py            # bilateral_filter / to_grayscale / histogram_equalize
├── segmentation.py             # apply_threshold / detect_edges / rgb_prior / hsv_prior / combine_masks / region_grow
├── postprocessing.py           # morphological_clean / filter_components
├── evaluation.py               # load_ground_truth / calculate_metrics / save_ground_truth_image
├── pipeline.py                 # DEFAULT_CONFIG + run_pipeline()
├── experiment.py               # 실험 조합 정의 및 비교 실행
└── main.py                     # 진입점
```

---

## 실행

```bash
pip install -r requirements.txt

# 단일 실행 (DEFAULT_CONFIG 기준)
python main.py

# 실험 비교 실행 (결과 이미지 src/result/ 에 저장)
python experiment.py
```

---

## 평가 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| **IoU** | TP / (TP+FP+FN) | 예측과 정답이 얼마나 겹치는가 |
| **Precision** | TP / (TP+FP) | 구름이라고 한 것 중 실제 구름 비율 |
| **Recall** | TP / (TP+FN) | 실제 구름 중 검출한 비율 |
| **F1** | 조화평균(Prec, Recall) | Precision과 Recall의 균형 |

Ground Truth는 적외선(CTT) 기반이므로, 가시광선 RGB로 식별 불가능한 권운·박운도 정답에 포함된다. 이 구조적 차이가 Precision/Recall 분리 분석을 필요로 한다.

---

## 실험 설계

`python experiment.py` 실행 시 아래 6개 조합을 순서대로 비교한다:

| # | 실험 | 증명하는 것 |
|---|------|------------|
| 1 | `baseline (no prior, no rg)` | 아무 기법 없이 Otsu+Canny만 사용한 기준선 |
| 2 | `both_prior (no region_grow)` | Prior가 Precision을 극대화함 (Prec ≈ 0.998) |
| 3 | `both_prior + region_grow` | Region Growing이 Recall을 회복시킴 |
| 4 | `best_combo (both_thresh + v>90)` | Prior 방법 중 F1 최고 조합 |
| 5 | `low_thresh (t=30) + both_prior` | Threshold를 낮춰도 Prior가 있으면 결과 동일 → Prior가 실질적 필터임을 증명 |
| 6 | `low_thresh (t=30) + no_prior` | Prior 제거 시 Recall 최대(≈0.975)지만 Precision 하락 → Precision-Recall 트레이드오프 |

---

## 파라미터 조정

`pipeline.py`의 `DEFAULT_CONFIG`에서 변경:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `threshold.method` | `"otsu"` | `"otsu"` / `"adaptive"` / `"both"` |
| `threshold.value` | `0` (Otsu) | 고정값 사용 시 숫자 입력 |
| `hsv_prior.s_max` | `60` | 낮출수록 더 엄격한 무채색 조건 |
| `hsv_prior.v_min` | `150` | 낮출수록 더 어두운 구름까지 검출 |
| `hsv_relaxed.v_min` | `120` | Region Growing 확장 범위 |
| `region_grow.iterations` | `3` | 확장 반복 횟수 |
