"""
debug_segmentation.py

특정 프레임의 중간 과정 시각화 + 선택 전략 비교.

Usage:
  python debug_segmentation.py                       # MAYSAK frame1
  python debug_segmentation.py MAYSAK 7              # MAYSAK frame7
  python debug_segmentation.py HAISHEN 3
  python debug_segmentation.py MAYSAK 7 160          # v_min=160 고정

출력 이미지 (debug_out/):
  <TYPHOON>_frame<N>_strategy.png  ← 2행 3열:
    row1: 원본 / binary mask / closing 후 mask
    row2: area 선택 / elongation 선택 / y³-weighted 선택   ← 세 전략 비교
  <TYPHOON>_frame<N>_v<M>.png      ← v_min 지정 시 상세 4열 시각화

마커 색상:
  초록(큰 원) = area 최대
  파랑(큰 원) = elongation 최소
  빨강(큰 원) = y³-weighted centroid  ← 새 전략
"""

import os
import sys
import copy
import cv2
import numpy as np

from pipeline import run_pipeline
from typhoon_track import TYPHOON_CONFIG

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TYPHOON_BASE = os.path.join(BASE_DIR, "typhoon")
OUT_DIR      = os.path.join(BASE_DIR, "debug_out")
os.makedirs(OUT_DIR, exist_ok=True)

typhoon_name = sys.argv[1] if len(sys.argv) > 1 else "MAYSAK"
frame_idx    = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
single_vmin  = int(sys.argv[3]) if len(sys.argv) > 3 else None

src_dir = os.path.join(TYPHOON_BASE, typhoon_name, "src")
pngs = sorted([
    f for f in os.listdir(src_dir)
    if f.endswith(".png") and f not in ("path.png", "path_crop.png")
])
if frame_idx >= len(pngs):
    print(f"프레임 인덱스 {frame_idx+1} 초과 (총 {len(pngs)}개)")
    sys.exit(1)

fname    = pngs[frame_idx]
img_path = os.path.join(src_dir, fname)
orig     = cv2.imread(img_path)[42:, :]
H, W     = orig.shape[:2]
print(f"\n처리 중: {fname}  ({W}×{H})")


# ─── 도우미 ──────────────────────────────────────────────────

def elongation(comp_mask):
    M = cv2.moments(comp_mask)
    if M['m00'] == 0:
        return 999.0
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    d  = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    l2 = (mu20 + mu02 - d) / 2
    l1 = (mu20 + mu02 + d) / 2
    return l1 / l2 if l2 > 0 else 999.0


def do_close(mask_255, k=15, it=2):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel, iterations=it)


def analyze(merged, label):
    img_area = merged.shape[0] * merged.shape[1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(merged)
    h = merged.shape[0]
    print(f"\n  [{label}]  총 {n-1}개 component")

    area_idx   = None if n <= 1 else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    elong_idx, best_e = None, float('inf')

    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < img_area * 0.005:
            continue
        e  = elongation((labels == i).astype(np.uint8))
        cx, cy = centroids[i]
        print(f"    #{i:2d}: area={a:7d}  e={e:6.2f}  cx={cx:.0f}  cy={cy:.0f}")
        if a >= img_area * 0.01 and e < best_e:
            best_e, elong_idx = e, i

    # y³-weighted centroid (가장 큰 component 내에서)
    yw_cx, yw_cy = None, None
    if area_idx is not None:
        ys, xs = np.where(labels == area_idx)
        if len(ys):
            w = (ys.astype(np.float32) / h) ** 3
            tw = w.sum()
            if tw > 1e-6:
                yw_cx = float(np.dot(xs.astype(np.float32), w) / tw)
                yw_cy = float(np.dot(ys.astype(np.float32), w) / tw)

    if area_idx:
        ax, ay = centroids[area_idx]
        print(f"    → area    : #{area_idx}  cx={ax:.0f}  cy={ay:.0f}")
    if elong_idx:
        ex, ey = centroids[elong_idx]
        print(f"    → elongation: #{elong_idx}  e={best_e:.2f}  cx={ex:.0f}  cy={ey:.0f}")
    if yw_cx is not None:
        print(f"    → y³-weight : cx={yw_cx:.0f}  cy={yw_cy:.0f}")

    return n, labels, stats, centroids, area_idx, elong_idx, yw_cx, yw_cy


def draw_marker(img, cx, cy, color, radius=14, label=""):
    if cx is None:
        return img
    out = img.copy()
    pt = (int(cx), int(cy))
    cv2.circle(out, pt, radius, color, 3)
    cv2.circle(out, pt, 4, color, -1)
    if label:
        cv2.putText(out, label, (pt[0]+16, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (pt[0]+16, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def make_comp_vis(orig_img, labels, stats, centroids, img_area,
                  area_idx, elong_idx, yw_cx, yw_cy):
    COLORS = [
        (255,80,80),(80,180,255),(80,255,120),(255,210,0),
        (200,0,255),(0,220,180),(255,130,0),(120,120,255),
    ]
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for i in range(1, labels.max()+1):
        vis[labels == i] = COLORS[(i-1) % len(COLORS)]
    # 작은 label은 무시
    for i in range(1, labels.max()+1):
        if stats[i, cv2.CC_STAT_AREA] < img_area * 0.005:
            continue
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        e = elongation((labels == i).astype(np.uint8))
        txt = f"#{i} e={e:.1f} a={stats[i,cv2.CC_STAT_AREA]//1000}k"
        cv2.putText(vis, txt, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255,255,255), 1, cv2.LINE_AA)
    mk = (vis.sum(2) > 0).astype(np.uint8)[:,:,None]
    out = orig_img.astype(np.float32)
    out = out*(1-mk*0.45) + vis.astype(np.float32)*mk*0.55
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 마커
    if area_idx is not None:
        out = draw_marker(out, centroids[area_idx][0], centroids[area_idx][1],
                          (0,200,0), label="area")
    if elong_idx is not None:
        out = draw_marker(out, centroids[elong_idx][0], centroids[elong_idx][1],
                          (255,80,0), label="elong")
    if yw_cx is not None:
        out = draw_marker(out, yw_cx, yw_cy, (0,0,255), label="y3w")
    return out


def add_title(img, text, fs=0.55):
    out = img.copy()
    cv2.rectangle(out, (0,0), (img.shape[1], 24), (20,20,20), -1)
    cv2.putText(out, text, (5,17), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (220,220,220), 1, cv2.LINE_AA)
    return out


def config_with_vmin(v):
    cfg = copy.deepcopy(TYPHOON_CONFIG)
    cfg["hsv_prior"]["v_min"]   = v
    cfg["hsv_relaxed"]["v_min"] = max(40, v - 30)
    return cfg


# ─── 메인 ────────────────────────────────────────────────────
vmin = single_vmin if single_vmin is not None else TYPHOON_CONFIG["hsv_prior"]["v_min"]
print(f"v_min = {vmin}")

cfg          = config_with_vmin(vmin)
binary_mask, _ = run_pipeline(img_path, cfg)
mask_255     = (binary_mask > 0).astype(np.uint8) * 255
img_area     = H * W
closed       = do_close(mask_255)

n, labels, stats, centroids, ai, ei, yw_cx, yw_cy = analyze(closed, f"15×15×2  v={vmin}")

# ─── 시각화 ──────────────────────────────────────────────────
comp_vis = make_comp_vis(orig, labels, stats, centroids, img_area,
                          ai, ei, yw_cx, yw_cy)

# 마커만 원본에 오버레이한 버전도 추가
orig_marked = orig.copy()
if ai is not None:
    orig_marked = draw_marker(orig_marked, centroids[ai][0], centroids[ai][1],
                               (0,200,0), label="area")
if ei is not None:
    orig_marked = draw_marker(orig_marked, centroids[ei][0], centroids[ei][1],
                               (255,80,0), label="elong")
if yw_cx is not None:
    orig_marked = draw_marker(orig_marked, yw_cx, yw_cy, (0,0,255), label="y3w")

row1 = np.hstack([
    add_title(orig,                                              "1. 원본"),
    add_title(cv2.cvtColor(mask_255, cv2.COLOR_GRAY2BGR),       f"2. Binary Mask  v_min={vmin}"),
    add_title(cv2.cvtColor(closed,   cv2.COLOR_GRAY2BGR),        "3. Closed 15×15×2"),
])
row2 = np.hstack([
    add_title(comp_vis,    "4. Components  [초록=area / 주황=elong / 빨강=y³w]"),
    add_title(orig_marked, "5. 원본 위 마커 비교  [초록=area / 주황=elong / 빨강=y³w]"),
])

# 행 너비 맞추기
mw = min(row1.shape[1], row2.shape[1]*2//2)
row1_r = cv2.resize(row1, (row2.shape[1], row1.shape[0]))

grid = np.vstack([row1_r, row2])

suffix = f"v{vmin}" if single_vmin else "strategy"
out_path = os.path.join(OUT_DIR, f"debug_{typhoon_name}_frame{frame_idx+1:02d}_{suffix}.png")
cv2.imwrite(out_path, grid)
print(f"\n저장: {out_path}  ({grid.shape[1]}×{grid.shape[0]})")
print("마커: 초록=area 최대  /  주황=elongation 최소  /  빨강=y³-weighted  ← 새 전략")
