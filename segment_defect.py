"""
Multi-region segmentation for metal part images - segment_defect.py

Produces a 4-class per-pixel label maps:
 0 background - light gray
 1 good_blue - clean blue-paited part surface
 2 defective - rust, scratches, or other surface damages
 3. wire - metal hook and suspension wire

"""

import cv2
import numpy as np
from pathlib import Path
import argparse


LABEL_BG = 0
LABEL_BLUE = 1
LABEL_DEFECT = 2
LABEL_WIRE = 3
N_CLASSES = 4

CLASS_NAMES = ["background", "good_blue", "defective", "wire"]

# BRG false-color palette (index= class id)
PALETTE = np.array([
    [25, 25, 25],  # 0 background
    [200, 80, 0],  # 1 good blue
    [0, 0, 220],   # 2 defective
    [180, 180, 180],  # 3 wire
], dtype=np.uint8)


def preprocess(img: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """bilateral filter + CLAHE. Resizes only when target_size is given."""
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b= cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


BG_SAT_MAX = 35
BG_VAL_MIN = 110

WIRE_SAT_MAX = 40
WIRE_VAL_MIN = 25


HOLE_AREA_FRAC = 0.004
RUST_RANGES = [
    (np.array([5, 50, 40]), np.array([25, 255, 230])),
    (np.array([25, 30, 30]), np.array([50, 200, 200]))
]

BLUE_H_LO, BLUE_H_HI = 88, 150
BLUE_S_MIN = 45
BLUE_V_MIN = 30

SMALL_REGION_FRAC = 0.0003
DEFECT_MERGE_RADIUS = 18
DEFECT_MIN_CORE_FRAC = 0.0004
DEFECT_THRESHOLD = 0.0035


def _part_masks(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)

    surface = (~((s < BG_SAT_MAX) & (v > BG_VAL_MIN))).astype(np.uint8) * 255
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, k9)
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, k9)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(surface, connectivity=8)
    if n > 1:
        idx = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        surface = (lbl == idx).astype(np.uint8) *225

    contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    silhouette = np.zeros_like(surface)
    if contours:
        cv2.drawContours(silhouette, contours, -1, 255, cv2.FILLED)

    return surface, silhouette


def _classify(hsv: np.ndarray, part_silhouette: np.ndarray) -> np.ndarray:
    """ Assign each pixel a class id """

    h, w = hsv.shape[:2]
    hue = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    out = np.full((h, w), -1, dtype=np.int8)
    in_part = part_silhouette > 0
    low_chroma = (s < WIRE_SAT_MAX) & (v >= WIRE_VAL_MIN)
    out[low_chroma & ~in_part] = LABEL_BG
    out[(v < WIRE_VAL_MIN) & ~in_part] = LABEL_BG

    inside_lc = (low_chroma & in_part).astype(np.uint8) *255
    if cv2.countNonZero(inside_lc) > 0:
        n_lc, lc_lbl, lc_stats, _ = cv2.connectedComponentsWithStats(inside_lc, connectivity=8)
        hole_thresh = h*w*HOLE_AREA_FRAC
        for i in range(1, n_lc):
            region = lc_lbl == i
            label = (LABEL_BG if lc_stats[i, cv2.CC_STAT_AREA] >= hole_thresh else LABEL_WIRE)
            out[region] = label

    defect = np.zeros((h, w), dtype=bool)
    for lo, hi in RUST_RANGES:
        defect |= (hue >= int(lo[0])) & (hue <= int(hi[0])) & (s >= int(lo[1])) & (s <= int(hi[1])) & (v >= int(lo[2])) & (v <= int(hi[2]))
    out[defect & (out < 0)] = LABEL_DEFECT

    blue = ((hue >= BLUE_H_LO) & (hue <= BLUE_H_HI) & (s >= BLUE_S_MIN) & (v >= BLUE_V_MIN) & (out < 0))
    out[blue] = LABEL_BLUE
    out[(out < 0) & ~in_part] = LABEL_BG

    return out


def _fill_unclassified(labels: np.ndarray) -> np.ndarray:
    """

    :param labels:
    :return:
    """
    out = labels.copy().astype(np.int8)
    unc = labels < 0
    if not unc.any():
        return out

    min_d = np.full(labels.shape, np.inf, dtype=np.float32)
    for lab in range(N_CLASSES):
        seed = labels == lab
        if not seed.any():
            continue
        dt = cv2.distanceTransform((~seed).astype(np.uint8)*255, cv2.DIST_L2, 5)

        update = unc & (dt < min_d)
        min_d[update] = dt[update]
        out[update] = lab

    return out


def _enhance_defect_regions(labels: np.ndarray, hsv: np.ndarray, part_silhouette: np.ndarray) -> np.ndarray:
    """ improve scratch / defect mask quality"""

    h, w = hsv.shape[:2]
    hue = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)

    out = labels.copy()
    core = (labels == LABEL_DEFECT).astype(np.uint8) *255
    in_part = part_silhouette > 0
    is_blue = ((hue >= BLUE_H_LO) & (hue <= BLUE_H_HI) & (s >= BLUE_S_MIN) & (v >= BLUE_V_MIN))

    anomaly = (in_part & ~is_blue & (labels != LABEL_WIRE) & (labels != LABEL_BG) & (v >= WIRE_VAL_MIN)).astype(np.uint8) * 255
    core = cv2.bitwise_or(core, anomaly)
    if cv2.countNonZero(core) == 0:
        return out

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k3)

    r = DEFECT_MERGE_RADIUS
    k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
    merged = cv2.dilate(seed, k_merge)
    merged = cv2.bitwise_and(merged, part_silhouette)

    min_core_px = max(60, int(h * w * DEFECT_MIN_CORE_FRAC))
    n_lbl, lbl_map, _, _ = cv2.connectedComponentsWithStats(merged, connectivity=4)

    new_defect = np.zeros((h,w), dtype=np.uint8)
    for i in range(1, n_lbl):
        region = lbl_map == i
        if int(np.count_nonzero(seed[region])) < min_core_px:
            continue
        new_defect[region & (seed > 0)] = 255

    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    new_defect = cv2.morphologyEx(new_defect, cv2.MORPH_CLOSE, k_bridge)
    new_defect = cv2.bitwise_and(new_defect, part_silhouette)

    can_update = (out != LABEL_WIRE) & (out != LABEL_BG)
    out[(new_defect > 0) & can_update] = LABEL_DEFECT

    return out


def _refine(labels: np.ndarray) -> np.ndarray:
    """clean up each class"""
    h, w = labels.shape
    out = np.full_like(labels, -1)
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for lab in range(N_CLASSES):
        m = (labels == lab).astype(np.uint8)
        if lab != LABEL_DEFECT:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_o)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_c)
        out[m>0] = lab

    min_area = max(30, int(h*w*SMALL_REGION_FRAC))
    m = (out == LABEL_WIRE).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            out[lbl == i] = -1

    return _fill_unclassified(out)


def segment(img: np.ndarray) -> np.ndarray:
    """ segment image into 4 labelled regions"""

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, part_sil = _part_masks(hsv)
    labels = _classify(hsv, part_sil)
    labels = _fill_unclassified(labels)
    labels = _enhance_defect_regions(labels, hsv, part_sil)
    labels = _refine(labels)

    return labels.astype(np.uint8)


def classify(labels: np.ndarray, threshold: float = DEFECT_THRESHOLD) -> tuple[str, float]:
    """ Classify a segmented image as good or bad"""

    defect_frac = float(np.count_nonzero(labels == LABEL_DEFECT)) /labels.size
    verdict = "bad" if defect_frac >= threshold else "good"
    return verdict, defect_frac


def labels_to_color(labels: np.ndarray) -> np.ndarray:
    """ Convert label map to BRG color image"""
    return PALETTE[labels]

def save_results(img: np.ndarray, labels: np.ndarray, out_dir: Path, stem: str, save_overlay: bool=True) -> None:
    """Write label map, false-color seg, and optional overlay"""
    cv2.imwrite(str(out_dir/f"{stem}_labels.png"), labels)
    seg = labels_to_color(labels)
    cv2.imwrite(str(out_dir/f"{stem}_seg.png"), seg)

    if save_overlay:
        blended = cv2.addWeighted(img, 0.55, seg, 0.45, 0)
        cv2.imwrite(str(out_dir/f"{stem}_overlay.png"), blended)


def main() -> None:

    parser = argparse.ArgumentParser(description="multi-region segmentation")
    parser.add_argument("--input", default="data", help="Input image directory")
    parser.add_argument("--out", default="output/seg", help="Output directory for labels/seg/overlay")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"), help="Resize before segmenting)")
    parser.add_argument("--no-preprocess", action='store_true', help="skip bilateral filter + CLAHE")
    parser.add_argument("--no-overlay", action='store_true', help="skip overlay generation")
    parser.add_argument("--threshold", type=float, default=DEFECT_THRESHOLD, metavar="T", help=f"defect fraction threshold for bad/good verdict")

    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.size) if args.size else None

    images = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {in_dir}")
        return

    palette_desc = "  ".join(f"{i}={CLASS_NAMES[i]}" for i in range(N_CLASSES))

    print(f"segment {len(images)} image(s) -> {out_dir}/")
    print(f"Labels: {palette_desc}")
    print(f"Threshold: defect >= {args.threshold *100:.2f} % -> bad")
    print()

    n_good, n_bad = 0, 0

    for src in images:
        raw = cv2.imread(str(src))
        if raw is None:
            print(f" [skip] {src.name}")
            continue

        img = raw if args.no_preprocess else preprocess(raw, target_size)
        labels = segment(img)
        stem = src.stem
        verdict, defect_frac = classify(labels, threshold=args.threshold)

        h, w = labels.shape
        tot = h*w
        cov = "  ".join(f"{CLASS_NAMES[i][:4]} {np.count_nonzero(labels == i)/tot * 100:.1f}%" for i in range(N_CLASSES))
        tag = "GOOD" if verdict == "good" else f"BAD (defect {defect_frac *100:.2f}%)"
        print(f"  {src.name:<28} {w}x{h} {cov} -> {tag}")

        if verdict == "good":
            n_good + 1
        else:
            n_bad += 1

        save_results(img, labels, out_dir, stem, save_overlay=not args.no_overlay)

    print(f"\n Summary: {n_good} good / {n_bad} bad "
          f" (threshold {args.threshold *100:.2f}%)")
    print("DONE")


if __name__ == "__main__":
    main()








