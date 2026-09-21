import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_defect import classify, preprocess, segment, DEFECT_THRESHOLD


def load_labels(path: Path):
    """ Return list of (filename, label) from a whitespace-delimited label file.
    label is normalized to lowercase 'good' or 'bad'"""
    entries = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f" [warn] line {lineno}: expected '<name> <label>', got {line!r}")
                continue
            name, label = parts[0], parts[1].lower()
            continue
        entries.append((name, label))
    return entries

def confusion(tp, tn, fp, fn):
    total = tp+tn+fp+fn
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    f1 = (2 * precision * recall) /(precision + recall) if (precision + recall) > 0 else 0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall, accuracy=accuracy, f1=f1)


def roc_curve(gt, scores):
    """comput (fpr, tpr, threshold, auc) by sweeping every unique score value"""
    n_pos = int(gt.sum())
    n_neg = int((~gt).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0., 1.0]), np.array([0., 1.]), np.array([1., 0.]), float("nan")
    thresholds = np.concatenate([[np.nextafter(scores.max(), np.inf)], np.sort(np.unique(scores))[::-1], [0, 0]])
    fprs, tprs = [], []
    for t in thresholds:
        pred = scores >= t
        tpr = int((pred&gt).sum()) / n_pos
        fpr = int((pred& (~gt)).sum()) /n_neg
        tprs.append(tpr)
        fprs.append(fpr)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    auc = float(np.trapz(tprs[order]), fprs[order])
    return fprs, tprs, thresholds, auc


def plot_roc(fprs, tprs, auc, op_fpr, op_tpr, threshold, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fprs, tprs, color="steelblue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.scatter([op_fpr], [op_tpr], color="crimson", zorder=5, s=80,
               label=f"Threshold {threshold * 100:.2f}% f(TPR={op_tpr:.2f}, FPR={op_fpr:.2f}")
    ax.set_xlabel("Fasle Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Roc Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="evaluate good part classifier")
    parser.add_argument("--labels", required=True, help="Label file: '<image_name> <good|bad>' per line")
    parser.add_argument("--input", default="data", help="Image directory")
    parser.add_argument("--threshold", type=float, default=DEFECT_THRESHOLD, metavar="T", help="defect fraction threshold for bad verdict")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"), help="Resize images before segmenting")
    parser.add_argument("--no_preprocess", action="store_true", help="Skip lilateral filter + CLAHE")
    parser.add_argument("--roc", default='roc.png', help='output path for roc curve plot')
    args = parser.parse_args()

    label_path = Path(args.labels)
    in_dir = Path(args.input)
    roc_path = Path(args.roc)
    target_size = tuple(args.size) if args.size else None

    entries = load_labels(label_path)
    if not entries:
        print("NO valid entries in label file = aborting")
        sys.exit(1)
    print(f"eval.py - {len(entries)} labelled image(s)")
    print(f" input dir: {in_dir}")
    print(f" Threshold: defect >= {args.threshold * 100:.2f}% -> bad")
    print()

    gt_list = []
    score_list = []
    for name, gt_label in entries:
        img_path = in_dir/name
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f" [skip] {name} (cannot read)")
            continue
        img = raw if args.no_preprocess else preprocess(raw, target_size)
        labels_map = segment(img)
        verdict, defect_frac = classify(labels_map, threshold=args.threshold)

        gt_pos = gt_label == "bad"
        pred_pos = verdict == "bad"

        if gt_pos and pred_pos: outcome="TP"
        elif not gt_pos and not pred_pos: outcome="TN"
        elif not gt_pos and pred_pos: outcome="FP"
        else: outcome="FN"

        gt_list.append(gt_pos)
        score_list.append(defect_frac)

    if not gt_list:
        print("NO images processed")
        sys.exit(1)

    gt = np.array(gt_list, dtype=bool)
    scores = np.array(score_list, dtype=float)

    preds = scores >= args.threshold
    tp = int((preds&gt).sum())
    tn = int((~preds&~gt).sum())
    fp = int((preds&~gt).sum())
    fn = int((~preds&gt).sum())
    m = confusion(tp, tn, fp, fn)

    fprs, tprs,_, auc = roc_curve(gt, scores)
    n_pos = int(gt.sum())
    n_neg = int((~gt).sum())
    op_tpr = tp/n_pos if n_pos > 0 else 0
    op_fpr = fp/n_neg if n_neg > 0 else 0

    print()
    print("-" * 52)
    print(f" Confusion matrix (positive = bad)")
    print(f" {'TP':>4} = {tp:>4}  FN = {fn:>4}")
    print(f" {'FP':>4} = {fp:>4}  TN = {tn:>4}")
    print()
    _fmt = lambda v: f"{v:.4f}" if v == v else " n/a"
    print(f" Precision: {_fmt(m['precision'])} "
          f"({tp} of {tp+fp} predicted bad are truly bad")
    print(f" Recall: {_fmt(m['recall'])} "
          f"({tp} of {tp+fn} actual bad are caught")
    print(f" Accuracy : {_fmt(m['accuracy'])} "
          f"({tp+tn} of {tp+tn+fp+fn} predictions are correct")
    print(f" F1 score: {_fmt(m['f1'])}")
    print(f"ROC AUC: {_fmt(auc)}")
    print("-"*52)

    plot_roc(fprs, tprs, auc, op_fpr, op_tpr, args.threshold, roc_path)
    print(f"\n ROC curve saved -> {roc_path}")




if __name__ == "__main__":
    main()

