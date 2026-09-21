# Metal Part Defect Inspection

HSV-based segmentation and good / bad classification for blue-painted metal parts.
No deep-learning weights required — runs on CPU with OpenCV alone.

---

## Project structure

```
parts_classifier/
├── segment_defect.py        # 4-class segmentation + good/bad classifier
├── eval.py           # Evaluation: metrics table + ROC curve
├── requirements.txt
├── data/             # Raw input images *.png files
└── output/           # Generated masks and plots (created at runtime)
```

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## segment.py

Segments each image into four labelled regions and classifies the part as **good** or **bad**.

### Output classes

| Label | Class | False colour |
|-------|-------|-------------|
| 0 | background | near-black |
| 1 | good\_blue | vivid blue |
| 2 | defective | bright red |
| 3 | wire | silver |

A part is classified **bad** when the fraction of `defective` pixels meets or exceeds `--threshold` (default 0.35 %).

### Output files (written to `--out`)

| File | Description |
|------|-------------|
| `{stem}_labels.png` | uint8 label map — pixel value equals class id (0 – 3) |
| `{stem}_seg.png` | False-colour visualisation |
| `{stem}_overlay.png` | 55 / 45 blend of original and false-colour |

### Usage

```bash
# Segment all images in data/, write results to output/seg/
python segment.py --input data --out output/seg

# Custom defect threshold (2 %) for the good/bad verdict
python segment.py --input data --threshold 0.02

# Resize images to 512×512 before segmenting
python segment.py --input data --size 512 512

# Skip bilateral filter + CLAHE
python segment.py --input data --no-preprocess

# Skip overlay generation
python segment.py --input data --no-overlay
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `data` | Input image directory |
| `--out` | `output/seg_v2` | Output directory |
| `--threshold` | `0.01` | Defect fraction ≥ threshold → bad |
| `--size W H` | original size | Resize before segmenting |
| `--no-preprocess` | off | Skip bilateral filter + CLAHE |
| `--no-overlay` | off | Skip overlay image generation |

### Console output

```
segment.py  —  5 image(s)  →  output/seg/
  Labels:    0=background  1=good_blue  2=defective  3=wire
  Threshold: defect ≥ 1.00%  →  bad

  good_001.png                 1280×960  back 32.1%  good 61.4%  defe 0.1%  wire 6.4%  →  GOOD
  scratches_000.png                1280×960  back 31.8%  good 54.2%  defe 7.3%  wire 6.7%  →  BAD  (defect 7.32%)
  total_rust_003.png               1280×960  back 30.2%  good 28.1%  defe 35.4%  wire 6.3%  →  BAD  (defect 35.40%)

Summary:  1 good  /  2 bad  (threshold 1.00%)
```

---

## eval.py

Evaluates the classifier against ground-truth labels.  
Prints a confusion matrix, precision / recall / accuracy / F1, and saves a ROC curve.

**Positive class = bad** (scratches, major rust, total rust)  
**Negative class = good** (clean blue part)

### Label file format

Plain text, one image per line, whitespace-separated:

```
# labels.txt
good_001.png   good
good_002.png   good
scratches.png  bad
major_rust.png bad
total_rust.png bad
```

Blank lines and lines starting with `#` are ignored.

### Usage

```bash
# Basic evaluation with default 1 % threshold
python eval.py --labels labels.txt --input data

# Custom threshold
python eval.py --labels labels.txt --input data --threshold 0.02

# Save ROC curve to a specific path
python eval.py --labels labels.txt --input data --roc output/roc.png

# Resize images before segmenting (must match how segment.py was tuned)
python eval.py --labels labels.txt --input data --size 512 512
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--labels` | *(required)* | Path to the ground-truth label file |
| `--input` | `data` | Directory containing the images |
| `--threshold` | `0.01` | Defect fraction threshold (same as segment.py) |
| `--size W H` | original size | Resize before segmenting |
| `--no-preprocess` | off | Skip bilateral filter + CLAHE |
| `--roc` | `roc.png` | Output path for the ROC curve plot |

### Console output

```
  image                           gt    pred  defect   outcome
  ------------------------------  ----  ----  -------  -------
  good_001.png                    good  good    0.12%  TN
  good_002.png                    good  good    0.08%  TN
  scratches_001.png                   bad   bad     7.32%  TP
  major_rust_003.png                  bad   bad    22.10%  TP
  total_rust_004.png                  bad   bad    35.40%  TP

────────────────────────────────────────────────────────────────
  Confusion matrix  (positive = bad)
  TP =    3   FN =    0
  FP =    0   TN =    2

  Precision : 1.0000  (3 of 3 predicted-bad are truly bad)
  Recall    : 1.0000  (3 of 3 actual-bad are caught)
  Accuracy  : 1.0000  (5 of 5 correct)
  F1 score  : 1.0000
  ROC AUC   : 1.0000
────────────────────────────────────────────────────────────────

ROC curve saved → roc.png
```

### ROC curve

The plot sweeps every unique defect-fraction value as a classification threshold, giving a full ROC curve with AUC. The chosen operating point (`--threshold`) is marked in red.

---

## Tuning the threshold

Run `eval.py` and look at the **defect %** column:

- Good images cluster near **0 %**.
- Scratch images are typically **1 – 10 %**.
- Major / total rust images are typically **10 – 50 %+**.

Set `--threshold` to a value in the gap between the highest good-image score and the lowest bad-image score.  
The ROC curve shows AUC and helps pick the operating point that balances precision and recall for your use case.
