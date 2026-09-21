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
    pass


def confusion(tp, tn, fp, fn):
    pass


def roc_curve(gt, scores):
    pass


def plot_roc():
    pass


def main():
    pass


if __name__ == "__main__":
    main()

