#!/usr/bin/env python3
"""
Parse desktop and fpga float/posit logs and plot per-feature drift
between desktop and fpga for each format side-by-side.

Usage:
    python plot_drift.py
    python plot_drift.py desktop_float.log fpga_float.log desktop_posit.log fpga_posit.log
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

WINDOW_RE = re.compile(r"^PROCESS WINDOW (\d+) WITH MODEL (\d+)")
SECTION_RE = re.compile(r"^(IMU|AUDIO) FEATURES$")


def parse_log(path):
    data = {}
    if not os.path.exists(path):
        return data

    with open(path, "r") as f:
        lines = [l.rstrip("\n") for l in f]

    i = 0
    current_window = None
    while i < len(lines):
        line = lines[i].strip()

        m = WINDOW_RE.match(line)
        if m:
            current_window = int(m.group(1))
            i += 1
            continue

        m = SECTION_RE.match(line)
        if m and current_window is not None:
            kind = m.group(1)
            i += 1
            if i < len(lines):
                values = [float(x) for x in lines[i].split()]
                data[(current_window, kind)] = np.array(values)
            i += 1
            continue

        i += 1

    return data


def build_matrix(data_a, data_b, kind, drift_type="absolute"):
    common_windows = sorted(
        w for (w, k) in data_a.keys()
        if k == kind and (w, k) in data_b
    )

    if not common_windows:
        return [], None

    n_feats = len(data_a[(common_windows[0], kind)])
    matrix = np.zeros((n_feats, len(common_windows)))

    for col, w in enumerate(common_windows):
        a = data_a[(w, kind)]
        b = data_b[(w, kind)]
        if len(a) != len(b):
            print(f"WARNING: {kind} window {w} length mismatch "
                  f"({len(a)} vs {len(b)}), skipping")
            continue
        if drift_type == "absolute":
            matrix[:, col] = np.abs(a - b)
        elif drift_type == "relative":
            matrix[:, col] = np.abs(a - b) / (np.abs(a) + 1e-9)

    return common_windows, matrix


def get_feature_labels(kind, n_features):
    labels = []
    if kind == "AUDIO":
        audio_base = {
            0: "SPECTRAL_DECREASE", 1: "SPECTRAL_SLOPE", 2: "SPECTRAL_ROLLOFF",
            3: "SPECTRAL_CENTROID", 4: "SPECTRAL_SPREAD", 5: "SPECTRAL_KURTOSIS",
            6: "SPECTRAL_SKEW", 7: "SPECTRAL_FLATNESS", 8: "SPECTRAL_STD",
            9: "SPECTRAL_ENTROPY", 10: "DOMINANT_FREQUENCY",
        }
        for i in range(n_features):
            if i in audio_base:
                name = audio_base[i]
            elif 11 <= i <= 13:
                name = f"PSD_{i-10}"
            elif 14 <= i < 270:
                mfcc_idx = i - 14
                family = mfcc_idx // 64
                coeff = mfcc_idx % 64
                fam_name = ["MEAN", "STD", "MAX", "ENTROPY"][family]
                name = f"MFCC_{fam_name}_{coeff}"
            elif i == 270:
                name = "ZRC"
            elif i == 271:
                name = "RMS"
            elif i == 272:
                name = "CREST_FACTOR"
            elif 273 <= i < 292:
                name = f"EEPD_{i-273}"
            else:
                name = "UNKNOWN"
            labels.append(f"[{i}] {name}")
    elif kind == "IMU":
        imu_bases = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z", "GYRO_Y", "GYRO_P", "GYRO_R", "ACCEL_COMBO", "GYRO_COMBO"]
        imu_families = ["LINE_LENGTH", "ZRC", "KURTOSIS", "RMS", "CREST_FACTOR"] + [f"AZC_{j}" for j in range(8)]
        for i in range(n_features):
            base_idx = i // 13
            fam_idx = i % 13
            if base_idx < len(imu_bases) and fam_idx < len(imu_families):
                name = f"{imu_bases[base_idx]}_{imu_families[fam_idx]}"
            else:
                name = "UNKNOWN"
            labels.append(f"[{i}] {name}")
    else:
        labels = [f"[{i}]" for i in range(n_features)]
    return labels


def plot_side_by_side_heatmap(pair1_res, pair2_res, kind, drift_type, out_path):
    win1, mat1, label1 = pair1_res
    win2, mat2, label2 = pair2_res

    if mat1 is None and mat2 is None:
        print(f"No overlapping {kind} windows found, skipping plot")
        return

    n_feats = mat1.shape[0] if mat1 is not None else mat2.shape[0]
    max_wins = max(len(win1) if win1 else 0, len(win2) if win2 else 0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(12, max_wins * 0.6), max(6, n_feats * 0.15)), sharey=True
    )

    labels = get_feature_labels(kind, n_feats)

    for ax, win, mat, title in [(ax1, win1, mat1, label1), (ax2, win2, mat2, label2)]:
        if mat is None or len(win) == 0:
            ax.set_title(f"{title} (No Data)")
            continue

        max_drift = mat.max()

        im = ax.imshow(
            mat, aspect="auto", cmap="inferno",
            norm=PowerNorm(gamma=0.3, vmin=0, vmax=max_drift)
        )
        ax.set_xlabel("Window")
        ax.set_title(f"{title}\nMax Drift: {max_drift:.4e}")
        ax.set_xticks(range(len(win)))
        ax.set_xticklabels(win, rotation=90, fontsize=6)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{drift_type.capitalize()} drift")

    ax1.set_ylabel("Feature")
    ax1.set_yticks(range(n_feats))
    ax1.set_yticklabels(labels, fontsize=6)

    # fig.suptitle(f"{kind} Feature {drift_type.capitalize()} Drift Comparison (Desktop vs FPGA)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    if len(sys.argv) == 5:
        p1_a, p1_b, p2_a, p2_b = sys.argv[1:5]
        pair1 = (p1_a, p1_b, f"Float ({os.path.basename(p1_a)} vs {os.path.basename(p1_b)})")
        pair2 = (p2_a, p2_b, f"Posit ({os.path.basename(p2_a)} vs {os.path.basename(p2_b)})")
    else:
        pair1 = ("desktop_float.log", "fpga_float.log", "Float (Desktop vs FPGA)")
        pair2 = ("desktop_posit.log", "fpga_posit.log", "Posit (Desktop vs FPGA)")

    data1_a = parse_log(pair1[0])
    data1_b = parse_log(pair1[1])
    data2_a = parse_log(pair2[0])
    data2_b = parse_log(pair2[1])

    for kind in ("IMU", "AUDIO"):
        for drift_type in ("absolute", "relative"):
            win1, mat1 = build_matrix(data1_a, data1_b, kind, drift_type)
            win2, mat2 = build_matrix(data2_a, data2_b, kind, drift_type)

            res1 = (win1, mat1, pair1[2])
            res2 = (win2, mat2, pair2[2])

            plot_side_by_side_heatmap(
                res1, res2, kind, drift_type, f"{kind.lower()}_drift_{drift_type}_comparison.png"
            )


if __name__ == "__main__":
    main()
