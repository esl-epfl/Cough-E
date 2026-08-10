#!/usr/bin/env python3
"""
Parse desktop.log and fpga.log (same window/model output format) and plot
per-feature drift between the two precision runs as heatmaps.

Usage:
    python plot_drift.py
    (expects desktop.log and fpga.log in the same directory, or pass paths)
    python plot_drift.py path/to/desktop.log path/to/fpga.log
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

WINDOW_RE = re.compile(r"^PROCESS WINDOW (\d+) WITH MODEL (\d+)")
SECTION_RE = re.compile(r"^(IMU|AUDIO) FEATURES$")


def parse_log(path):
    """
    Returns dict[(window, kind)] = np.array of feature values
    kind is 'IMU' or 'AUDIO'
    """
    data = {}
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
    """
    Align windows present in both logs for a given kind ('IMU' or 'AUDIO').
    Returns (windows_sorted, drift_matrix) where drift_matrix is
    features x windows.
    """
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


def plot_heatmap(windows, matrix, kind, drift_type, out_path):
    if matrix is None:
        print(f"No overlapping {kind} windows found, skipping plot")
        return

    max_drift = matrix.max()

    fig, ax = plt.subplots(figsize=(max(8, len(windows) * 0.4), max(6, matrix.shape[0] * 0.15)))
    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0, vmax=max_drift)

    ax.set_xlabel("Window")
    ax.set_ylabel("Feature")
    if drift_type == "absolute":
        ax.set_title(f"{kind} feature absolute drift |desktop - fpga|")
    else:
        ax.set_title(f"{kind} feature relative drift |desktop - fpga| / |desktop|")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=90, fontsize=6)

    labels = []
    if kind == "AUDIO":
        audio_base = {
            0: "SPECTRAL_DECREASE", 1: "SPECTRAL_SLOPE", 2: "SPECTRAL_ROLLOFF",
            3: "SPECTRAL_CENTROID", 4: "SPECTRAL_SPREAD", 5: "SPECTRAL_KURTOSIS",
            6: "SPECTRAL_SKEW", 7: "SPECTRAL_FLATNESS", 8: "SPECTRAL_STD",
            9: "SPECTRAL_ENTROPY", 10: "DOMINANT_FREQUENCY",
        }
        for i in range(matrix.shape[0]):
            if i in audio_base:
                name = audio_base[i]
            elif 11 <= i <= 13:
                name = f"POWER_SPECTRAL_DENSITY_BAND_{i-10}"
            elif 14 <= i < 270:
                mfcc_idx = i - 14
                family = mfcc_idx // 64
                coeff = mfcc_idx % 64
                fam_name = ["MEAN", "STD", "MAX", "ENTROPY"][family]
                name = f"MFCC_{fam_name}_{coeff}"
            elif i == 270:
                name = "ZERO_CROSSING_RATE"
            elif i == 271:
                name = "ROOT_MEANS_SQUARED"
            elif i == 272:
                name = "CREST_FACTOR"
            elif 273 <= i < 292:
                name = f"ENERGY_ENVELOPE_PEAK_DETECT_{i-273}"
            else:
                name = "UNKNOWN"
            labels.append(f"[{i}] {name}")
    elif kind == "IMU":
        imu_bases = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z", "GYRO_Y", "GYRO_P", "GYRO_R", "ACCEL_COMBO", "GYRO_COMBO"]
        imu_families = ["LINE_LENGTH", "ZERO_CROSSING_RATE", "KURTOSIS", "ROOT_MEANS_SQUARED", "CREST_FACTOR"] + [f"APPROXIMATE_ZERO_CROSSING_{j}" for j in range(8)]
        for i in range(matrix.shape[0]):
            base_idx = i // 13
            fam_idx = i % 13
            if base_idx < len(imu_bases) and fam_idx < len(imu_families):
                name = f"{imu_bases[base_idx]}_{imu_families[fam_idx]}"
            else:
                name = "UNKNOWN"
            labels.append(f"[{i}] {name}")
    else:
        labels = [f"[{i}]" for i in range(matrix.shape[0])]

    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(labels, fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute drift" if drift_type == "absolute" else "Relative drift")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"{kind} ({drift_type}): max drift = {max_drift:.6f}")


def main():
    desktop_path = sys.argv[1] if len(sys.argv) > 1 else "desktop.log"
    fpga_path = sys.argv[2] if len(sys.argv) > 2 else "fpga.log"

    data_a = parse_log(desktop_path)
    data_b = parse_log(fpga_path)

    for kind in ("IMU", "AUDIO"):
        for drift_type in ("absolute", "relative"):
            windows, matrix = build_matrix(data_a, data_b, kind, drift_type)
            plot_heatmap(windows, matrix, kind, drift_type, f"{kind.lower()}_drift_{drift_type}_heatmap.png")


if __name__ == "__main__":
    main()
