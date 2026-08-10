#!/usr/bin/env python3
"""
Compare numerical accuracy of upos16q / upos32nq posit log files against
float log files under ../evaluation/numerical_analysis/{float,upos16q,upos32nq}/*.log

For each feature (IMU_i / AUDIO_i), computes:
  - avg_float_value : mean absolute float value across all windows/files (magnitude reference)
  - MRE_upos32nq, MAE_upos32nq : mean relative / absolute error of upos32nq vs float
  - MRE_upos16q,  MAE_upos16q  : mean relative / absolute error of upos16q  vs float

Relative error is set to N/A (excluded from MRE) when abs(float_value) < REL_ERR_THRESHOLD,
since dividing by a near-zero float blows up the relative error and is not meaningful.
MAE is always computed regardless of magnitude.

Only files present in ALL THREE directories are used; if a file is missing from any
directory, a warning is printed and that filename is skipped entirely.

Duplicate (type, window) entries within a single file: only the first occurrence is used.
"""

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

REL_ERR_THRESHOLD = 1e-4
DECIMAL_PLACES = 10  # fixed decimal places for CSV output, no scientific notation


def format_fixed(x, decimals=DECIMAL_PLACES):
    """Format a float as a fixed-point decimal string (never scientific notation).
    Truncates (does not round) at `decimals` places."""
    if x is None:
        return "N/A"
    sign = "-" if x < 0 else ""
    x = abs(x)
    factor = 10 ** decimals
    truncated = math.floor(x * factor) / factor
    return f"{sign}{truncated:.{decimals}f}"


# ---------------------------------------------------------------------------
# Feature labeling (from provided benchmark tool)
# ---------------------------------------------------------------------------
def feature_label(kind, i):
    if kind == "AUDIO":
        audio_base = {
            0: "SPECTRAL_DECREASE", 1: "SPECTRAL_SLOPE", 2: "SPECTRAL_ROLLOFF",
            3: "SPECTRAL_CENTROID", 4: "SPECTRAL_SPREAD", 5: "SPECTRAL_KURTOSIS",
            6: "SPECTRAL_SKEW", 7: "SPECTRAL_FLATNESS", 8: "SPECTRAL_STD",
            9: "SPECTRAL_ENTROPY", 10: "DOMINANT_FREQUENCY",
        }
        if i in audio_base:
            name = audio_base[i]
        elif 11 <= i <= 13:
            name = f"POWER_SPECTRAL_DENSITY_BAND_{i - 10}"
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
            name = f"ENERGY_ENVELOPE_PEAK_DETECT_{i - 273}"
        else:
            name = "UNKNOWN"
        return f"AUDIO[{i}] {name}"

    elif kind == "IMU":
        imu_bases = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z", "GYRO_Y", "GYRO_P", "GYRO_R",
                     "ACCEL_COMBO", "GYRO_COMBO"]
        imu_families = (["LINE_LENGTH", "ZERO_CROSSING_RATE", "KURTOSIS",
                          "ROOT_MEANS_SQUARED", "CREST_FACTOR"] +
                         [f"APPROXIMATE_ZERO_CROSSING_{j}" for j in range(8)])
        base_idx = i // 13
        fam_idx = i % 13
        if base_idx < len(imu_bases) and fam_idx < len(imu_families):
            name = f"{imu_bases[base_idx]}_{imu_families[fam_idx]}"
        else:
            name = "UNKNOWN"
        return f"IMU[{i}] {name}"

    return f"{kind}[{i}] UNKNOWN"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log_file(path):
    """
    Returns dict: (type_str, window_int) -> list[float]
    Only the first occurrence of a given (type, window) is kept.
    """
    data = {}
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"WARNING: malformed line {line_no} in {path}, skipping", file=sys.stderr)
                continue
            kind = parts[0]
            if kind not in ("IMU", "AUDIO"):
                print(f"WARNING: unknown line type '{kind}' at line {line_no} in {path}, skipping",
                      file=sys.stderr)
                continue
            try:
                window = int(parts[1])
                values = [float(x) for x in parts[2:]]
            except ValueError:
                print(f"WARNING: could not parse numeric fields at line {line_no} in {path}, skipping",
                      file=sys.stderr)
                continue
            key = (kind, window)
            if key not in data:
                data[key] = values
            # else: duplicate window -> keep first occurrence only
    return data


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="../evaluation/numerical_analysis",
        help="Base directory containing float/upos16q/upos32nq subfolders "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="numerical_accuracy.csv",
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()

    base = args.base_dir
    dirs = {
        "float": os.path.join(base, "float"),
        "upos16q": os.path.join(base, "upos16q"),
        "upos32nq": os.path.join(base, "upos32nq"),
    }

    for label, d in dirs.items():
        if not os.path.isdir(d):
            print(f"ERROR: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    # Collect basenames present in each dir
    filesets = {}
    for label, d in dirs.items():
        filesets[label] = {os.path.basename(p) for p in glob.glob(os.path.join(d, "*.log"))}

    all_names = filesets["float"] | filesets["upos16q"] | filesets["upos32nq"]
    common_names = filesets["float"] & filesets["upos16q"] & filesets["upos32nq"]

    for name in sorted(all_names - common_names):
        missing_from = [label for label in dirs if name not in filesets[label]]
        print(f"WARNING: '{name}' missing from {missing_from}; ignoring this file in all folders",
              file=sys.stderr)

    if not common_names:
        print("ERROR: no common log files found across all three directories", file=sys.stderr)
        sys.exit(1)

    # Accumulators keyed by (kind, idx)
    float_vals_acc = defaultdict(list)          # for magnitude
    rel_err_acc = defaultdict(lambda: defaultdict(list))   # rel_err_acc[posit_label][(kind,idx)] -> list
    abs_err_acc = defaultdict(lambda: defaultdict(list))

    posit_labels = ["upos32nq", "upos16q"]  # posit32 first per requirement

    for name in sorted(common_names):
        float_data = parse_log_file(os.path.join(dirs["float"], name))

        # magnitude accumulation from float data alone
        for (kind, window), values in float_data.items():
            for idx, v in enumerate(values):
                float_vals_acc[(kind, idx)].append(abs(v))

        for posit_label in posit_labels:
            posit_data = parse_log_file(os.path.join(dirs[posit_label], name))
            for key, float_values in float_data.items():
                if key not in posit_data:
                    continue
                posit_values = posit_data[key]
                if len(posit_values) != len(float_values):
                    print(f"WARNING: feature count mismatch for {key} in {name} "
                          f"({posit_label}: {len(posit_values)} vs float: {len(float_values)}), "
                          f"skipping this line", file=sys.stderr)
                    continue
                kind = key[0]
                for idx, (fv, pv) in enumerate(zip(float_values, posit_values)):
                    abs_err = abs(pv - fv)
                    abs_err_acc[posit_label][(kind, idx)].append(abs_err)
                    if abs(fv) >= REL_ERR_THRESHOLD:
                        rel_err_acc[posit_label][(kind, idx)].append(abs_err / abs(fv))
                    # else: excluded from MRE (N/A case)

    # Build sorted list of feature keys: IMU first (by idx), then AUDIO (by idx)
    all_keys = set(float_vals_acc.keys())
    for posit_label in posit_labels:
        all_keys |= set(abs_err_acc[posit_label].keys())
    sorted_keys = sorted(all_keys, key=lambda k: (0 if k[0] == "IMU" else 1, k[1]))

    def mean(lst):
        return sum(lst) / len(lst) if lst else None

    def std(lst):
        # population stdev of the (abs) float values for this feature, to flag
        # features with a large swing in magnitude across windows/files
        if not lst:
            return None
        if len(lst) == 1:
            return 0.0
        return statistics.pstdev(lst)

    rows = []
    for kind, idx in sorted_keys:
        label = feature_label(kind, idx)
        float_list = float_vals_acc.get((kind, idx), [])
        avg_float = mean(float_list)
        std_float = std(float_list)

        row = {
            "feature": label,
            "avg_float_value": format_fixed(avg_float),
            "std_float_value": format_fixed(std_float),
        }
        for posit_label in posit_labels:
            mre = mean(rel_err_acc[posit_label].get((kind, idx), []))
            mae = mean(abs_err_acc[posit_label].get((kind, idx), []))
            row[f"MRE_{posit_label}"] = format_fixed(mre)
            row[f"MAE_{posit_label}"] = format_fixed(mae)
        rows.append(row)

    fieldnames = ["feature", "avg_float_value", "std_float_value",
                  "MRE_upos32nq", "MAE_upos32nq",
                  "MRE_upos16q", "MAE_upos16q"]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
