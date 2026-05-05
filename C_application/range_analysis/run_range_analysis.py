"""
Range analysis pipeline for the Cough-E C application.

Compiles with -DRANGE_ANALYSIS, runs across recordings, collects all RANGE|...
lines, and aggregates min/max/absmax statistics per (section, function, variable).

Usage (from repo root):
    python C_application/range_analysis/run_range_analysis.py
    python C_application/range_analysis/run_range_analysis.py --subjects 14287 14342
    python C_application/range_analysis/run_range_analysis.py --dataset_path /path/to/public_dataset
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

# Reuse the evaluation infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from transform_dataset import (
    transform_all, make_recording_suffix,
    SOUNDS, NOISES, MOVEMENTS, TRIALS,
)

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C_APP_DIR = os.path.join(REPO_ROOT, "C_application")
MAIN_H_PATH = os.path.join(C_APP_DIR, "main.h")
BUILD_DIR = os.path.join(C_APP_DIR, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "cough-e")
OUTPUT_DIR = os.path.join(C_APP_DIR, "range_analysis")
DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "Datasets", "full_dataset_test")

MAIN_H_ORIGINAL = None


def backup_main_h():
    global MAIN_H_ORIGINAL
    with open(MAIN_H_PATH, 'r') as f:
        MAIN_H_ORIGINAL = f.read()


def restore_main_h():
    if MAIN_H_ORIGINAL is not None:
        with open(MAIN_H_PATH, 'w') as f:
            f.write(MAIN_H_ORIGINAL)


def update_main_h(audio_relpath, imu_relpath, bio_relpath):
    with open(MAIN_H_PATH, 'r') as f:
        content = f.read()

    content = re.sub(r'#include <input_data/.*audio_input.*\.h>',
                     f'#include <input_data/{audio_relpath}>', content)
    content = re.sub(r'#include <input_data/.*imu_input.*\.h>',
                     f'#include <input_data/{imu_relpath}>', content)
    content = re.sub(r'#include <input_data/.*bio_input.*\.h>',
                     f'#include <input_data/{bio_relpath}>', content)

    with open(MAIN_H_PATH, 'w') as f:
        f.write(content)


def compile_c_app():
    import signal
    try:
        proc = subprocess.Popen(
            ["make", "-C", C_APP_DIR, "CFLAGS=-DEVALUATION_MODE -DRANGE_ANALYSIS"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True
        )
        stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        print(f"  Compilation timed out, skipping", flush=True)
        return False
    if proc.returncode != 0:
        print(f"  Compilation failed: {stderr}", flush=True)
        return False
    return True


def run_c_app():
    result = subprocess.run([EXECUTABLE], capture_output=True, text=True, timeout=60)
    return result.stdout


def parse_range_lines(output):
    """Extract all RANGE|... lines and return list of parsed tuples."""
    records = []
    for line in output.strip().split('\n'):
        if line.startswith("RANGE|"):
            parts = line.split("|")
            if len(parts) == 8:
                section = parts[1]
                func = parts[2]
                var = parts[3]
                length = int(parts[4])
                mn = float(parts[5])
                mx = float(parts[6])
                absmx = float(parts[7])
                records.append((section, func, var, length, mn, mx, absmx))
    return records


def aggregate(all_records):
    """
    Aggregate records across all recordings.
    For each (section, func, var): track global min, global max, global absmax,
    and count of observations.
    """
    stats = defaultdict(lambda: {
        "min": float("inf"),
        "max": float("-inf"),
        "absmax": 0.0,
        "count": 0,
        "len": 0,
        "always_positive": True,
    })

    for section, func, var, length, mn, mx, absmx in all_records:
        key = (section, func, var)
        s = stats[key]
        if mn < s["min"]:
            s["min"] = mn
        if mx > s["max"]:
            s["max"] = mx
        if absmx > s["absmax"]:
            s["absmax"] = absmx
        if mn < 0.0:
            s["always_positive"] = False
        s["count"] += 1
        s["len"] = length  # typically constant per key

    return stats


def get_subjects(dataset_path):
    """List subject directories in the dataset."""
    subjects = []
    for entry in sorted(os.listdir(dataset_path)):
        full = os.path.join(dataset_path, entry)
        if os.path.isdir(full) and entry.isdigit():
            subjects.append(entry)
    return subjects


def main():
    parser = argparse.ArgumentParser(description="Range analysis for Cough-E")
    parser.add_argument("--subjects", nargs="+", help="Specific subject IDs")
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.isdir(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    # Transform dataset if needed
    input_data_dir = os.path.join(C_APP_DIR, "input_data")
    print("Ensuring dataset is transformed...")
    transform_all(dataset_path, input_data_dir, args.subjects)

    subjects = args.subjects or get_subjects(dataset_path)
    print(f"Running range analysis on {len(subjects)} subjects")

    backup_main_h()

    all_records = []
    n_recordings = 0

    try:
        for subject in subjects:
            for trial in TRIALS:
                for movement in MOVEMENTS:
                    for noise in NOISES:
                        for sound in SOUNDS:
                            suffix = make_recording_suffix(subject, trial, movement, noise, sound)
                            sub_dir = os.path.join(input_data_dir, subject)
                            audio_file = f"audio_input_{suffix}.h"

                            if not os.path.exists(os.path.join(sub_dir, audio_file)):
                                continue

                            audio_rel = f"{subject}/audio_input_{suffix}.h"
                            imu_rel = f"{subject}/imu_input_{suffix}.h"
                            bio_rel = f"{subject}/bio_input_{subject}.h"

                            update_main_h(audio_rel, imu_rel, bio_rel)

                            if not compile_c_app():
                                print(f"  SKIP {subject}/{suffix} (compile fail)")
                                continue

                            output = run_c_app()
                            records = parse_range_lines(output)
                            all_records.extend(records)
                            n_recordings += 1

                            print(f"  {subject}/{suffix}: {len(records)} range samples", flush=True)

    finally:
        restore_main_h()

    if not all_records:
        print("No range data collected.")
        return

    # Aggregate
    stats = aggregate(all_records)

    # Sort by section, then function, then variable
    sorted_keys = sorted(stats.keys())

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "range_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "function", "variable", "len", "global_min",
                          "global_max", "global_absmax", "always_positive", "n_observations"])
        for key in sorted_keys:
            s = stats[key]
            writer.writerow([
                key[0], key[1], key[2], s["len"],
                f"{s['min']:.6e}", f"{s['max']:.6e}", f"{s['absmax']:.6e}",
                "yes" if s["always_positive"] else "no",
                s["count"]
            ])

    print(f"\nResults written to {csv_path}")
    print(f"Total recordings processed: {n_recordings}")
    print(f"Total range observations: {len(all_records)}")
    print(f"Unique (section, function, variable) keys: {len(stats)}")

    # Print summary table
    print(f"\n{'Section':<16} {'Function':<28} {'Variable':<20} {'Len':>5} "
          f"{'Min':>13} {'Max':>13} {'AbsMax':>13} {'Sign':>6} {'N':>6}")
    print("-" * 130)

    for key in sorted_keys:
        s = stats[key]
        sign = "U" if s["always_positive"] else "S"
        print(f"{key[0]:<16} {key[1]:<28} {key[2]:<20} {s['len']:>5} "
              f"{s['min']:>13.4e} {s['max']:>13.4e} {s['absmax']:>13.4e} {sign:>6} {s['count']:>6}")


if __name__ == "__main__":
    main()
