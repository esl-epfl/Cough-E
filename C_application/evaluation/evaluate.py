"""
Evaluation pipeline for the Cough-E C application.

Pipeline per recording:
  1. Generate C header files (via transform_dataset.py)
  2. Update main.h includes to point to the generated headers
  3. Compile the C application
  4. Run the C application and capture output
  5. Parse COUGH_SEG lines to get detected cough segment boundaries
  6. Compare with ground truth using event-based scoring (timescoring)

Usage:
    python C_application/evaluation/evaluate.py                                     # full pipeline, all subjects
    python C_application/evaluation/evaluate.py full --subjects 14287 14342         # specific subjects
    python C_application/evaluation/evaluate.py full --dataset_path /path/to/data   # custom dataset path
    python C_application/evaluation/evaluate.py aggregate --csv C_application/evaluation/results.csv  # re-aggregate from CSV
    python C_application/evaluation/evaluate.py error                               # regression-style error metrics
    python C_application/evaluation/evaluate.py both                                # ML metrics + regression-style error metrics
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

try:
    from timescoring.annotations import Annotation
    from timescoring import scoring
except ModuleNotFoundError:
    Annotation = None
    scoring = None

sys.path.insert(0, os.path.dirname(__file__))
from transform_dataset import (
    transform_recording, transform_all, make_recording_suffix,
    AUDIO_FS_TARGET, IMU_FS, FS_IMU,
    SOUNDS, NOISES, MOVEMENTS, TRIALS,
)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

# Scoring parameters (matching ML_methodology/config/scoring/default.yaml)
TOLERANCE_START = 0.25
TOLERANCE_END = 0.25
MIN_COUGH_DURATION = 0.1
MAX_EVENT_DURATION = 0.6
MIN_DURATION_BTWN_EVENTS = 0
MIN_OVERLAP = MIN_COUGH_DURATION / 0.8  # 0.125

# Paths (relative to repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C_APP_DIR = os.path.join(REPO_ROOT, "C_application")
MAIN_H_PATH = os.path.join(C_APP_DIR, "main.h")
INPUT_DATA_DIR = os.path.join(C_APP_DIR, "input_data")
BUILD_DIR = os.path.join(C_APP_DIR, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "cough-e")

DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "Datasets", "full_dataset_test")

# Original main.h content for backup/restore
MAIN_H_ORIGINAL = None

CSV_FIELDNAMES = [
    "subject", "trial", "movement", "noise", "sound",
    "tp_evt", "fp_evt", "fn_evt", "se_evt", "ppv_evt", "f1_evt",
    "duration",
]

ERROR_CSV_FIELDNAMES = [
    "subject", "trial", "movement", "noise", "sound",
    "recording",
    "checks", "global_max_abs",
    "cont_n", "cont_sq_err", "cont_sq_float", "cont_rmse", "cont_rel_rmse_pct", "cont_max_abs",
    "count_n", "count_abs_err", "count_abs_float", "count_mae", "count_wape_pct", "count_max_abs",
]

ERROR_KERNEL_CSV_FIELDNAMES = [
    "feature", "metric_type", "n", "recordings",
    "rmse", "rel_rmse_pct", "mae", "wape_pct", "max_abs",
]

REGRESSION_HARNESS = os.path.join(C_APP_DIR, "test", "fxp_model_regression.c")
REGRESSION_SRCS = (
    sorted(glob.glob(os.path.join(C_APP_DIR, "Src", "*.c"))) +
    sorted(glob.glob(os.path.join(C_APP_DIR, "kiss_fftr", "*.c"))) +
    sorted(glob.glob(os.path.join(C_APP_DIR, "FxP", "imu", "*.c")))
)
REGRESSION_INC = [
    f"-I{os.path.join(C_APP_DIR, 'Inc')}",
    f"-I{os.path.join(C_APP_DIR, 'FxP')}",
    f"-I{C_APP_DIR}",
    f"-I{os.path.join(C_APP_DIR, 'kiss_fftr')}",
    f"-I{os.path.join(C_APP_DIR, 'range_analysis')}",
]
REGRESSION_CFLAGS = [
    "-Wall", "-Wextra", "-Wno-unused-function", "-Wno-unused-parameter",
    "-std=c11", "-O2", "-DFXP_MODE",
]


# ──────────────────────────────────────────────
#  main.h management
# ──────────────────────────────────────────────

def backup_main_h():
    """Save current main.h content so it can be restored after evaluation."""
    global MAIN_H_ORIGINAL
    with open(MAIN_H_PATH, 'r') as f:
        MAIN_H_ORIGINAL = f.read()


def restore_main_h():
    """Restore main.h to its original content."""
    if MAIN_H_ORIGINAL is not None:
        with open(MAIN_H_PATH, 'w') as f:
            f.write(MAIN_H_ORIGINAL)


def update_main_h(audio_relpath, imu_relpath, bio_relpath):
    """Replace the 3 input data #include lines in main.h."""
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


# ──────────────────────────────────────────────
#  Compile & run
# ──────────────────────────────────────────────

def compile_c_app(extra_flags=""):
    """Compile the C application with EVALUATION_MODE enabled. Returns True on success."""
    flags = "-DEVALUATION_MODE"
    if extra_flags:
        flags += " " + extra_flags
    result = subprocess.run(["make", "-C", C_APP_DIR, f"CFLAGS={flags}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Compilation failed: {result.stderr}")
        return False
    return True


def run_c_app():
    """Run the compiled C application and return stdout."""
    try:
        result = subprocess.run([EXECUTABLE], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("    WARNING: C app timed out after 60s (possible stuck)", flush=True)
        return ""
    return result.stdout


# ──────────────────────────────────────────────
#  Output parsing
# ──────────────────────────────────────────────

def parse_c_output(output, audio_fs=AUDIO_FS_TARGET):
    """
    Parse C application output to extract detected cough segments.

    The C app's FSM resets when IMU data runs out (re-processing from the start).
    We detect this by grouping segments by postprocessing period and stopping
    when a period's segments match an earlier period (indicating FSM restart).

    Returns list of (start_sec, end_sec) tuples.
    """
    periods = []
    current_period_segs = []

    for line in output.strip().split('\n'):
        seg_match = re.match(r'COUGH_SEG:\s+(\d+)\s+(\d+)', line)
        peaks_match = re.match(r'N_PEAKS FINAL:\s+(\d+)', line)

        if seg_match:
            start_sample = int(seg_match.group(1))
            end_sample = int(seg_match.group(2))
            current_period_segs.append((start_sample, end_sample))
        elif peaks_match:
            periods.append(current_period_segs)
            current_period_segs = []

    # Detect FSM reset: stop at first repeated period signature
    seen_signatures = set()
    first_pass_periods = []
    for period_segs in periods:
        sig = tuple(period_segs)
        if sig in seen_signatures and len(sig) > 0:
            break
        seen_signatures.add(sig)
        first_pass_periods.append(period_segs)

    # Flatten and deduplicate, converting samples to seconds
    segments = []
    seen_segments = set()
    for period_segs in first_pass_periods:
        for start_sample, end_sample in period_segs:
            key = (start_sample, end_sample)
            if key not in seen_segments:
                seen_segments.add(key)
                segments.append((start_sample / audio_fs, end_sample / audio_fs))

    return segments


# ──────────────────────────────────────────────
#  Ground truth & binary masks
# ──────────────────────────────────────────────

def load_ground_truth(dataset_path, subj_id, trial, mov, noise, sound):
    """Load ground truth cough events. Returns empty list for non-cough sounds."""
    if sound != "cough":
        return []
    gt_path = os.path.join(dataset_path, subj_id,
                           f'trial_{trial}', f'mov_{mov}',
                           f'background_noise_{noise}', sound,
                           'ground_truth.json')
    if not os.path.exists(gt_path):
        return []
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    return list(zip(gt["start_times"], gt["end_times"]))


def get_recording_duration(dataset_path, subj_id, trial, mov, noise, sound):
    """Get recording duration in seconds from IMU CSV line count."""
    imu_path = os.path.join(dataset_path, subj_id,
                            f'trial_{trial}', f'mov_{mov}',
                            f'background_noise_{noise}', sound,
                            'imu.csv')
    if os.path.exists(imu_path):
        with open(imu_path, 'r') as f:
            n_lines = sum(1 for _ in f) - 1  # subtract header
        return n_lines / IMU_FS
    return 0.0


def create_binary_mask(events, duration):
    """
    Create a binary mask at FS_IMU resolution from a list of (start, end) events.
    Mirrors edge_ai.get_ground_truth_regions().
    """
    n_samples = int(round(duration * FS_IMU))
    mask = np.zeros(n_samples)
    for start, end in events:
        s = min(int(round(start * FS_IMU)), n_samples)
        e = min(int(round(end * FS_IMU)), n_samples)
        mask[s:e] = 1
        if 0 < s < n_samples:
            mask[s - 1] = 0
    return mask


# ──────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────

def score_recording(gt_events, pred_events, duration):
    """
    Compute event-based scoring using timescoring.EventScoring.

    Parameters match ML_methodology/config/scoring/default.yaml.
    """
    if Annotation is None or scoring is None:
        raise RuntimeError(
            "timescoring is required for ML event metrics. "
            "Install dependencies or run the 'error' command for regression metrics only."
        )

    gt_mask = create_binary_mask(gt_events, duration)
    pred_mask = create_binary_mask(pred_events, duration)

    labels = Annotation(gt_mask, FS_IMU)
    pred = Annotation(pred_mask, FS_IMU)

    param = scoring.EventScoring.Parameters(
        TOLERANCE_START, TOLERANCE_END, MIN_OVERLAP,
        MAX_EVENT_DURATION, MIN_DURATION_BTWN_EVENTS
    )
    scores = scoring.EventScoring(labels, pred, param)

    return {
        "tp_evt": scores.tp,
        "fp_evt": scores.fp,
        "fn_evt": scores.refTrue - scores.tp,
        "se_evt": scores.sensitivity,
        "ppv_evt": scores.precision,
        "f1_evt": scores.f1,
    }


# ──────────────────────────────────────────────
#  Per-recording evaluation
# ──────────────────────────────────────────────

def evaluate_recording(subj_id, trial, mov, noise, sound,
                       dataset_path, input_data_dir):
    """Full pipeline for a single recording: transform -> compile -> run -> parse -> score."""
    if not hasattr(evaluate_recording, "_extra_flags"):
        evaluate_recording._extra_flags = ""
    result = transform_recording(subj_id, trial, mov, noise, sound,
                                 dataset_path, input_data_dir)
    if result is None:
        return None

    suffix, audio_relpath, imu_relpath, bio_relpath = result
    update_main_h(audio_relpath, imu_relpath, bio_relpath)

    if not compile_c_app(extra_flags=evaluate_recording._extra_flags):
        print(f"  FAILED to compile for {suffix}")
        return None

    try:
        output = run_c_app()
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {suffix}")
        return None

    pred_segments = parse_c_output(output)
    gt_events = load_ground_truth(dataset_path, subj_id, trial, mov, noise, sound)
    duration = get_recording_duration(dataset_path, subj_id, trial, mov, noise, sound)

    scores = score_recording(gt_events, pred_segments, duration)
    scores.update({
        "subject": subj_id,
        "trial": trial,
        "movement": mov,
        "noise": noise,
        "sound": sound,
        "duration": duration,
    })

    return scores


def get_subject_ids(dataset_path, subjects=None):
    """Return ordered subject IDs for the selected dataset subset."""
    if subjects is not None:
        return subjects
    return sorted([
        s for s in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, s))
    ])


def iter_recordings(dataset_path, subjects=None,
                    trials=TRIALS, movements=MOVEMENTS,
                    noises=NOISES, sounds=SOUNDS):
    """Yield all recording keys for the selected subset."""
    for subj_id in get_subject_ids(dataset_path, subjects):
        for trial in trials:
            for mov in movements:
                for noise in noises:
                    for sound in sounds:
                        yield subj_id, trial, mov, noise, sound


def evaluate_subjects(dataset_path, subjects=None,
                      trials=TRIALS, movements=MOVEMENTS,
                      noises=NOISES, sounds=SOUNDS):
    """Evaluate all recordings for the given subjects. Backs up and restores main.h."""
    backup_main_h()
    all_results = []

    try:
        for subj_id in get_subject_ids(dataset_path, subjects):
            print(f"\n=== Subject {subj_id} ===")
            for trial in trials:
                for mov in movements:
                    for noise in noises:
                        for sound in sounds:
                            rec_id = f"t{trial}_{mov}_{noise}_{sound}"
                            result = evaluate_recording(
                                subj_id, trial, mov, noise, sound,
                                dataset_path, INPUT_DATA_DIR)
                            if result is not None:
                                all_results.append(result)
                                print(f"  {rec_id}: TP_evt={result['tp_evt']} "
                                      f"FP_evt={result['fp_evt']} FN_evt={result['fn_evt']}")
    finally:
        restore_main_h()

    return all_results


# ──────────────────────────────────────────────
#  FxP regression-style error metrics
# ──────────────────────────────────────────────

def compile_regression_harness(imu_relpath, out_bin):
    """
    Compile fxp_model_regression.c for one recording by injecting IMU_HEADER.
    """
    imu_header = f"input_data/{imu_relpath}"
    cmd = (
        ["gcc"] +
        REGRESSION_CFLAGS +
        [f"-DIMU_HEADER=<{imu_header}>", "-o", out_bin, REGRESSION_HARNESS] +
        REGRESSION_SRCS +
        REGRESSION_INC +
        ["-lm"]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [COMPILE ERROR] {imu_relpath}")
        if result.stderr:
            print(result.stderr[:800])
        return False
    return True


def parse_regression_metrics(output):
    """
    Parse machine-readable lines emitted by fxp_model_regression.c.
    """
    parsed = {}
    kernel_rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("REG_METRICS_"):
            if line.startswith("REG_KERNEL_"):
                parts = line.split(",")
                tag = parts[0]
                kv = {}
                for part in parts[1:]:
                    if "=" not in part:
                        continue
                    key, val = part.split("=", 1)
                    kv[key.strip()] = val.strip()
                kv["_tag"] = tag
                kernel_rows.append(kv)
            continue
        parts = line.split(",")
        tag = parts[0]
        kv = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            kv[key.strip()] = val.strip()
        parsed[tag] = kv

    cont = parsed.get("REG_METRICS_CONT")
    count = parsed.get("REG_METRICS_COUNT")
    meta = parsed.get("REG_METRICS_META")
    if cont is None or count is None or meta is None:
        return None

    try:
        out = {
            "cont_n": int(cont["n"]),
            "cont_sq_err": float(cont["sum_sq_err"]),
            "cont_sq_float": float(cont["sum_sq_float"]),
            "cont_max_abs": float(cont["max_abs"]),
            "count_n": int(count["n"]),
            "count_abs_err": float(count["sum_abs_err"]),
            "count_abs_float": float(count["sum_abs_float"]),
            "count_max_abs": float(count["max_abs"]),
            "checks": int(meta["checks"]),
            "global_max_abs": float(meta["global_max_abs"]),
        }
        typed_kernels = []
        for k in kernel_rows:
            if k.get("_tag") == "REG_KERNEL_CONT":
                typed_kernels.append({
                    "metric_type": "continuous",
                    "signal": k["signal"],
                    "feature": k["feature"],
                    "n": int(k["n"]),
                    "sum_sq_err": float(k["sum_sq_err"]),
                    "sum_sq_float": float(k["sum_sq_float"]),
                    "max_abs": float(k["max_abs"]),
                })
            elif k.get("_tag") == "REG_KERNEL_COUNT":
                typed_kernels.append({
                    "metric_type": "count_based",
                    "signal": k["signal"],
                    "feature": k["feature"],
                    "n": int(k["n"]),
                    "sum_abs_err": float(k["sum_abs_err"]),
                    "sum_abs_float": float(k["sum_abs_float"]),
                    "max_abs": float(k["max_abs"]),
                })
        out["kernel_rows"] = typed_kernels
        return out
    except (KeyError, ValueError):
        return None


def run_regression_harness(bin_path):
    """Run compiled regression harness and parse machine-readable metrics."""
    try:
        result = subprocess.run([bin_path], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return parse_regression_metrics(result.stdout)


def _enrich_error_row(row):
    """Add derived per-recording metrics to an error row."""
    cont_n = row["cont_n"]
    count_n = row["count_n"]

    cont_rmse = math.sqrt(row["cont_sq_err"] / cont_n) if cont_n > 0 else 0.0
    cont_baseline_rms = math.sqrt(row["cont_sq_float"] / cont_n) if cont_n > 0 and row["cont_sq_float"] > 0 else 0.0
    cont_rel_rmse_pct = (100.0 * cont_rmse / cont_baseline_rms) if cont_baseline_rms > 0 else 0.0

    count_mae = (row["count_abs_err"] / count_n) if count_n > 0 else 0.0
    count_wape_pct = (100.0 * row["count_abs_err"] / row["count_abs_float"]) if row["count_abs_float"] > 0 else 0.0

    row["cont_rmse"] = cont_rmse
    row["cont_rel_rmse_pct"] = cont_rel_rmse_pct
    row["count_mae"] = count_mae
    row["count_wape_pct"] = count_wape_pct
    return row


def evaluate_error_metrics(dataset_path, subjects=None,
                           trials=TRIALS, movements=MOVEMENTS,
                           noises=NOISES, sounds=SOUNDS):
    """
    Run fxp_model_regression metrics across the selected dataset subset.
    """
    rows = []
    current_subject = None

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "fxp_model_regression_eval")

        for subj_id, trial, mov, noise, sound in iter_recordings(
            dataset_path, subjects=subjects, trials=trials,
            movements=movements, noises=noises, sounds=sounds,
        ):
            if subj_id != current_subject:
                current_subject = subj_id
                print(f"\n=== Subject {subj_id} (error metrics) ===")

            rec_id = f"t{trial}_{mov}_{noise}_{sound}"
            transformed = transform_recording(
                subj_id, trial, mov, noise, sound, dataset_path, INPUT_DATA_DIR
            )
            if transformed is None:
                continue

            suffix, _, imu_relpath, _ = transformed
            if not compile_regression_harness(imu_relpath, bin_path):
                print(f"  {rec_id}: SKIP (compile error)")
                continue

            metrics = run_regression_harness(bin_path)
            if metrics is None:
                print(f"  {rec_id}: SKIP (runtime/parse error)")
                continue

            row = {
                "subject": subj_id,
                "trial": trial,
                "movement": mov,
                "noise": noise,
                "sound": sound,
                "recording": suffix,
                **metrics,
            }
            _enrich_error_row(row)
            rows.append(row)
            print(f"  {rec_id}: checks={row['checks']}")

    return rows


def compute_error_aggregate(rows):
    """Aggregate regression-style error metrics across recordings."""
    cont_n = sum(r["cont_n"] for r in rows)
    cont_sq_err = sum(r["cont_sq_err"] for r in rows)
    cont_sq_float = sum(r["cont_sq_float"] for r in rows)
    cont_max_abs = max((r["cont_max_abs"] for r in rows), default=0.0)

    count_n = sum(r["count_n"] for r in rows)
    count_abs_err = sum(r["count_abs_err"] for r in rows)
    count_abs_float = sum(r["count_abs_float"] for r in rows)
    count_max_abs = max((r["count_max_abs"] for r in rows), default=0.0)

    checks = sum(r["checks"] for r in rows)
    global_max_abs = max((r["global_max_abs"] for r in rows), default=0.0)

    cont_rmse = math.sqrt(cont_sq_err / cont_n) if cont_n > 0 else 0.0
    cont_baseline_rms = math.sqrt(cont_sq_float / cont_n) if cont_n > 0 and cont_sq_float > 0 else 0.0
    cont_rel_rmse_pct = (100.0 * cont_rmse / cont_baseline_rms) if cont_baseline_rms > 0 else 0.0

    count_mae = (count_abs_err / count_n) if count_n > 0 else 0.0
    count_wape_pct = (100.0 * count_abs_err / count_abs_float) if count_abs_float > 0 else 0.0

    return {
        "total_recordings": len(rows),
        "checks": checks,
        "global_max_abs": global_max_abs,
        "cont_n": cont_n,
        "cont_sq_err": cont_sq_err,
        "cont_sq_float": cont_sq_float,
        "cont_rmse": cont_rmse,
        "cont_rel_rmse_pct": cont_rel_rmse_pct,
        "cont_max_abs": cont_max_abs,
        "count_n": count_n,
        "count_abs_err": count_abs_err,
        "count_abs_float": count_abs_float,
        "count_mae": count_mae,
        "count_wape_pct": count_wape_pct,
        "count_max_abs": count_max_abs,
    }


def compute_error_per_subject(rows):
    """Compute per-subject error-metric aggregates."""
    by_subject = {}
    for subj_id in sorted(set(r["subject"] for r in rows)):
        subj_rows = [r for r in rows if r["subject"] == subj_id]
        by_subject[subj_id] = compute_error_aggregate(subj_rows)
    return by_subject


def compute_kernel_aggregate(rows):
    """
    Aggregate metrics per kernel feature across all recordings.
    Kernels are grouped by feature family name (e.g., LINE_LENGTH, KURTOSIS, AZC_0...).
    """
    acc = {}
    for r in rows:
        for k in r.get("kernel_rows", []):
            feat = k["feature"]
            if feat not in acc:
                acc[feat] = {
                    "feature": feat,
                    "metric_type": k["metric_type"],
                    "n": 0,
                    "_recordings": set(),
                    "sum_sq_err": 0.0,
                    "sum_sq_float": 0.0,
                    "sum_abs_err": 0.0,
                    "sum_abs_float": 0.0,
                    "max_abs": 0.0,
                }
            a = acc[feat]
            a["_recordings"].add(r["recording"])
            a["n"] += k["n"]
            a["max_abs"] = max(a["max_abs"], k["max_abs"])
            if k["metric_type"] == "continuous":
                a["sum_sq_err"] += k["sum_sq_err"]
                a["sum_sq_float"] += k["sum_sq_float"]
            else:
                a["sum_abs_err"] += k["sum_abs_err"]
                a["sum_abs_float"] += k["sum_abs_float"]

    out = []
    for feat in sorted(acc):
        a = acc[feat]
        if a["metric_type"] == "continuous":
            rmse = math.sqrt(a["sum_sq_err"] / a["n"]) if a["n"] > 0 else 0.0
            baseline_rms = math.sqrt(a["sum_sq_float"] / a["n"]) if a["n"] > 0 and a["sum_sq_float"] > 0 else 0.0
            rel_rmse_pct = (100.0 * rmse / baseline_rms) if baseline_rms > 0 else 0.0
            out.append({
                "feature": feat,
                "metric_type": "continuous",
                "n": a["n"],
                "recordings": len(a["_recordings"]),
                "rmse": rmse,
                "rel_rmse_pct": rel_rmse_pct,
                "mae": 0.0,
                "wape_pct": 0.0,
                "max_abs": a["max_abs"],
            })
        else:
            mae = (a["sum_abs_err"] / a["n"]) if a["n"] > 0 else 0.0
            wape_pct = (100.0 * a["sum_abs_err"] / a["sum_abs_float"]) if a["sum_abs_float"] > 0 else 0.0
            out.append({
                "feature": feat,
                "metric_type": "count_based",
                "n": a["n"],
                "recordings": len(a["_recordings"]),
                "rmse": 0.0,
                "rel_rmse_pct": 0.0,
                "mae": mae,
                "wape_pct": wape_pct,
                "max_abs": a["max_abs"],
            })
    return out


# ──────────────────────────────────────────────
#  Aggregation
# ──────────────────────────────────────────────

def compute_aggregate_metrics(results):
    """Compute aggregate event-based metrics across all recordings."""
    total_duration_hrs = sum(r["duration"] for r in results) / 3600.0

    tp = sum(r["tp_evt"] for r in results)
    fp = sum(r["fp_evt"] for r in results)
    fn = sum(r["fn_evt"] for r in results)

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * se * pr / (se + pr) if (se + pr) > 0 else 0.0
    fphr = fp / total_duration_hrs if total_duration_hrs > 0 else 0.0

    return {
        "se_evt": se, "pr_evt": pr, "f1_evt": f1, "fphr_evt": fphr,
        "tp_evt": tp, "fp_evt": fp, "fn_evt": fn,
        "total_recordings": len(results),
        "total_duration_hrs": total_duration_hrs,
    }


# ──────────────────────────────────────────────
#  Output: CSV, JSON, terminal
# ──────────────────────────────────────────────

def save_results_csv(results, output_path):
    """Save per-recording results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_FIELDNAMES})
    print(f"Results CSV saved to {output_path}")


def load_results_csv(csv_path):
    """Load per-recording results from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                if k not in ("subject", "trial", "movement", "noise", "sound"):
                    row[k] = float(row[k])
            results.append(row)
    return results


def save_summary_json(aggregate, output_path, per_subject=None):
    """Save aggregate metrics to JSON."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "event_based": {
                "SE": round(aggregate["se_evt"], 4),
                "PR": round(aggregate["pr_evt"], 4),
                "F1": round(aggregate["f1_evt"], 4),
                "FP_hr": round(aggregate["fphr_evt"], 1),
                "TP": aggregate["tp_evt"],
                "FP": aggregate["fp_evt"],
                "FN": aggregate["fn_evt"],
            },
            "total_recordings": aggregate["total_recordings"],
            "total_duration_hrs": round(aggregate["total_duration_hrs"], 3),
        },
    }
    if per_subject:
        summary["per_subject"] = per_subject

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Summary JSON saved to {output_path}")


def build_per_subject_json(per_subject):
    """Build per-subject metrics dict for JSON output."""
    result = {}
    for subj, a in per_subject.items():
        result[subj] = {
            "event_based": {
                "SE": round(a["se_evt"], 4),
                "PR": round(a["pr_evt"], 4),
                "F1": round(a["f1_evt"], 4),
                "FP_hr": round(a["fphr_evt"], 1),
                "TP": a["tp_evt"],
                "FP": a["fp_evt"],
                "FN": a["fn_evt"],
            },
        }
    return result


def print_results(results, aggregate):
    """Print per-subject and overall results to terminal."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    subjects = sorted(set(r["subject"] for r in results))
    per_subject = {}
    for subj in subjects:
        subj_results = [r for r in results if r["subject"] == subj]
        a = compute_aggregate_metrics(subj_results)
        per_subject[subj] = a
        print(f"\nSubject {subj}:")
        print(f"  SE={a['se_evt']:.3f}  PR={a['pr_evt']:.3f}  "
              f"F1={a['f1_evt']:.3f}  FP/hr={a['fphr_evt']:.1f}  "
              f"(TP={a['tp_evt']} FP={a['fp_evt']} FN={a['fn_evt']})")

    print(f"\n{'=' * 70}")
    print(f"OVERALL ({aggregate['total_recordings']} recordings, "
          f"{aggregate['total_duration_hrs']:.3f} hrs)")
    print(f"{'=' * 70}")
    print(f"  SE    = {aggregate['se_evt']:.4f}")
    print(f"  PR    = {aggregate['pr_evt']:.4f}")
    print(f"  F1    = {aggregate['f1_evt']:.4f}")
    print(f"  FP/hr = {aggregate['fphr_evt']:.1f}")
    print(f"  TP={aggregate['tp_evt']}  FP={aggregate['fp_evt']}  FN={aggregate['fn_evt']}")
    print("=" * 70)

    return per_subject


def save_error_results_csv(results, output_path):
    """Save per-recording regression-style error metrics to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ERROR_CSV_FIELDNAMES})
    print(f"Error metrics CSV saved to {output_path}")


def save_error_kernel_summary_csv(kernel_rows, output_path):
    """Save aggregated per-kernel error metrics to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_KERNEL_CSV_FIELDNAMES)
        writer.writeheader()
        for r in kernel_rows:
            writer.writerow({k: r[k] for k in ERROR_KERNEL_CSV_FIELDNAMES})
    print(f"Kernel summary CSV saved to {output_path}")


def print_error_results(results, aggregate):
    """Print per-subject and overall regression-style error metrics."""
    print("\n" + "=" * 70)
    print("REGRESSION-STYLE ERROR METRICS (FxP vs FLOAT)")
    print("=" * 70)

    per_subject = compute_error_per_subject(results)
    for subj_id in sorted(per_subject):
        a = per_subject[subj_id]
        print(f"\nSubject {subj_id}:")
        print(f"  CONT: RMSE={a['cont_rmse']:.6g}  RelRMSE={a['cont_rel_rmse_pct']:.4f}%  MaxAbs={a['cont_max_abs']:.6g}")
        print(f"  AZC : MAE={a['count_mae']:.6g}  WAPE={a['count_wape_pct']:.4f}%  MaxAbs={a['count_max_abs']:.6g}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL ({aggregate['total_recordings']} recordings)")
    print(f"{'=' * 70}")
    print(f"  checks            = {aggregate['checks']}")
    print(f"  global_max_abs    = {aggregate['global_max_abs']:.6g}")
    print(f"  CONT n            = {aggregate['cont_n']}")
    print(f"  CONT RMSE         = {aggregate['cont_rmse']:.6g}")
    print(f"  CONT RelRMSE      = {aggregate['cont_rel_rmse_pct']:.4f}%")
    print(f"  CONT MaxAbs       = {aggregate['cont_max_abs']:.6g}")
    print(f"  COUNT n           = {aggregate['count_n']}")
    print(f"  COUNT MAE         = {aggregate['count_mae']:.6g}")
    print(f"  COUNT WAPE        = {aggregate['count_wape_pct']:.4f}%")
    print(f"  COUNT MaxAbs      = {aggregate['count_max_abs']:.6g}")
    print("=" * 70)

    return per_subject


def print_kernel_summary(kernel_rows):
    """Print aggregated per-kernel metrics."""
    print("\n" + "=" * 70)
    print("PER-KERNEL AGGREGATE ERROR METRICS")
    print("=" * 70)
    print(f"{'Kernel':<28} {'Type':<11} {'N':>8} {'Metric':>14} {'MaxAbs':>14}")
    print("-" * 70)
    for r in kernel_rows:
        if r["metric_type"] == "continuous":
            metric = f"RMSE={r['rmse']:.5g} ({r['rel_rmse_pct']:.3f}%)"
            mtype = "cont"
        else:
            metric = f"WAPE={r['wape_pct']:.3f}%"
            mtype = "count"
        print(f"{r['feature']:<28} {mtype:<11} {r['n']:>8} {metric:>14} {r['max_abs']:>14.6g}")
    print("=" * 70)


def _build_error_per_subject_json(per_subject):
    """Build serialisable per-subject error metrics block."""
    result = {}
    for subj_id, a in per_subject.items():
        result[subj_id] = {
            "continuous": {
                "N": a["cont_n"],
                "RMSE": round(a["cont_rmse"], 8),
                "RelRMSE_pct": round(a["cont_rel_rmse_pct"], 6),
                "MaxAbs": round(a["cont_max_abs"], 8),
            },
            "count_based": {
                "N": a["count_n"],
                "MAE": round(a["count_mae"], 8),
                "WAPE_pct": round(a["count_wape_pct"], 6),
                "MaxAbs": round(a["count_max_abs"], 8),
            },
            "checks": a["checks"],
            "global_max_abs": round(a["global_max_abs"], 8),
        }
    return result


def _build_kernel_summary_json(kernel_rows):
    """Build serialisable kernel summary block."""
    out = []
    for r in kernel_rows:
        entry = {
            "feature": r["feature"],
            "metric_type": r["metric_type"],
            "N": r["n"],
            "recordings": r["recordings"],
            "max_abs": round(r["max_abs"], 8),
        }
        if r["metric_type"] == "continuous":
            entry["rmse"] = round(r["rmse"], 8)
            entry["rel_rmse_pct"] = round(r["rel_rmse_pct"], 6)
        else:
            entry["mae"] = round(r["mae"], 8)
            entry["wape_pct"] = round(r["wape_pct"], 6)
        out.append(entry)
    return out


def save_error_summary_json(aggregate, output_path, per_subject=None, kernel_rows=None):
    """Save aggregated regression-style error metrics to JSON."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "continuous": {
                "N": aggregate["cont_n"],
                "RMSE": round(aggregate["cont_rmse"], 8),
                "RelRMSE_pct": round(aggregate["cont_rel_rmse_pct"], 6),
                "MaxAbs": round(aggregate["cont_max_abs"], 8),
            },
            "count_based": {
                "N": aggregate["count_n"],
                "MAE": round(aggregate["count_mae"], 8),
                "WAPE_pct": round(aggregate["count_wape_pct"], 6),
                "MaxAbs": round(aggregate["count_max_abs"], 8),
            },
            "checks": aggregate["checks"],
            "global_max_abs": round(aggregate["global_max_abs"], 8),
            "total_recordings": aggregate["total_recordings"],
        },
    }
    if per_subject:
        summary["per_subject"] = _build_error_per_subject_json(per_subject)
    if kernel_rows:
        summary["per_kernel"] = _build_kernel_summary_json(kernel_rows)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Error metrics summary JSON saved to {output_path}")


# ──────────────────────────────────────────────
#  CLI subcommands
# ──────────────────────────────────────────────

def cmd_transform(args):
    """Generate dataset only."""
    transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)


def cmd_run(args):
    """Run evaluation (transforms dataset if needed)."""
    output_dir = args.output_dir or os.path.dirname(__file__)
    if not getattr(args, "no_save", False):
        os.makedirs(output_dir, exist_ok=True)

    compile_flags = []
    if getattr(args, "fxp", False):
        compile_flags.append("-DFXP_MODE")
    kissfft_fixed = getattr(args, "kissfft_fixed", None)
    if kissfft_fixed is not None:
        compile_flags.append(f"-DFIXED_POINT={kissfft_fixed}")

    evaluate_recording._extra_flags = " ".join(compile_flags)
    if compile_flags:
        print(f"Custom compile flags enabled ({evaluate_recording._extra_flags})")
    else:
        print("Using default float compile flags")

    results = evaluate_subjects(
        args.dataset_path,
        subjects=args.subjects,
        sounds=args.sounds,
        noises=args.noises,
    )

    if not results:
        print("No recordings processed. Check dataset path and subject IDs.")
        sys.exit(1)

    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    if getattr(args, "no_save", False):
        print("Skipping CSV/JSON output (--no-save enabled)")
    else:
        save_results_csv(results, os.path.join(output_dir, "results.csv"))
        save_summary_json(aggregate, os.path.join(output_dir, "summary.json"),
                          build_per_subject_json(per_subject))


def cmd_aggregate(args):
    """Compute aggregate metrics from an existing CSV."""
    results = load_results_csv(args.csv)
    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    output_dir = os.path.dirname(args.csv) or "."
    save_summary_json(aggregate, os.path.join(output_dir, "summary.json"),
                      build_per_subject_json(per_subject))


def cmd_full(args):
    """Generate dataset + run evaluation."""
    print("Step 1/2: Transforming dataset...")
    transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)
    print("\nStep 2/2: Running evaluation...")
    cmd_run(args)


def cmd_compare(args):
    """Run float then FxP evaluation and print a side-by-side comparison."""
    output_dir = args.output_dir or os.path.dirname(__file__)
    if not getattr(args, "no_save", False):
        os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset_path

    print("=== Float evaluation ===")
    evaluate_recording._extra_flags = ""
    float_results = evaluate_subjects(dataset_path, subjects=args.subjects,
                                      sounds=args.sounds, noises=args.noises)
    float_agg = compute_aggregate_metrics(float_results)

    print("\n=== FxP evaluation ===")
    fxp_flags = ["-DFXP_MODE"]
    kissfft_fixed = getattr(args, "kissfft_fixed", None)
    if kissfft_fixed is not None:
        fxp_flags.append(f"-DFIXED_POINT={kissfft_fixed}")
    fxp_flags = " ".join(fxp_flags)
    print(f"Compile flags: {fxp_flags}")
    evaluate_recording._extra_flags = fxp_flags
    fxp_results = evaluate_subjects(dataset_path, subjects=args.subjects,
                                    sounds=args.sounds, noises=args.noises)
    fxp_agg = compute_aggregate_metrics(fxp_results)

    # Save individual CSVs
    if not getattr(args, "no_save", False):
        save_results_csv(float_results, os.path.join(output_dir, "results_float.csv"))
        save_results_csv(fxp_results,   os.path.join(output_dir, "results_fxp.csv"))

    # Print comparison
    print("\n" + "=" * 70)
    print("FLOAT vs FxP COMPARISON")
    print("=" * 70)
    fmt = "  {:<10} {:>8} {:>8} {:>8} {:>10}"
    print(fmt.format("Mode", "SE", "PR", "F1", "FP/hr"))
    print("  " + "-" * 46)
    print(fmt.format("Float",
                     f"{float_agg['se_evt']:.4f}",
                     f"{float_agg['pr_evt']:.4f}",
                     f"{float_agg['f1_evt']:.4f}",
                     f"{float_agg['fphr_evt']:.1f}"))
    print(fmt.format("FxP",
                     f"{fxp_agg['se_evt']:.4f}",
                     f"{fxp_agg['pr_evt']:.4f}",
                     f"{fxp_agg['f1_evt']:.4f}",
                     f"{fxp_agg['fphr_evt']:.1f}"))
    delta_se = fxp_agg['se_evt'] - float_agg['se_evt']
    delta_pr = fxp_agg['pr_evt'] - float_agg['pr_evt']
    delta_f1 = fxp_agg['f1_evt'] - float_agg['f1_evt']
    delta_fp = fxp_agg['fphr_evt'] - float_agg['fphr_evt']
    print(fmt.format("Delta",
                     f"{delta_se:+.4f}",
                     f"{delta_pr:+.4f}",
                     f"{delta_f1:+.4f}",
                     f"{delta_fp:+.1f}"))
    print("=" * 70)

    # Save comparison JSON
    import json
    comparison = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "dataset": dataset_path,
        "subjects": args.subjects,
        "float": {k: round(float_agg[k], 4) if isinstance(float_agg[k], float)
                  else float_agg[k] for k in float_agg},
        "fxp":   {k: round(fxp_agg[k],   4) if isinstance(fxp_agg[k],   float)
                  else fxp_agg[k]   for k in fxp_agg},
        "delta": {
            "se_evt":   round(delta_se, 4),
            "pr_evt":   round(delta_pr, 4),
            "f1_evt":   round(delta_f1, 4),
            "fphr_evt": round(delta_fp, 1),
        },
    }
    if getattr(args, "no_save", False):
        print("\nSkipping comparison JSON output (--no-save enabled)")
    else:
        cmp_path = os.path.join(output_dir, "comparison_float_vs_fxp.json")
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {cmp_path}")


def cmd_error(args):
    """Run regression-style FxP-vs-float error metrics across the selected dataset."""
    rows = evaluate_error_metrics(
        args.dataset_path,
        subjects=args.subjects,
        sounds=args.sounds,
        noises=args.noises,
    )
    if not rows:
        print("No recordings processed for error metrics. Check dataset path and subject IDs.")
        sys.exit(1)

    aggregate = compute_error_aggregate(rows)
    print_error_results(rows, aggregate)
    kernel_rows = compute_kernel_aggregate(rows)
    print_kernel_summary(kernel_rows)


def cmd_both(args):
    """
    Run ML metrics and regression-style error metrics in one command.
    ML phase can be either run-mode or compare-mode.
    """
    ml_mode = getattr(args, "ml_mode", "compare")
    if ml_mode == "run":
        print("=== ML metrics (run mode) ===")
        cmd_run(args)
    else:
        print("=== ML metrics (compare mode) ===")
        cmd_compare(args)

    print("\n=== Regression-style error metrics ===")
    cmd_error(args)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cough-E C application against full_dataset_test")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_args(p, fxp_flag=False, kissfft_flag=False, save_flag=False):
        p.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to full_dataset_test (default: {DEFAULT_DATASET_PATH})")
        p.add_argument("--subjects", nargs="+", type=str, default=None,
                        help="Specific subject IDs (default: all)")
        p.add_argument("--sounds", nargs="+", type=str, default=SOUNDS)
        p.add_argument("--noises", nargs="+", type=str, default=NOISES)
        p.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: evaluation/)")
        if fxp_flag:
            p.add_argument("--fxp", action="store_true", default=False,
                           help="Compile with -DFXP_MODE (IMU fixed-point kernels)")
        if kissfft_flag:
            p.add_argument(
                "--kissfft-fixed",
                type=int,
                choices=[16, 32],
                default=None,
                help=(
                    "Compile KissFFT in fixed-point mode with selected precision "
                    "(-DFIXED_POINT=16 or -DFIXED_POINT=32)."
                ),
            )
        if save_flag:
            p.add_argument(
                "--no-save",
                action="store_true",
                default=False,
                help="Print metrics only; do not write CSV/JSON outputs.",
            )

    p_transform = subparsers.add_parser("transform", help="Generate C headers from dataset")
    add_common_args(p_transform)
    p_transform.set_defaults(func=cmd_transform)

    p_run = subparsers.add_parser("run", help="Run evaluation")
    add_common_args(p_run, fxp_flag=True, kissfft_flag=True, save_flag=True)
    p_run.set_defaults(func=cmd_run)

    p_agg = subparsers.add_parser("aggregate", help="Compute metrics from existing CSV")
    p_agg.add_argument("--csv", type=str, required=True, help="Path to results CSV")
    p_agg.set_defaults(func=cmd_aggregate)

    p_full = subparsers.add_parser("full", help="Transform dataset + run evaluation")
    add_common_args(p_full, fxp_flag=True, kissfft_flag=True, save_flag=True)
    p_full.set_defaults(func=cmd_full)

    p_compare = subparsers.add_parser("compare",
                                      help="Run float and FxP evaluations side-by-side")
    add_common_args(p_compare, kissfft_flag=True, save_flag=True)
    p_compare.set_defaults(func=cmd_compare)

    p_error = subparsers.add_parser("error",
                                    help="Run regression-style error metrics on selected recordings")
    add_common_args(p_error)
    p_error.set_defaults(func=cmd_error)

    p_both = subparsers.add_parser("both",
                                   help="Run ML metrics and regression-style error metrics")
    add_common_args(p_both, fxp_flag=True, kissfft_flag=True, save_flag=True)
    p_both.add_argument("--ml_mode", choices=["compare", "run"], default="compare",
                        help="ML phase mode for 'both' command (default: compare)")
    p_both.set_defaults(func=cmd_both)

    args = parser.parse_args()

    # Default to 'compare' when no subcommand given
    if args.command is None:
        args = parser.parse_args(["compare"])

    args.func(args)


if __name__ == "__main__":
    main()
