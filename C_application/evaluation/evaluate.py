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
    python C_application/evaluation/evaluate.py
    python C_application/evaluation/evaluate.py --mode fxp
    python C_application/evaluation/evaluate.py --mode fxp --twiddle 16
    python C_application/evaluation/evaluate.py --mode fxp-error
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

try:
    from timescoring.annotations import Annotation
    from timescoring import scoring
except ModuleNotFoundError:
    Annotation = None
    scoring = None

sys.path.insert(0, os.path.dirname(__file__))
_TRANSFORM_IMPORT_ERROR = None
try:
    from transform_dataset import (
        transform_recording,
        AUDIO_FS_TARGET, IMU_FS, FS_IMU,
        SOUNDS, NOISES, MOVEMENTS, TRIALS,
    )
except ModuleNotFoundError as exc:
    _TRANSFORM_IMPORT_ERROR = exc

    def _missing_transform_dependency(*_args, **_kwargs):
        raise RuntimeError(
            "transform_dataset dependencies are missing. Install required Python packages "
            "(for example scipy) to run evaluation commands."
        ) from _TRANSFORM_IMPORT_ERROR

    transform_recording = _missing_transform_dependency
    AUDIO_FS_TARGET = 8000
    IMU_FS = 100
    FS_IMU = 100
    SOUNDS = ["cough"]
    NOISES = ["none"]
    MOVEMENTS = ["none"]
    TRIALS = ["1"]


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

FXP_DIR = os.path.join(os.path.dirname(__file__), "fxp")

# Original main.h content for backup/restore
MAIN_H_ORIGINAL = None

CSV_FIELDNAMES = [
    "subject", "trial", "movement", "noise", "sound",
    "tp_evt", "fp_evt", "fn_evt", "se_evt", "ppv_evt", "f1_evt",
    "duration",
]


class EvaluationError(RuntimeError):
    """Raised when a real dataset recording cannot be evaluated."""


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
            "Install dependencies or run evaluate.py --mode fxp-error for FxP error metrics."
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
        raise EvaluationError(f"transform failed for {subj_id} t{trial} {mov} {noise} {sound}")
    suffix, audio_relpath, imu_relpath, bio_relpath = result

    update_main_h(audio_relpath, imu_relpath, bio_relpath)

    if not compile_c_app(extra_flags=evaluate_recording._extra_flags):
        print(f"  FAILED to compile for {suffix}")
        raise EvaluationError(f"C application compile failed for {suffix}")

    try:
        output = run_c_app()
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {suffix}")
        raise EvaluationError(f"C application timed out for {suffix}") from None

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


def get_subject_ids(dataset_path):
    """Return ordered subject IDs for the dataset."""
    return sorted([
        s for s in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, s))
    ])


def _recording_path(dataset_path, subj_id, trial, mov, noise, sound):
    return os.path.join(dataset_path, subj_id,
                        f'trial_{trial}', f'mov_{mov}',
                        f'background_noise_{noise}', sound)


def iter_existing_recordings(dataset_path):
    """Return the real non-empty dataset recordings in canonical evaluation order."""
    recordings = []
    for subj_id in get_subject_ids(dataset_path):
        for trial in TRIALS:
            for mov in MOVEMENTS:
                for noise in NOISES:
                    for sound in SOUNDS:
                        rec_path = _recording_path(dataset_path, subj_id, trial, mov, noise, sound)
                        if os.path.isdir(rec_path) and os.listdir(rec_path):
                            recordings.append((subj_id, trial, mov, noise, sound))
    return recordings


def evaluate_subjects(dataset_path):
    """Evaluate all recordings. Backs up and restores main.h."""
    backup_main_h()
    all_results = []
    recordings = iter_existing_recordings(dataset_path)
    print(f"Expected recordings: {len(recordings)}")

    try:
        current_subject = None
        for subj_id, trial, mov, noise, sound in recordings:
            if subj_id != current_subject:
                current_subject = subj_id
                print(f"\n=== Subject {subj_id} ===")
            rec_id = f"t{trial}_{mov}_{noise}_{sound}"
            result = evaluate_recording(
                subj_id, trial, mov, noise, sound,
                dataset_path, INPUT_DATA_DIR)
            all_results.append(result)
            print(f"  {rec_id}: TP_evt={result['tp_evt']} "
                  f"FP_evt={result['fp_evt']} FN_evt={result['fn_evt']}")
    finally:
        restore_main_h()

    if len(all_results) != len(recordings):
        raise EvaluationError(f"processed {len(all_results)} of {len(recordings)} expected recordings")

    return all_results


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

# ──────────────────────────────────────────────
#  FxP-vs-float kernel error mode
# ──────────────────────────────────────────────

def _fxp_build_tag(*parts):
    """Create a compact unique tag for per-recording harness build artifacts."""
    raw = "__".join(str(part) for part in parts)
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{os.getpid()}_{digest}"


def _cleanup_fxp_harness(binary_path, build_tag):
    """Remove per-recording harness artifacts after the harness has run."""
    if binary_path:
        try:
            os.remove(binary_path)
        except FileNotFoundError:
            pass
        shutil.rmtree(f"{binary_path}.dSYM", ignore_errors=True)
    if build_tag:
        shutil.rmtree(os.path.join(FXP_DIR, ".build", build_tag), ignore_errors=True)


def _compile_fxp_error_harness(audio_relpath, imu_relpath, twiddle):
    """Recompile the FxP error harness against a specific recording's headers."""
    tag = _fxp_build_tag("stage", twiddle, audio_relpath, imu_relpath)
    target = f"fxp_stage_harness_{tag}"
    header_flags = (
        f"-include input_data/{audio_relpath} "
        f"-include input_data/{imu_relpath}"
    )
    cmd = [
        "make", "-B", "-C", FXP_DIR, target,
        f"TARGET_STAGE={target}",
        f"BUILD_TAG={tag}",
        f"FFT_MODE=-DFIXED_POINT={twiddle}",
        f"HEADER_FLAGS={header_flags}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr, os.path.join(FXP_DIR, ".build", tag, target), tag


def _compile_fxp_progressive_harness(audio_relpath, imu_relpath, bio_relpath, twiddle):
    """Build the mixed float/FxP progressive feature-block harness."""
    tag = _fxp_build_tag("progressive", twiddle, audio_relpath, imu_relpath, bio_relpath)
    target = f"fxp_progressive_harness_{tag}"
    header_flags = (
        f"-DPROGRESSIVE_INPUTS_INJECTED "
        f"-include input_data/{audio_relpath} "
        f"-include input_data/{imu_relpath} "
        f"-include input_data/{bio_relpath}"
    )
    cmd = [
        "make", "-B", "-C", FXP_DIR, target,
        f"TARGET_PROGRESSIVE={target}",
        f"BUILD_TAG={tag}",
        f"FFT_MODE=-DFIXED_POINT={twiddle}",
        f"HEADER_FLAGS={header_flags}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr, os.path.join(FXP_DIR, ".build", tag, target), tag


def _parse_fxp_kernel_acc(output):
    """Parse FXP_KERNEL_ACC lines into {(block, kernel): accumulator dict}."""
    rows = {}
    for line in output.splitlines():
        if not line.startswith("FXP_KERNEL_ACC,"):
            continue
        kv = {}
        for part in line.split(",")[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        block = kv.get("block", "")
        kernel = kv.get("kernel", "")
        rows[(block, kernel)] = {
            "n": int(kv.get("n", "0")),
            "sum_sq_err": float(kv.get("sum_sq_err", "0")),
            "sum_sq_ref": float(kv.get("sum_sq_ref", "0")),
            "max_abs_err": float(kv.get("max_abs_err", "0")),
            "max_abs_ref": float(kv.get("max_abs_ref", "0")),
            "nonfinite": int(kv.get("nonfinite", "0")),
        }
    return rows


def _merge_acc(into, frm):
    """Sum partial accumulators (per recording) into a running total."""
    for key, src in frm.items():
        dst = into.setdefault(key, {
            "n": 0, "sum_sq_err": 0.0, "sum_sq_ref": 0.0,
            "max_abs_err": 0.0, "max_abs_ref": 0.0, "nonfinite": 0,
        })
        dst["n"] += src["n"]
        dst["sum_sq_err"] += src["sum_sq_err"]
        dst["sum_sq_ref"] += src["sum_sq_ref"]
        dst["nonfinite"] += src.get("nonfinite", 0)
        if src["max_abs_err"] > dst["max_abs_err"]:
            dst["max_abs_err"] = src["max_abs_err"]
        if src["max_abs_ref"] > dst["max_abs_ref"]:
            dst["max_abs_ref"] = src["max_abs_ref"]


def _acc_to_metrics(acc):
    """Convert raw accumulator state to percent error metrics.

    RelL2% is normalized by reference energy. RMSE% and MaxAbs% share the
    max-reference denominator, so RMSE% is guaranteed to be <= MaxAbs%.
    """
    import math

    if acc["n"] <= 0 or acc["sum_sq_ref"] <= 0:
        rel_l2_pct = 0.0
    else:
        rel_l2_pct = 100.0 * math.sqrt(acc["sum_sq_err"] / acc["sum_sq_ref"])

    if acc["max_abs_ref"] > 0:
        rmse_pct = 100.0 * math.sqrt(acc["sum_sq_err"] / acc["n"]) / acc["max_abs_ref"]
        max_abs_pct = 100.0 * acc["max_abs_err"] / acc["max_abs_ref"]
    else:
        rmse_pct = 0.0
        max_abs_pct = 0.0
    return rel_l2_pct, rmse_pct, max_abs_pct


def _print_fxp_table(table, header):
    """Print kernel error rows grouped by block (audio first, then imu)."""
    print(header)
    for block in ("audio", "imu"):
        kernels = sorted(k for (b, k) in table if b == block)
        for kernel in kernels:
            rel_l2_pct, rmse_pct, max_abs_pct = _acc_to_metrics(table[(block, kernel)])
            nonfinite = table[(block, kernel)].get("nonfinite", 0)
            suffix = f"  NonFinite={nonfinite}" if nonfinite else ""
            print(f"  {block:<5}  {kernel:<22}  RelL2%={rel_l2_pct:7.3f}  "
                  f"RMSE%={rmse_pct:7.3f}  MaxAbs%={max_abs_pct:7.3f}{suffix}")


def _evaluate_fxp_errors(dataset_path, twiddle):
    """Loop the dataset, compile the FxP harness per recording, accumulate kernel errors."""
    overall = {}
    per_subject = {}
    n_recordings = 0
    total_duration = 0.0
    recordings = iter_existing_recordings(dataset_path)
    print(f"Expected recordings: {len(recordings)}")

    current_subject = None
    subj_acc = {}
    for subj_id, trial, mov, noise, sound in recordings:
        if subj_id != current_subject:
            if current_subject is not None:
                per_subject[current_subject] = subj_acc
                _print_fxp_table(subj_acc, f"\nSubject {current_subject}:")
            current_subject = subj_id
            subj_acc = {}
            print(f"\n=== Subject {subj_id} ===", flush=True)

        rec_id = f"t{trial}_{mov}_{noise}_{sound}"
        result = transform_recording(subj_id, trial, mov, noise, sound,
                                     dataset_path, INPUT_DATA_DIR)
        if result is None:
            raise EvaluationError(f"transform failed for {subj_id} {rec_id}")
        suffix, audio_relpath, imu_relpath, _ = result

        ok, err, harness_bin, build_tag = _compile_fxp_error_harness(audio_relpath, imu_relpath, twiddle)
        if not ok:
            print(f"  {rec_id}: harness compile FAILED\n{err}", flush=True)
            _cleanup_fxp_harness(harness_bin, build_tag)
            raise EvaluationError(f"FxP error harness compile failed for {suffix}")

        try:
            run = subprocess.run([harness_bin], capture_output=True,
                                 text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"  {rec_id}: TIMEOUT", flush=True)
            raise EvaluationError(f"FxP error harness timed out for {suffix}") from None
        finally:
            _cleanup_fxp_harness(harness_bin, build_tag)

        if run.returncode != 0:
            print(f"  {rec_id}: harness run FAILED\n{run.stderr}", flush=True)
            raise EvaluationError(f"FxP error harness run failed for {suffix}")

        rows = _parse_fxp_kernel_acc(run.stdout)
        if not rows:
            print(f"  {rec_id}: no FXP_KERNEL_ACC output", flush=True)
            raise EvaluationError(f"FxP error harness produced no kernel metrics for {suffix}")

        _merge_acc(subj_acc, rows)
        _merge_acc(overall, rows)
        n_recordings += 1
        total_duration += get_recording_duration(dataset_path, subj_id,
                                                 trial, mov, noise, sound)
        n_kernels = len(rows)
        print(f"  {rec_id}: {n_kernels} kernels accumulated", flush=True)

    if current_subject is not None:
        per_subject[current_subject] = subj_acc
        _print_fxp_table(subj_acc, f"\nSubject {current_subject}:")

    if n_recordings != len(recordings):
        raise EvaluationError(f"processed {n_recordings} of {len(recordings)} expected recordings")

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"OVERALL ({n_recordings} recordings, {total_duration / 3600.0:.3f} hrs)")
    print(bar)
    if overall:
        _print_fxp_table(overall, "Dataset-wide kernel error:")
    else:
        print("  No FXP kernel accumulators were collected.")


FXP_BLOCKS = [
    "audio_fft", "audio_psd", "audio_mel", "audio_scalar", "audio_all",
    "imu_raw", "imu_l2", "imu_all",
]


def _evaluate_progressive_recording(subj_id, trial, mov, noise, sound,
                                    dataset_path, input_data_dir, fxp_block, twiddle):
    result = transform_recording(subj_id, trial, mov, noise, sound,
                                 dataset_path, input_data_dir)
    if result is None:
        raise EvaluationError(f"transform failed for {subj_id} t{trial} {mov} {noise} {sound}")

    suffix, audio_relpath, imu_relpath, bio_relpath = result
    ok, err, harness_bin, build_tag = _compile_fxp_progressive_harness(audio_relpath, imu_relpath, bio_relpath, twiddle)
    if not ok:
        print(f"  FAILED to compile progressive harness for {suffix}:\n{err}")
        _cleanup_fxp_harness(harness_bin, build_tag)
        raise EvaluationError(f"progressive harness compile failed for {suffix}")

    try:
        run = subprocess.run([harness_bin, fxp_block],
                             capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {suffix}")
        raise EvaluationError(f"progressive harness timed out for {suffix}") from None
    finally:
        _cleanup_fxp_harness(harness_bin, build_tag)
    if run.returncode != 0:
        print(f"  progressive harness failed for {suffix}:\n{run.stderr}")
        raise EvaluationError(f"progressive harness failed for {suffix}")

    pred_segments = parse_c_output(run.stdout)
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


def _evaluate_progressive_subjects(dataset_path, fxp_block, twiddle):
    all_results = []
    recordings = iter_existing_recordings(dataset_path)
    print(f"Expected recordings: {len(recordings)}")
    current_subject = None
    for subj_id, trial, mov, noise, sound in recordings:
        if subj_id != current_subject:
            current_subject = subj_id
            print(f"\n=== Subject {subj_id} ===")
        rec_id = f"t{trial}_{mov}_{noise}_{sound}"
        result = _evaluate_progressive_recording(
            subj_id, trial, mov, noise, sound,
            dataset_path, INPUT_DATA_DIR, fxp_block, twiddle)
        all_results.append(result)
        print(f"  {rec_id}: TP_evt={result['tp_evt']} "
              f"FP_evt={result['fp_evt']} FN_evt={result['fn_evt']}")
    if len(all_results) != len(recordings):
        raise EvaluationError(f"processed {len(all_results)} of {len(recordings)} expected recordings")
    return all_results


def _run_progressive_eval(fxp_block, twiddle):
    print(f"Using progressive FxP feature block: {fxp_block} (twiddle={twiddle})")
    results = _evaluate_progressive_subjects(DEFAULT_DATASET_PATH, fxp_block, twiddle)
    if not results:
        print("No recordings processed. Check the default dataset path.")
        sys.exit(1)

    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(output_dir, f"results_fxp_block_{fxp_block}.csv"))
    save_summary_json(aggregate, os.path.join(output_dir, f"summary_fxp_block_{fxp_block}.json"),
                      build_per_subject_json(per_subject))

    return aggregate


def _compile_flags_for_mode(mode, twiddle):
    if mode == "float":
        return ""
    return f"-DFXP_MODE -DFIXED_POINT={twiddle}"


def _run_mode_eval(mode, twiddle):
    compile_flags = _compile_flags_for_mode(mode, twiddle)
    evaluate_recording._extra_flags = compile_flags

    if mode == "float":
        print("Using float mode compile flags")
    else:
        print(f"Using fxp mode compile flags: {compile_flags}")

    results = evaluate_subjects(DEFAULT_DATASET_PATH)
    if not results:
        print("No recordings processed. Check the default dataset path.")
        sys.exit(1)

    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    save_results_csv(results, os.path.join(output_dir, f"results_{mode}.csv"))
    save_summary_json(aggregate, os.path.join(output_dir, f"summary_{mode}.json"),
                      build_per_subject_json(per_subject))

    return aggregate


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Evaluate Cough-E ML metrics in float or FxP mode."
    )
    parser.add_argument("--mode", choices=["float", "fxp", "fxp-error"], default="float",
                        help="float / fxp ML metrics, or fxp-error for kernel-level FxP-vs-float error metrics")
    parser.add_argument("--twiddle", type=int, choices=[16, 32], default=32,
                        help="KissFFT twiddle precision (used in fxp / fxp-error modes)")
    parser.add_argument("--fxp-block", choices=FXP_BLOCKS,
                        help="run mixed float/FxP ML metrics with only this feature block replaced by FxP outputs")

    args = parser.parse_args(argv)
    try:
        if args.fxp_block:
            if args.mode != "float":
                parser.error("--fxp-block is an evaluation mode by itself; do not combine it with --mode fxp or --mode fxp-error")
            _run_progressive_eval(args.fxp_block, args.twiddle)
        elif args.mode == "fxp-error":
            _evaluate_fxp_errors(DEFAULT_DATASET_PATH, args.twiddle)
        else:
            _run_mode_eval(args.mode, args.twiddle)
    except EvaluationError as exc:
        print(f"\nEvaluation aborted: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
