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
import json
import os
import re
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
FXP_ERROR_HARNESS_BIN = os.path.join(FXP_DIR, "fxp_stage_harness")

# Original main.h content for backup/restore
MAIN_H_ORIGINAL = None

CSV_FIELDNAMES = [
    "subject", "trial", "movement", "noise", "sound",
    "tp_evt", "fp_evt", "fn_evt", "se_evt", "ppv_evt", "f1_evt",
    "duration",
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


def get_subject_ids(dataset_path):
    """Return ordered subject IDs for the dataset."""
    return sorted([
        s for s in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, s))
    ])


def evaluate_subjects(dataset_path):
    """Evaluate all recordings. Backs up and restores main.h."""
    backup_main_h()
    all_results = []

    try:
        for subj_id in get_subject_ids(dataset_path):
            print(f"\n=== Subject {subj_id} ===")
            for trial in TRIALS:
                for mov in MOVEMENTS:
                    for noise in NOISES:
                        for sound in SOUNDS:
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

def _compile_fxp_error_harness(audio_relpath, imu_relpath, twiddle):
    """Recompile the FxP error harness against a specific recording's headers."""
    header_flags = (
        f"-include input_data/{audio_relpath} "
        f"-include input_data/{imu_relpath}"
    )
    cmd = [
        "make", "-B", "-C", FXP_DIR, "fxp_stage_harness",
        f"FFT_MODE=-DFIXED_POINT={twiddle}",
        f"HEADER_FLAGS={header_flags}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


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
        }
    return rows


def _merge_acc(into, frm):
    """Sum partial accumulators (per recording) into a running total."""
    for key, src in frm.items():
        dst = into.setdefault(key, {
            "n": 0, "sum_sq_err": 0.0, "sum_sq_ref": 0.0,
            "max_abs_err": 0.0, "max_abs_ref": 0.0,
        })
        dst["n"] += src["n"]
        dst["sum_sq_err"] += src["sum_sq_err"]
        dst["sum_sq_ref"] += src["sum_sq_ref"]
        if src["max_abs_err"] > dst["max_abs_err"]:
            dst["max_abs_err"] = src["max_abs_err"]
        if src["max_abs_ref"] > dst["max_abs_ref"]:
            dst["max_abs_ref"] = src["max_abs_ref"]


def _acc_to_metrics(acc):
    """Convert raw accumulator state to (rmse_pct, max_abs_pct)."""
    if acc["n"] <= 0 or acc["sum_sq_ref"] <= 0:
        rmse_pct = 0.0
    else:
        import math
        rmse_pct = 100.0 * math.sqrt(acc["sum_sq_err"] / acc["sum_sq_ref"])
    if acc["max_abs_ref"] > 0:
        max_abs_pct = 100.0 * acc["max_abs_err"] / acc["max_abs_ref"]
    else:
        max_abs_pct = 0.0
    return rmse_pct, max_abs_pct


def _print_fxp_table(table, header):
    """Print kernel error rows grouped by block (audio first, then imu)."""
    print(header)
    for block in ("audio", "imu"):
        kernels = sorted(k for (b, k) in table if b == block)
        for kernel in kernels:
            rmse_pct, max_abs_pct = _acc_to_metrics(table[(block, kernel)])
            print(f"  {block:<5}  {kernel:<22}  RMSE%={rmse_pct:7.3f}  MaxAbs%={max_abs_pct:7.3f}")


def _evaluate_fxp_errors(dataset_path, twiddle):
    """Loop the dataset, compile the FxP harness per recording, accumulate kernel errors."""
    overall = {}
    per_subject = {}
    n_recordings = 0
    total_duration = 0.0

    for subj_id in get_subject_ids(dataset_path):
        print(f"\n=== Subject {subj_id} ===", flush=True)
        subj_acc = {}
        for trial in TRIALS:
            for mov in MOVEMENTS:
                for noise in NOISES:
                    for sound in SOUNDS:
                        rec_id = f"t{trial}_{mov}_{noise}_{sound}"
                        result = transform_recording(subj_id, trial, mov, noise, sound,
                                                     dataset_path, INPUT_DATA_DIR)
                        if result is None:
                            continue
                        suffix, audio_relpath, imu_relpath, _ = result

                        ok, err = _compile_fxp_error_harness(audio_relpath, imu_relpath, twiddle)
                        if not ok:
                            print(f"  {rec_id}: harness compile FAILED\n{err}", flush=True)
                            continue

                        try:
                            run = subprocess.run([FXP_ERROR_HARNESS_BIN], capture_output=True,
                                                 text=True, timeout=120)
                        except subprocess.TimeoutExpired:
                            print(f"  {rec_id}: TIMEOUT", flush=True)
                            continue

                        rows = _parse_fxp_kernel_acc(run.stdout)
                        if not rows:
                            print(f"  {rec_id}: no FXP_KERNEL_ACC output", flush=True)
                            continue

                        _merge_acc(subj_acc, rows)
                        _merge_acc(overall, rows)
                        n_recordings += 1
                        total_duration += get_recording_duration(dataset_path, subj_id,
                                                                 trial, mov, noise, sound)
                        n_kernels = len(rows)
                        print(f"  {rec_id}: {n_kernels} kernels accumulated", flush=True)

        per_subject[subj_id] = subj_acc
        if subj_acc:
            _print_fxp_table(subj_acc, f"\nSubject {subj_id}:")

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"OVERALL ({n_recordings} recordings, {total_duration / 3600.0:.3f} hrs)")
    print(bar)
    if overall:
        for block in ("audio", "imu"):
            kernels = sorted(k for (b, k) in overall if b == block)
            for kernel in kernels:
                rmse_pct, max_abs_pct = _acc_to_metrics(overall[(block, kernel)])
                print(f"  {block:<5}  {kernel:<22}  RMSE%={rmse_pct:7.3f}  MaxAbs%={max_abs_pct:7.3f}")
    print(bar)


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

    args = parser.parse_args(argv)
    if args.mode == "fxp-error":
        _evaluate_fxp_errors(DEFAULT_DATASET_PATH, args.twiddle)
    else:
        _run_mode_eval(args.mode, args.twiddle)


if __name__ == "__main__":
    main()
