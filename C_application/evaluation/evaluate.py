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
    python C_application/evaluation/evaluate.py --compare
    python C_application/evaluation/evaluate.py --mode fxp --subjects 14287 14342 --sounds cough laugh

FxP error and regression harnesses live in:
    python C_application/evaluation/fxp/fxp_harness.py ...
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
        transform_recording, transform_all, make_recording_suffix,
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
    transform_all = _missing_transform_dependency
    make_recording_suffix = _missing_transform_dependency
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
            "Install dependencies or run evaluation/fxp/fxp_harness.py for FxP error metrics."
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
    skip_transform = getattr(evaluate_recording, "_skip_transform", False)

    if skip_transform:
        suffix = f"{subj_id}_t{trial}_{mov}_{noise}_{sound}"
        audio_relpath = f"{subj_id}/audio_input_{suffix}.h"
        imu_relpath = f"{subj_id}/imu_input_{suffix}.h"
        bio_relpath = f"{subj_id}/bio_input_{subj_id}.h"

        audio_abs = os.path.join(input_data_dir, audio_relpath)
        imu_abs = os.path.join(input_data_dir, imu_relpath)
        bio_abs = os.path.join(input_data_dir, bio_relpath)
        if not (os.path.exists(audio_abs) and os.path.exists(imu_abs) and os.path.exists(bio_abs)):
            print(f"  SKIP {suffix}: --skip-transform set but required headers are missing")
            return None
    else:
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



COMPAT_COMMANDS = {
    "transform",
    "run",
    "aggregate",
    "full",
    "compare",
}


def _compile_flags_for_mode(mode, twiddle):
    if mode == "float":
        return ""
    return f"-DFXP_MODE -DFIXED_POINT={twiddle}"


def _run_mode_eval(args, mode, save_outputs=True):
    compile_flags = _compile_flags_for_mode(mode, args.twiddle)
    evaluate_recording._extra_flags = compile_flags

    if mode == "float":
        print("Using float mode compile flags")
    else:
        print(f"Using fxp mode compile flags: {compile_flags}")

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

    if save_outputs and not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_results_csv(results, os.path.join(args.output_dir, "results.csv"))
        save_summary_json(aggregate, os.path.join(args.output_dir, "summary.json"),
                          build_per_subject_json(per_subject))

    return aggregate


def _print_compare_table(float_agg, fxp_agg):
    fmt = "  {:<10} {:>8} {:>8} {:>8} {:>10}"
    print("\n" + "=" * 70)
    print("FLOAT vs FxP COMPARISON")
    print("=" * 70)
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
    print(fmt.format("Delta",
                     f"{(fxp_agg['se_evt'] - float_agg['se_evt']):+.4f}",
                     f"{(fxp_agg['pr_evt'] - float_agg['pr_evt']):+.4f}",
                     f"{(fxp_agg['f1_evt'] - float_agg['f1_evt']):+.4f}",
                     f"{(fxp_agg['fphr_evt'] - float_agg['fphr_evt']):+.1f}"))
    print("=" * 70)


def _execute_unified(args, user_set_twiddle=False):
    if user_set_twiddle and args.mode == "float":
        print("Warning: --twiddle is ignored when --mode float is selected.")

    evaluate_recording._skip_transform = bool(args.skip_transform)

    if not args.skip_transform:
        transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)

    if args.compare:
        print("=== Float evaluation ===")
        float_agg = _run_mode_eval(args, mode="float", save_outputs=False)
        print("\n=== FxP evaluation ===")
        fxp_agg = _run_mode_eval(args, mode="fxp", save_outputs=False)
        _print_compare_table(float_agg, fxp_agg)

        if not args.no_save:
            print("Compare mode is console-only by default; no compare CSV/JSON artifacts were written.")
        return

    _run_mode_eval(args, mode=args.mode, save_outputs=True)


def _run_compat_cli(argv):
    print("DEPRECATION: positional evaluate.py subcommands are compatibility shims.", flush=True)
    print("Please migrate to: python evaluation/evaluate.py [--mode float|fxp] [--twiddle 16|32] [flags]", flush=True)
    print("For FxP error/regression metrics use: python evaluation/fxp/fxp_harness.py ...", flush=True)

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "aggregate":
        p = argparse.ArgumentParser(prog="evaluate.py aggregate")
        p.add_argument("--csv", type=str, required=True, help="Path to results CSV")
        args = p.parse_args(rest)
        results = load_results_csv(args.csv)
        aggregate = compute_aggregate_metrics(results)
        per_subject = print_results(results, aggregate)
        output_dir = os.path.dirname(args.csv) or "."
        save_summary_json(aggregate, os.path.join(output_dir, "summary.json"),
                          build_per_subject_json(per_subject))
        return

    p = argparse.ArgumentParser(prog=f"evaluate.py {cmd}")
    p.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--subjects", nargs="+", type=str, default=None)
    p.add_argument("--sounds", nargs="+", type=str, default=SOUNDS)
    p.add_argument("--noises", nargs="+", type=str, default=NOISES)
    p.add_argument("--output_dir", type=str, default=os.path.dirname(__file__))
    p.add_argument("--no-save", action="store_true", default=False)
    p.add_argument("--fxp", action="store_true", default=False)
    p.add_argument("--kissfft-fixed", type=int, choices=[16, 32], default=None)
    args = p.parse_args(rest)

    if cmd == "transform":
        transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)
        return

    mode = "fxp" if args.fxp else "float"
    twiddle = args.kissfft_fixed if args.kissfft_fixed is not None else 32
    namespace = argparse.Namespace(
        mode=mode,
        twiddle=twiddle,
        subjects=args.subjects,
        sounds=args.sounds,
        noises=args.noises,
        dataset_path=args.dataset_path,
        compare=(cmd == "compare"),
        output_dir=args.output_dir,
        no_save=args.no_save,
        skip_transform=(cmd in {"run", "compare"}),
    )
    _execute_unified(namespace, user_set_twiddle=(args.kissfft_fixed is not None))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in COMPAT_COMMANDS:
        _run_compat_cli(argv)
        return

    parser = argparse.ArgumentParser(
        description="Unified evaluator CLI for float/FxP Cough-E runtime"
    )
    parser.add_argument("--mode", choices=["float", "fxp"], default="float",
                        help="Pipeline precision mode (default: float)")
    parser.add_argument("--twiddle", type=int, choices=[16, 32], default=32,
                        help="KissFFT twiddle precision (used only in fxp mode)")
    parser.add_argument("--subjects", nargs="+", type=str, default=None,
                        help="Specific subject IDs (default: all)")
    parser.add_argument("--sounds", nargs="+", type=str, default=SOUNDS)
    parser.add_argument("--noises", nargs="+", type=str, default=NOISES)
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=DEFAULT_DATASET_PATH,
                        help=f"Path to full_dataset_test (default: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run float and fxp sequentially and print delta in console")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=os.path.dirname(__file__),
                        help="Output directory for CSV/JSON in single-mode runs")
    parser.add_argument("--no-save", action="store_true", default=False,
                        help="Print metrics only; do not write CSV/JSON outputs")
    parser.add_argument("--skip-transform", action="store_true", default=False,
                        help="Skip WAV->header regeneration for iterative runs")

    args = parser.parse_args(argv)
    _execute_unified(args, user_set_twiddle=("--twiddle" in argv))


if __name__ == "__main__":
    main()
