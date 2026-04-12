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
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from timescoring.annotations import Annotation
from timescoring import scoring

import numpy as np

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


def evaluate_subjects(dataset_path, subjects=None,
                      trials=TRIALS, movements=MOVEMENTS,
                      noises=NOISES, sounds=SOUNDS):
    """Evaluate all recordings for the given subjects. Backs up and restores main.h."""
    if subjects is None:
        subjects = sorted([
            s for s in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, s))
        ])

    backup_main_h()
    all_results = []

    try:
        for subj_id in subjects:
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


# ──────────────────────────────────────────────
#  CLI subcommands
# ──────────────────────────────────────────────

def cmd_transform(args):
    """Generate dataset only."""
    transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)


def cmd_run(args):
    """Run evaluation (transforms dataset if needed)."""
    output_dir = args.output_dir or os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)

    if getattr(args, "fxp", False):
        evaluate_recording._extra_flags = "-DFXP_MODE -DFIXED_POINT=16"
        print("FxP mode enabled (-DFXP_MODE -DFIXED_POINT=16)")
    else:
        evaluate_recording._extra_flags = ""

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
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset_path

    print("=== Float evaluation ===")
    evaluate_recording._extra_flags = ""
    float_results = evaluate_subjects(dataset_path, subjects=args.subjects,
                                      sounds=args.sounds, noises=args.noises)
    float_agg = compute_aggregate_metrics(float_results)

    print("\n=== FxP evaluation ===")
    evaluate_recording._extra_flags = "-DFXP_MODE -DFIXED_POINT=16"
    fxp_results = evaluate_subjects(dataset_path, subjects=args.subjects,
                                    sounds=args.sounds, noises=args.noises)
    fxp_agg = compute_aggregate_metrics(fxp_results)

    # Save individual CSVs
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
    cmp_path = os.path.join(output_dir, "comparison_float_vs_fxp.json")
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {cmp_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Cough-E C application against full_dataset_test")
    subparsers = parser.add_subparsers(dest="command")

    def add_common_args(p, fxp_flag=False):
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
                           help="Compile with -DFXP_MODE (fixed-point kernels)")

    p_transform = subparsers.add_parser("transform", help="Generate C headers from dataset")
    add_common_args(p_transform)
    p_transform.set_defaults(func=cmd_transform)

    p_run = subparsers.add_parser("run", help="Run evaluation")
    add_common_args(p_run, fxp_flag=True)
    p_run.set_defaults(func=cmd_run)

    p_agg = subparsers.add_parser("aggregate", help="Compute metrics from existing CSV")
    p_agg.add_argument("--csv", type=str, required=True, help="Path to results CSV")
    p_agg.set_defaults(func=cmd_aggregate)

    p_full = subparsers.add_parser("full", help="Transform dataset + run evaluation")
    add_common_args(p_full, fxp_flag=True)
    p_full.set_defaults(func=cmd_full)

    p_compare = subparsers.add_parser("compare",
                                      help="Run float and FxP evaluations side-by-side")
    add_common_args(p_compare)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    # Default to 'compare' when no subcommand given
    if args.command is None:
        args = parser.parse_args(["compare"])

    args.func(args)


if __name__ == "__main__":
    main()
