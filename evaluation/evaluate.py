"""
Evaluation pipeline for the Cough-E C application.

Pipeline per recording:
1. Generate C header files (via transform_dataset.py)
2. Update main.h includes to point to the generated headers
3. Compile the C application
4. Run the C application and capture output
5. Parse COUGH_SEG lines to get detected cough segment boundaries
6. Compare with ground truth using event-based scoring

Usage:
    # Generate dataset only
    python evaluate.py transform --dataset_path /path/to/public_dataset

    # Run evaluation (generates dataset if needed)
    python evaluate.py run --dataset_path /path/to/public_dataset

    # Aggregate metrics from an existing CSV (no re-run)
    python evaluate.py aggregate --csv evaluation/results.csv

    # Generate dataset + run evaluation in one command
    python evaluate.py full --dataset_path /path/to/public_dataset
"""

import numpy as np
import json
import os
import subprocess
import re
import argparse
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add parent path for imports
sys.path.insert(0, os.path.dirname(__file__))
from transform_dataset import (
    transform_recording, transform_all, make_recording_suffix,
    AUDIO_FS_TARGET, IMU_FS, SOUNDS, NOISES, MOVEMENTS, TRIALS
)

# Scoring parameters matching ML_methodology/config/scoring/default.yaml
TOLERANCE_START = 0.25
TOLERANCE_END = 0.25
MIN_COUGH_DURATION = 0.1
MAX_EVENT_DURATION = 0.6
MIN_DURATION_BTWN_EVENTS = 0
WINDOW_LEN = 0.8  # audio window length in seconds

# C application paths (relative to repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
C_APP_DIR = os.path.join(REPO_ROOT, "C_application")
MAIN_H_PATH = os.path.join(C_APP_DIR, "main.h")
INPUT_DATA_DIR = os.path.join(C_APP_DIR, "input_data")
BUILD_DIR = os.path.join(C_APP_DIR, "build")
EXECUTABLE = os.path.join(BUILD_DIR, "cough-e")

# Original main.h content for restoring
MAIN_H_ORIGINAL = None


# ──────────────────────────────────────────────
#  main.h management
# ──────────────────────────────────────────────

def backup_main_h():
    global MAIN_H_ORIGINAL
    with open(MAIN_H_PATH, 'r') as f:
        MAIN_H_ORIGINAL = f.read()


def restore_main_h():
    if MAIN_H_ORIGINAL is not None:
        with open(MAIN_H_PATH, 'w') as f:
            f.write(MAIN_H_ORIGINAL)


def update_main_h(audio_relpath, imu_relpath, bio_relpath):
    """
    Rewrite main.h to include the specified input data headers.
    Relpaths are relative to input_data/ (e.g. "{subj_id}/audio_input_{suffix}.h").
    """
    main_h_content = f"""#ifndef _MAIN_H_
#define _MAIN_H_

#include <inttypes.h>

//////////////////////////////////////
/* Model to be used                 */
//////////////////////////////////////
#include <audio_model.h>
#include <imu_model.h>
//////////////////////////////////////


//////////////////////////////////////
/* Input data                       */
// ///////////////////////////////////
#include <input_data/{audio_relpath}>
#include <input_data/{imu_relpath}>
#include <input_data/{bio_relpath}>
//////////////////////////////////////


#include <audio_features.h>
#include <imu_features.h>


/* Threshold for the audio model */
#define AUDIO_TH    0.3

/* Threshold for the imu model */
#define IMU_TH    0.05

// Defines (in seconds) how often to provide the final estimation (execute post-processing)
#define TIME_DEADLINE_OUTPUT    1.5

// Maximum number of consecutive windows to be run by AUDIO model
#define N_MAX_WIND_AUD  4

////////////////////////////////////////////////
/* Define if to run in multi or unimodal mode */
////////////////////////////////////////////////

// Execute in multimodal mode, using both modalities cooperating
#define RUN_MIXED

#ifndef RUN_MIXED
    // #define RUN_ONLY_AUD        // Use only the audio modality

    #ifndef RUN_ONLY_AUD
        #define RUN_ONLY_IMU    // Use only the imu modality
    #endif
#endif
////////////////////////////////////////////////


#endif
"""
    with open(MAIN_H_PATH, 'w') as f:
        f.write(main_h_content)


# ──────────────────────────────────────────────
#  Compile & run
# ──────────────────────────────────────────────

def compile_c_app():
    subprocess.run(["make", "-C", C_APP_DIR, "clean"],
                   capture_output=True, text=True)
    result = subprocess.run(["make", "-C", C_APP_DIR],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Compilation failed: {result.stderr}")
        return False
    return True


def run_c_app():
    result = subprocess.run([EXECUTABLE], capture_output=True, text=True, timeout=30)
    return result.stdout


# ──────────────────────────────────────────────
#  Output parsing
# ──────────────────────────────────────────────

def parse_c_output(output, audio_fs=AUDIO_FS_TARGET):
    """
    Parse C application output to extract detected cough segments.

    The C app's FSM resets when IMU data runs out (re-processing from the start).
    We detect this by grouping segments by postprocessing period and detecting
    when a period's segments match an earlier period (indicating FSM restart).
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

    # Detect reset: find first period whose segment set was seen before
    seen_signatures = set()
    first_pass_periods = []
    for period_segs in periods:
        sig = tuple(period_segs)
        if sig in seen_signatures and len(sig) > 0:
            break
        seen_signatures.add(sig)
        first_pass_periods.append(period_segs)

    # Flatten and deduplicate
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
#  Ground truth & masks
# ──────────────────────────────────────────────

def load_ground_truth(dataset_path, subj_id, trial, mov, noise, sound):
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
    import pandas as pd
    imu_path = os.path.join(dataset_path, subj_id,
                            f'trial_{trial}', f'mov_{mov}',
                            f'background_noise_{noise}', sound,
                            'imu.csv')
    if os.path.exists(imu_path):
        imu_df = pd.read_csv(imu_path)
        return len(imu_df) / IMU_FS
    return 0.0


def create_binary_mask(events, duration, fs=IMU_FS):
    """
    Create a binary mask array from event list, matching ML methodology's
    get_ground_truth_regions() function.
    """
    n_samples = int(round(duration * fs))
    mask = np.zeros(n_samples)
    for start, end in events:
        start_idx = int(round(start * fs))
        end_idx = int(round(end * fs))
        start_idx = max(0, min(start_idx, n_samples))
        end_idx = max(0, min(end_idx, n_samples))
        mask[start_idx:end_idx] = 1
        if start_idx > 0:
            mask[start_idx - 1] = 0
    return mask


# ──────────────────────────────────────────────
#  Scoring (event-based)
# ──────────────────────────────────────────────

def score_recording(gt_events, pred_events, duration):
    """
    Compute event-based scoring, mirroring test.py line 371.

    Returns dict with:
        Event-based (timescoring.EventScoring): tp_evt, fp_evt, fn_evt, se_evt, ppv_evt, f1_evt
    """
    fs = IMU_FS
    gt_mask = create_binary_mask(gt_events, duration, fs)
    pred_mask = create_binary_mask(pred_events, duration, fs)

    from timescoring.annotations import Annotation
    from timescoring import scoring

    labels = Annotation(gt_mask, fs)
    pred = Annotation(pred_mask, fs)

    # Event-based scoring (matches test.py line 371)
    param = scoring.EventScoring.Parameters(
        TOLERANCE_START, TOLERANCE_END,
        MIN_COUGH_DURATION / WINDOW_LEN,
        MAX_EVENT_DURATION, MIN_DURATION_BTWN_EVENTS
    )
    scores_evt = scoring.EventScoring(labels, pred, param)

    return {
        "tp_evt": scores_evt.tp,
        "fp_evt": scores_evt.fp,
        "fn_evt": scores_evt.refTrue - scores_evt.tp,
        "se_evt": scores_evt.sensitivity,
        "ppv_evt": scores_evt.precision,
        "f1_evt": scores_evt.f1,
    }


# ──────────────────────────────────────────────
#  Per-recording evaluation
# ──────────────────────────────────────────────

def evaluate_recording(subj_id, trial, mov, noise, sound,
                       dataset_path, input_data_dir):
    """
    Full pipeline for a single recording:
    transform -> update main.h -> compile -> run -> parse -> score
    """
    result = transform_recording(subj_id, trial, mov, noise, sound,
                                 dataset_path, input_data_dir)
    if result is None:
        return None

    suffix, audio_relpath, imu_relpath, bio_relpath = result

    update_main_h(audio_relpath, imu_relpath, bio_relpath)

    if not compile_c_app():
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
    scores["subject"] = subj_id
    scores["trial"] = trial
    scores["movement"] = mov
    scores["noise"] = noise
    scores["sound"] = sound
    scores["n_pred"] = len(pred_segments)
    scores["n_true"] = len(gt_events)
    scores["duration"] = duration

    return scores


def evaluate_subjects(dataset_path, subjects=None,
                      trials=TRIALS, movements=MOVEMENTS,
                      noises=NOISES, sounds=SOUNDS):
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
                                      f"FP_evt={result['fp_evt']} FN_evt={result['fn_evt']} "
                                      f"pred={result['n_pred']} true={result['n_true']}")
    finally:
        restore_main_h()

    return all_results


# ──────────────────────────────────────────────
#  Aggregation
# ──────────────────────────────────────────────

def compute_aggregate_metrics(results):
    """
    Compute aggregate metrics across all recordings.
    Returns event-based aggregation.
    """
    total_duration_hrs = sum(r["duration"] for r in results) / 3600.0

    # Event-based aggregation (matches report_results.py / get_evb_metrics_per_threshold)
    tp_evt = sum(r["tp_evt"] for r in results)
    fp_evt = sum(r["fp_evt"] for r in results)
    fn_evt = sum(r["fn_evt"] for r in results)
    se_evt = tp_evt / (tp_evt + fn_evt) if (tp_evt + fn_evt) > 0 else 0.0
    pr_evt = tp_evt / (tp_evt + fp_evt) if (tp_evt + fp_evt) > 0 else 0.0
    f1_evt = 2 * se_evt * pr_evt / (se_evt + pr_evt) if (se_evt + pr_evt) > 0 else 0.0
    fphr_evt = fp_evt / total_duration_hrs if total_duration_hrs > 0 else 0.0

    return {
        "se_evt": se_evt, "pr_evt": pr_evt, "f1_evt": f1_evt, "fphr_evt": fphr_evt,
        "tp_evt": tp_evt, "fp_evt": fp_evt, "fn_evt": fn_evt,
        "total_recordings": len(results),
        "total_duration_hrs": total_duration_hrs,
    }


# ──────────────────────────────────────────────
#  Output: CSV + JSON summary
# ──────────────────────────────────────────────

CSV_FIELDNAMES = [
    "subject", "trial", "movement", "noise", "sound",
    "tp_evt", "fp_evt", "fn_evt", "se_evt", "ppv_evt", "f1_evt",
    "n_pred", "n_true", "duration",
]


def save_results_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_FIELDNAMES})
    print(f"Results CSV saved to {output_path}")


def load_results_csv(csv_path):
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

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Summary JSON saved to {output_path}")


def print_results(results, aggregate):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Per-subject breakdown
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

    # Overall
    print(f"\n{'=' * 70}")
    print(f"OVERALL ({aggregate['total_recordings']} recordings, "
          f"{aggregate['total_duration_hrs']:.3f} hrs)")
    print(f"{'=' * 70}")
    print(f"  Event-based metrics (timescoring.EventScoring):")
    print(f"    SE    = {aggregate['se_evt']:.4f}")
    print(f"    PR    = {aggregate['pr_evt']:.4f}")
    print(f"    F1    = {aggregate['f1_evt']:.4f}")
    print(f"    FP/hr = {aggregate['fphr_evt']:.1f}")
    print(f"    TP={aggregate['tp_evt']}  FP={aggregate['fp_evt']}  FN={aggregate['fn_evt']}")

    print(f"\n  Python baseline (event-based): SE=0.71  PR=0.86  F1=0.78")
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
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)

    results = evaluate_subjects(
        args.dataset_path,
        subjects=args.subjects,
        sounds=args.sounds,
        noises=args.noises,
    )

    if len(results) == 0:
        print("No recordings processed. Check dataset path and subject IDs.")
        sys.exit(1)

    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    csv_path = os.path.join(output_dir, "results.csv")
    save_results_csv(results, csv_path)

    per_subj_json = {}
    for subj, a in per_subject.items():
        per_subj_json[subj] = {
            "event_based": {
                "SE": round(a["se_evt"], 4), "PR": round(a["pr_evt"], 4),
                "F1": round(a["f1_evt"], 4), "FP_hr": round(a["fphr_evt"], 1),
                "TP": a["tp_evt"], "FP": a["fp_evt"], "FN": a["fn_evt"],
            },
        }

    json_path = os.path.join(output_dir, "summary.json")
    save_summary_json(aggregate, json_path, per_subj_json)


def cmd_aggregate(args):
    """Compute aggregate metrics from an existing CSV."""
    results = load_results_csv(args.csv)
    aggregate = compute_aggregate_metrics(results)
    per_subject = print_results(results, aggregate)

    output_dir = os.path.dirname(args.csv) or "."
    per_subj_json = {}
    for subj, a in per_subject.items():
        per_subj_json[subj] = {
            "event_based": {
                "SE": round(a["se_evt"], 4), "PR": round(a["pr_evt"], 4),
                "F1": round(a["f1_evt"], 4), "FP_hr": round(a["fphr_evt"], 1),
                "TP": a["tp_evt"], "FP": a["fp_evt"], "FN": a["fn_evt"],
            },
        }
    json_path = os.path.join(output_dir, "summary.json")
    save_summary_json(aggregate, json_path, per_subj_json)


def cmd_full(args):
    """Generate dataset + run evaluation in one command."""
    print("Step 1/2: Transforming dataset...")
    transform_all(args.dataset_path, INPUT_DATA_DIR, args.subjects)
    print("\nStep 2/2: Running evaluation...")
    cmd_run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Cough-E C application against public dataset")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Shared args
    def add_common_args(p):
        p.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the public_dataset directory")
        p.add_argument("--subjects", nargs="+", type=str, default=None,
                        help="Specific subject IDs (default: all)")
        p.add_argument("--sounds", nargs="+", type=str, default=SOUNDS)
        p.add_argument("--noises", nargs="+", type=str, default=NOISES)
        p.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output files (default: evaluation/)")

    # transform
    p_transform = subparsers.add_parser("transform", help="Generate C headers from dataset")
    add_common_args(p_transform)
    p_transform.set_defaults(func=cmd_transform)

    # run
    p_run = subparsers.add_parser("run", help="Run evaluation (generates headers if needed)")
    add_common_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # aggregate
    p_agg = subparsers.add_parser("aggregate", help="Compute metrics from existing CSV")
    p_agg.add_argument("--csv", type=str, required=True,
                        help="Path to results CSV file")
    p_agg.set_defaults(func=cmd_aggregate)

    # full
    p_full = subparsers.add_parser("full", help="Transform dataset + run evaluation")
    add_common_args(p_full)
    p_full.set_defaults(func=cmd_full)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)
