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
    python C_application/evaluation/evaluate.py -j 4
    python C_application/evaluation/evaluate.py --mode fxp -j 8
    python C_application/evaluation/evaluate.py --mode fxp --twiddle 32
    python C_application/evaluation/evaluate.py --mode fxp-error
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import queue
import re
import shutil
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

TOLERANCE_START = 0.25
TOLERANCE_END = 0.25
MIN_COUGH_DURATION = 0.1
MAX_EVENT_DURATION = 0.6
MIN_DURATION_BTWN_EVENTS = 0
MIN_OVERLAP = MIN_COUGH_DURATION / 0.8

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C_APP_DIR = os.path.join(REPO_ROOT, "C_application")
INPUT_DATA_DIR = os.path.join(C_APP_DIR, "input_data")

DEFAULT_DATASET_PATH = os.path.join(REPO_ROOT, "Datasets", "full_dataset_test")
FXP_DIR = os.path.join(os.path.dirname(__file__), "fxp")

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

def update_main_h(ws_path, audio_relpath, imu_relpath, bio_relpath):
    """Replace the 3 input data #include lines in the workspace main.h."""
    main_h_path = os.path.join(ws_path, "main.h")
    with open(main_h_path, 'r') as f:
        content = f.read()

    content = re.sub(r'#include <input_data/.*audio_input.*\.h>',
                     f'#include <input_data/{audio_relpath}>', content)
    content = re.sub(r'#include <input_data/.*imu_input.*\.h>',
                     f'#include <input_data/{imu_relpath}>', content)
    content = re.sub(r'#include <input_data/.*bio_input.*\.h>',
                     f'#include <input_data/{bio_relpath}>', content)

    with open(main_h_path, 'w') as f:
        f.write(content)


# ──────────────────────────────────────────────
#  Compile & run
# ──────────────────────────────────────────────

def compile_c_app(ws_path, extra_flags=""):
    """Compile the C application with EVALUATION_MODE enabled in the workspace."""
    flags = "-DEVALUATION_MODE"
    if extra_flags:
        flags += " " + extra_flags
    result = subprocess.run(["nice", "-n", "9", "make", "-C", ws_path, f"CFLAGS={flags}", "-j8"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Compilation failed: {result.stderr}")
        return False
    return True


def run_c_app(ws_path):
    """Run the compiled C application from the workspace and return stdout."""
    executable = os.path.join(ws_path, "build", "cough-e")
    try:
        result = subprocess.run(["nice", "-n", "19", executable], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("    WARNING: C app timed out after 300s (possible stuck)", flush=True)
        return ""
    return result.stdout


# ──────────────────────────────────────────────
#  Output parsing
# ──────────────────────────────────────────────

def parse_c_output(output, audio_fs=AUDIO_FS_TARGET):
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

    seen_signatures = set()
    first_pass_periods = []
    for period_segs in periods:
        sig = tuple(period_segs)
        if sig in seen_signatures and len(sig) > 0:
            break
        seen_signatures.add(sig)
        first_pass_periods.append(period_segs)

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
    imu_path = os.path.join(dataset_path, subj_id,
                            f'trial_{trial}', f'mov_{mov}',
                            f'background_noise_{noise}', sound,
                            'imu.csv')
    if os.path.exists(imu_path):
        with open(imu_path, 'r') as f:
            n_lines = sum(1 for _ in f) - 1
        return n_lines / IMU_FS
    return 0.0


def create_binary_mask(events, duration):
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

def _resolve_mode_from_compile_flags(extra_flags):
    if "-DUPOS_MODE -DPOSIT_SIZE=16 -DNO_UPOS_QUIRE" in extra_flags:
        return "upos16nq"
    elif "-DUPOS_MODE -DPOSIT_SIZE=32 -DNO_UPOS_QUIRE" in extra_flags:
        return "upos32nq"
    elif "-DUPOS_MODE -DPOSIT_SIZE=16" in extra_flags:
        return "upos16q"
    elif "-DUPOS_MODE -DPOSIT_SIZE=32" in extra_flags:
        return "upos32q"
    elif "-DUPOS_MODE" in extra_flags:
        return "upos"
    elif "-DFXP_MODE" in extra_flags:
        return "fxp"
    else:
        return "float"

def evaluate_recording_isolated(subj_id, trial, mov, noise, sound,
                                dataset_path, ws_path, extra_flags=""):
    input_data_dir = os.path.join(ws_path, "input_data")

    result = transform_recording(subj_id, trial, mov, noise, sound,
                                 dataset_path, input_data_dir)
    if result is None:
        raise EvaluationError(f"transform failed for {subj_id} t{trial} {mov} {noise} {sound}")
    suffix, audio_relpath, imu_relpath, bio_relpath = result

    update_main_h(ws_path, audio_relpath, imu_relpath, bio_relpath)

    if not compile_c_app(ws_path, extra_flags=extra_flags):
        print(f"  FAILED to compile for {suffix}")
        raise EvaluationError(f"C application compile failed for {suffix}")

    output = run_c_app(ws_path)
    if not output:
        raise EvaluationError(f"C application produced no output (or timed out) for {suffix}")

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

    if "-DPRECISION_ANALYSIS" in extra_flags:
        # append so we get result for same subject on different precision setup
        output_dir = os.path.join(os.path.dirname(__file__), "precision_analysis")
        open(os.path.join(output_dir, f"{subj_id}_t{trial}_{mov}_{noise}_{sound}.log"), "a")\
            .write("\n".join([x for x in output.split("\n") if x.startswith("PRECISION")]))

    if "-DDEBUG" in extra_flags:
        mode = _resolve_mode_from_compile_flags(extra_flags)
        output_dir = os.path.join(os.path.join(os.path.dirname(__file__), "numerical_analysis"), mode)

        current_window = -1
        parsed_output = ""
        output_lines = output.split("\n")
        for i in range(len(output_lines) - 1):
            line = output_lines[i]
            m = re.search(r"PROCESS WINDOW (\d+) WITH MODEL (\d+)", line)
            current_window = int(m.group(1)) if m else current_window

            if line == "IMU FEATURES":
                parsed_output += f"IMU {current_window} {output_lines[i+1]}\n"
            elif line == "AUDIO FEATURES":
                parsed_output += f"AUDIO {current_window} {output_lines[i+1]}\n"

        open(os.path.join(output_dir, f"{subj_id}_t{trial}_{mov}_{noise}_{sound}.log"), "w")\
            .write(parsed_output)

    return scores


def _worker_task(task_args, dataset_path, workspaces, log_pa, log_na, extra_flags):
    subj_id, trial, mov, noise, sound = task_args
    ws_path = workspaces.get()
    try:
        if log_pa:
            extra_flags += f" -DPRECISION_ANALYSIS"
        if log_na:
            extra_flags += f" -DDEBUG"
        return evaluate_recording_isolated(
            subj_id, trial, mov, noise, sound,
            dataset_path, ws_path, extra_flags
        )
    finally:
        workspaces.put(ws_path)


def get_subject_ids(dataset_path):
    return sorted([
        s for s in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, s))
    ])


def _recording_path(dataset_path, subj_id, trial, mov, noise, sound):
    return os.path.join(dataset_path, subj_id,
                        f'trial_{trial}', f'mov_{mov}',
                        f'background_noise_{noise}', sound)


def iter_existing_recordings(dataset_path):
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


def _validate_workspace(ws_path):
    """List workspace contents and check the files needed to build/run are present."""
    if not os.path.isdir(ws_path):
        return False
    try:
        entries = set(os.listdir(ws_path))
    except OSError:
        return False
    required = {"main.h", "Makefile", "input_data"}
    return required.issubset(entries)


def evaluate_subjects(dataset_path, log_pa=False, log_na=False, \
                      jobs=1, extra_flags="", reuse_workspace=False):
    all_results = []
    recordings = iter_existing_recordings(dataset_path)
    print(f"Expected recordings: {len(recordings)}")

    temp_dirs = []
    workspaces = queue.Queue()

    print(f"\nSetting up {jobs} workspace(s) in /tmp...")
    try:
        for i in range(jobs):
            if reuse_workspace:
                ws_path = os.path.join("/tmp", f"cough_e_eval_worker_{i}")
            else:
                ws_path = tempfile.mkdtemp(dir="/tmp", prefix=f"cough_e_eval_worker_{i}_")

            if reuse_workspace and _validate_workspace(ws_path):
                print(f"  Reusing existing workspace: {ws_path}")
            else:
                if reuse_workspace:
                    print(f"  No valid existing workspace at {ws_path}, building fresh")
                    os.makedirs(ws_path, exist_ok=True)

                shutil.copytree(
                    C_APP_DIR,
                    ws_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.*', 'evaluation', 'input_data')
                )

                ws_input_data = os.path.join(ws_path, "input_data")
                os.makedirs(ws_input_data, exist_ok=True)

                main_input_data = os.path.join(C_APP_DIR, "input_data")
                if os.path.exists(main_input_data):
                    for item in os.listdir(main_input_data):
                        main_item_path = os.path.join(main_input_data, item)
                        if os.path.isdir(main_item_path) and not item.startswith('.'):
                            ws_item_path = os.path.join(ws_input_data, item)
                            if not os.path.exists(ws_item_path):
                                os.symlink(main_item_path, ws_item_path)

            workspaces.put(ws_path)
            if not reuse_workspace:
                temp_dirs.append(ws_path)

        if recordings:
            warmup_ws = workspaces.get()
            try:
                print("\nWarming up ccache with a throwaway compile...")
                subj_id, trial, mov, noise, sound = recordings[0]
                warmup_input_data_dir = os.path.join(warmup_ws, "input_data")
                warmup_result = transform_recording(subj_id, trial, mov, noise, sound,
                                                     dataset_path, warmup_input_data_dir)
                if warmup_result is not None:
                    _, audio_relpath, imu_relpath, bio_relpath = warmup_result
                    update_main_h(warmup_ws, audio_relpath, imu_relpath, bio_relpath)
                    compile_c_app(warmup_ws, extra_flags=extra_flags)
            finally:
                workspaces.put(warmup_ws)

        print(f"Beginning parallel evaluation of {len(recordings)} recordings...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_rec = {
                executor.submit(_worker_task, rec, dataset_path, workspaces, log_pa, log_na, extra_flags): rec
                for rec in recordings
            }

            for future in concurrent.futures.as_completed(future_to_rec):
                rec = future_to_rec[future]
                subj_id, trial, mov, noise, sound = rec
                rec_id = f"t{trial}_{mov}_{noise}_{sound}"

                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                        print(f"  [{subj_id}] {rec_id}: TP_evt={result['tp_evt']} "
                              f"FP_evt={result['fp_evt']} FN_evt={result['fn_evt']}")
                except Exception as exc:
                    print(f"  [{subj_id}] {rec_id} generated an exception: {exc}")

    finally:
        if reuse_workspace:
            print("\n--reuse-workspace enabled: leaving workspaces in place.")
        else:
            print("\nCleaning up workspaces...")
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)

    if len(all_results) != len(recordings):
        print(f"WARNING Unreliable result: processed {len(all_results)} of {len(recordings)} expected recordings")

    return all_results


# ──────────────────────────────────────────────
#  Aggregation
# ──────────────────────────────────────────────

def compute_aggregate_metrics(results):
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
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in CSV_FIELDNAMES})
    print(f"Results CSV saved to {output_path}")


def save_summary_json(aggregate, output_path, per_subject=None):
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
    raw = "__".join(str(part) for part in parts)
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{os.getpid()}_{digest}"


def _cleanup_fxp_harness(binary_path, build_tag):
    if binary_path:
        try:
            os.remove(binary_path)
        except FileNotFoundError:
            pass
        shutil.rmtree(f"{binary_path}.dSYM", ignore_errors=True)
    if build_tag:
        shutil.rmtree(os.path.join(FXP_DIR, ".build", build_tag), ignore_errors=True)


def _compile_fxp_error_harness(audio_relpath, imu_relpath, twiddle):
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
            "max_abs_err": float(kv.get("max_abs_err", "0")),
            "nonfinite": int(kv.get("nonfinite", "0")),
        }
    return rows


def _merge_acc(into, frm):
    for key, src in frm.items():
        dst = into.setdefault(key, {
            "n": 0, "sum_sq_err": 0.0,
            "max_abs_err": 0.0, "nonfinite": 0,
        })
        dst["n"] += src["n"]
        dst["sum_sq_err"] += src["sum_sq_err"]
        dst["nonfinite"] += src.get("nonfinite", 0)
        if src["max_abs_err"] > dst["max_abs_err"]:
            dst["max_abs_err"] = src["max_abs_err"]


def _acc_to_metrics(acc):
    import math
    if acc["n"] <= 0:
        return 0.0, 0.0
    rmse = math.sqrt(acc["sum_sq_err"] / acc["n"])
    return rmse, acc["max_abs_err"]


def _print_fxp_table(table, header):
    print(header)
    for block in ("audio", "imu"):
        kernels = sorted(k for (b, k) in table if b == block)
        for kernel in kernels:
            rmse, max_abs = _acc_to_metrics(table[(block, kernel)])
            nonfinite = table[(block, kernel)].get("nonfinite", 0)
            suffix = f"  NonFinite={nonfinite}" if nonfinite else ""
            print(f"  {block:<5}  {kernel:<22}  RMSE={rmse:.6g}  "
                  f"MaxAbs={max_abs:.6g}{suffix}")


def _evaluate_fxp_errors(dataset_path, twiddle):
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
    elif mode == "upos":
        return "-DUPOS_MODE"
    elif mode == "upos16q":
        return "-DUPOS_MODE -DPOSIT_SIZE=16"
    elif mode == "upos16nq":
        return "-DUPOS_MODE -DPOSIT_SIZE=16 -DNO_UPOS_QUIRE"
    elif mode == "upos32q":
        return "-DUPOS_MODE -DPOSIT_SIZE=32"
    elif mode == "upos32nq":
        return "-DUPOS_MODE -DPOSIT_SIZE=32 -DNO_UPOS_QUIRE"
    else:
        return f"-DFXP_MODE -DFIXED_POINT={twiddle}"


def _run_mode_eval(mode, twiddle, log_pa, log_na, cflags, jobs, reuse_workspace):
    compile_flags = _compile_flags_for_mode(mode, twiddle) + f" {cflags}"

    if mode == "float":
        print("Using float mode compile flags")
    else:
        print(f"Using {mode} mode compile flags: {compile_flags}")

    results = evaluate_subjects(DEFAULT_DATASET_PATH, log_pa=log_pa, log_na=log_na, jobs=jobs,
                                extra_flags=compile_flags, reuse_workspace=reuse_workspace)
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

    # upos = universal posit (default config)
    # upos16q = universal posit16 with quire
    # upos16nq = universal posit16 without quire
    # upos32q = universal posit32 with quire
    # upos32nq = universal posit32 without quire
    parser.add_argument("--mode", choices=["float", "fxp", "fxp-error", "upos", "upos16q", "upos16nq", \
                        "upos32q", "upos32nq"], default="float",
                        help="float / fxp / upos ML metrics, or fxp-error for kernel-level FxP-vs-float error metrics")
    parser.add_argument("--twiddle", type=int, choices=[32], default=32,
                        help="KissFFT twiddle precision (FxP audio uses 32-bit KissFFT)")
    parser.add_argument("--fxp-block", choices=FXP_BLOCKS,
                        help="run mixed float/FxP ML metrics with only this feature block replaced by FxP outputs")
    parser.add_argument("--log-pa", action="store_true", help="Produce precision analysis log")
    parser.add_argument("--log-na", action="store_true", help="Produce numerical analysis log")
    parser.add_argument("--cflags", type=str, default="", help="Extra CFLAGS at compilation")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Number of threads for parallel evaluation (default: 1)")
    parser.add_argument("--reuse-workspace", action="store_true",
                            help="Reuse existing /tmp workspace(s) from a prior run if valid, "
                                 "and don't delete workspaces after running")

    args = parser.parse_args(argv)

    if args.log_pa:
        output_dir = os.path.join(os.path.dirname(__file__), "precision_analysis")
        os.makedirs(output_dir, exist_ok=True)

    if args.log_na:
        output_dir = os.path.join(os.path.join(os.path.dirname(__file__), "numerical_analysis"), args.mode)
        os.makedirs(output_dir, exist_ok=True)

    try:
        if args.fxp_block:
            if args.mode != "float":
                parser.error("--fxp-block is an evaluation mode by itself; do not combine it with --mode fxp or --mode fxp-error")
            _run_progressive_eval(args.fxp_block, args.twiddle)
        elif args.mode == "fxp-error":
            _evaluate_fxp_errors(DEFAULT_DATASET_PATH, args.twiddle)
        else:
            _run_mode_eval(args.mode, args.twiddle, args.log_pa, args.log_na, args.cflags, args.jobs, args.reuse_workspace)
    except EvaluationError as exc:
        print(f"\nEvaluation aborted: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
