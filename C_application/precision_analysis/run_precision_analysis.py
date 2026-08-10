import argparse
import subprocess
import json
import csv
import os
import shutil
from pathlib import Path

def cleanup_eval_workers():
    for path in Path("/tmp").glob("cough_e_eval_worker_*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def main():
    cleanup_eval_workers()

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", required=True)
    parser.add_argument("--from", dest="start_from", default=None)
    args = parser.parse_args()

    ctrl_groups = [
        ["compute_spec_decrease", "compute_spectral_slope", "compute_rolloff", "compute_centroid", "compute_spread", "compute_kurt", "compute_skew", "compute_rfft"],
        ["compute_periodogram", "compute_flatness", "compute_std", "compute_spectral_entropy", "get_domiant_freq", "normalized_bandpowers", "get_mfcc_features", "get_mel_spectrogram_features"],
        ["sub_mean", "compute_zrc", "get_rms", "get_crest", "eepd", "get_line_length", "get_kurtosis", "azc_computation"]
    ]

    # precisions = [("sreal_t", 1), ("mreal_t", 2)]
    precisions = [("sreal_t", 1)] # we skip mreal_t (posit24) for now

    csv_file = "precision_analysis.csv"
    file_exists = os.path.isfile(csv_file)

    start_skipping = args.start_from is not None

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["kernel", "precision", "SE", "PR", "F1", "FP_hr", "TP", "FP", "FN", "total_recordings", "total_duration_hrs"])

        for ctrl_idx in range(3):
            for kernel_idx, kernel_name in enumerate(ctrl_groups[ctrl_idx]):
                if start_skipping:
                    if kernel_name == args.start_from:
                        start_skipping = False
                    else:
                        continue

                for prec_name, prec_val in precisions:
                    print(f"Evaluating {prec_name} for {kernel_name}...")

                    ctrl_vals = [0, 0, 0]
                    shift = (7 - kernel_idx) * 2
                    ctrl_vals[ctrl_idx] = prec_val << shift

                    flags = f"-DPRECISION_CTRL1={ctrl_vals[0]} -DPRECISION_CTRL2={ctrl_vals[1]} -DPRECISION_CTRL3={ctrl_vals[2]}"

                    cmd = ["python3", "../evaluation/evaluate.py", "--mode", "upos", "--cflags", flags, "--log-pa", "--reuse-workspace", "-j", str(args.jobs)]
                    subprocess.run(cmd, check=True)

                    with open("../evaluation/summary_upos.json", "r") as json_f:
                        data = json.load(json_f)

                    ev = data["overall"]["event_based"]
                    ov = data["overall"]
                    writer.writerow([
                        kernel_name, prec_name,
                        ev["SE"], ev["PR"], ev["F1"], ev["FP_hr"],
                        ev["TP"], ev["FP"], ev["FN"],
                        ov["total_recordings"], ov["total_duration_hrs"]
                    ])

    cleanup_eval_workers()

if __name__ == "__main__":
    main()
