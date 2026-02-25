"""
Transform public dataset recordings into C header files for the Cough-E C application.

Converts WAV audio + CSV IMU + JSON biodata from the public dataset into the exact
.h format expected by the C application's input_data/ directory.

Files are organized into per-subject subfolders: input_data/{subj_id}/

Usage:
    python transform_dataset.py --dataset_path /path/to/public_dataset --output_dir /path/to/C_application/input_data
    python transform_dataset.py --dataset_path /path/to/public_dataset --output_dir /path/to/C_application/input_data --subjects 55502 14287 14342
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import decimate
import pandas as pd
import json
import os
import argparse

# Constants matching C application expectations
AUDIO_FS_ORIGINAL = 16000
AUDIO_FS_TARGET = 8000
IMU_FS = 100
AUDIO_NORM_DIVISOR = 1 << 29  # 2^29, matches ML_methodology/src/helpers.py load_audio()

# Experimental conditions to process (matches no_bystander_cough_training test conditions)
SOUNDS = ["cough", "laugh", "talk", "deep_breathing", "throat_clearing"]
NOISES = ["nothing", "traffic", "music", "someone_else_cough"]
MOVEMENTS = ["sit", "walk"]
TRIALS = ["1", "2", "3"]


def make_recording_suffix(subj_id, trial, mov, noise, sound):
    """Create a unique suffix for header file naming."""
    return f"{subj_id}_t{trial}_{mov}_{noise}_{sound}"


def generate_audio_header(audio, audio_len, suffix, output_dir):
    """Generate audio input C header file matching the format in input_data/."""
    filename = f"audio_input_{suffix}.h"
    guard = f"_AUDIO_INPUT_{suffix.upper()}_H_"

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n\n")
        f.write("/* Sampling frequency */\n")
        f.write(f"#define AUDIO_FS  {AUDIO_FS_TARGET}\n\n")
        f.write("/* Number of samples per each audiosignal */\n")
        f.write(f"#define AUDIO_LEN   {audio_len}\n\n\n")
        f.write("/* \n")
        f.write("\tThe samples are taken from two microphones, one facing the sking\n")
        f.write("\tthe other facing the air \n")
        f.write("    In this case only the air is used, this is a simplified input\n")
        f.write("    data set\n")
        f.write("*/\n")
        f.write(f"typedef struct audio_input_{suffix}\n")
        f.write("{\n")
        f.write("    float air[AUDIO_LEN];\n")
        f.write("} audio_input_t;\n\n")
        f.write("static const audio_input_t audio_in = {\n\n")
        f.write("\t{\n")

        for i, val in enumerate(audio):
            comma = "," if i < audio_len - 1 else ""
            f.write(f"\t\t{float(val)}{comma}\n")

        f.write("\t}\n")
        f.write("};\n\n")
        f.write("#endif\n")

    return filename


def generate_imu_header(imu_data, imu_len, suffix, output_dir):
    """Generate IMU input C header file matching the format in input_data/."""
    filename = f"imu_input_{suffix}.h"
    guard = f"_IMU_INPUT_{suffix.upper()}_H_"

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write("/* Sampling frequency of the IMU signal */\n")
        f.write(f"#define IMU_FS {IMU_FS}\n\n")
        f.write("/* \n")
        f.write("\tNumber of samples for the IMU signals \n")
        f.write("\tNote that each sample contains the meaasurement of each of\n")
        f.write("\tthe three axis of the accelerometer and gyroscope.\n")
        f.write("*/\n")
        f.write(f"#define IMU_LEN {imu_len}\n\n\n\n")
        f.write("/*\n")
        f.write("\tEach sub-array contains one sample per each IMU signal\n")
        f.write("\tin the following order:\n")
        f.write("\t{\n")
        f.write("\t\tAccelerometer_x,\n")
        f.write("\t\tAccelerometer_y,\n")
        f.write("\t\tAccelerometer_z,\n")
        f.write("\t\tGyroscope_z,\n")
        f.write("\t\tGyroscope_p,\n")
        f.write("\t\tGyroscope_r\n")
        f.write("\t}\n")
        f.write("*/\n\n")
        f.write(f"static const float imu_in[IMU_LEN][6] = {{\n\n")

        for i in range(imu_len):
            row = imu_data[i]
            f.write("\t{\n")
            for j, val in enumerate(row):
                comma = "," if j < 5 else ""
                f.write(f"\t\t{round(float(val), 2)}{comma}\n")
            comma = "," if i < imu_len - 1 else ""
            f.write(f"\t}}{comma}\n")

        f.write("};\n\n")
        f.write("#endif\n")

    return filename


def generate_bio_header(bio_path, subj_id, output_dir):
    """Generate biodata C header file matching the format in input_data/."""
    filename = f"bio_input_{subj_id}.h"
    filepath = os.path.join(output_dir, filename)

    # Skip if already generated for this subject
    if os.path.exists(filepath):
        return filename

    with open(bio_path, 'r') as f:
        bio = json.load(f)

    gender = 1.0 if bio["Gender"] == "Female" else 0.0
    bmi = bio["BMI"]
    guard = f"_BIO_INPUT_{subj_id}_H_"

    with open(filepath, 'w') as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"static const float gender = {int(gender)};\n")
        f.write(f"static const float bmi = {bmi};\n\n")
        f.write("#endif\n")

    return filename


def transform_recording(subj_id, trial, mov, noise, sound, dataset_path, output_dir):
    """
    Transform a single recording from the public dataset into C header files.
    Files are placed in output_dir/{subj_id}/ subfolder.

    Returns:
        tuple: (suffix, audio_relpath, imu_relpath, bio_relpath) or None if recording doesn't exist.
        The relpaths are relative to output_dir's parent (e.g. "input_data/{subj_id}/file.h"
        becomes "{subj_id}/file.h" relative to output_dir).
    """
    rec_path = os.path.join(dataset_path, subj_id,
                            f'trial_{trial}', f'mov_{mov}',
                            f'background_noise_{noise}', sound)

    if not os.path.exists(rec_path) or len(os.listdir(rec_path)) == 0:
        return None

    suffix = make_recording_suffix(subj_id, trial, mov, noise, sound)
    subj_dir = os.path.join(output_dir, subj_id)
    os.makedirs(subj_dir, exist_ok=True)

    # Check if already transformed (idempotent)
    audio_file = f"audio_input_{suffix}.h"
    if os.path.exists(os.path.join(subj_dir, audio_file)):
        imu_file = f"imu_input_{suffix}.h"
        bio_file = f"bio_input_{subj_id}.h"
        return suffix, f"{subj_id}/{audio_file}", f"{subj_id}/{imu_file}", f"{subj_id}/{bio_file}"

    # Load and downsample audio (outward-facing mic only, matching C app)
    wav_path = os.path.join(rec_path, 'outward_facing_mic.wav')
    if not os.path.exists(wav_path):
        return None
    fs, audio_raw = wavfile.read(wav_path)
    assert fs == AUDIO_FS_ORIGINAL, f"Expected {AUDIO_FS_ORIGINAL}Hz, got {fs}Hz"

    decimation_ratio = AUDIO_FS_ORIGINAL // AUDIO_FS_TARGET
    audio = decimate(audio_raw.astype(np.float64), decimation_ratio)
    audio = audio / AUDIO_NORM_DIVISOR

    # Load IMU
    imu_path = os.path.join(rec_path, 'imu.csv')
    if not os.path.exists(imu_path):
        return None
    imu_df = pd.read_csv(imu_path)
    imu_data = imu_df[['Accel x', 'Accel y', 'Accel z',
                        'Gyro Y', 'Gyro P', 'Gyro R']].values.astype(np.float64)

    # Align durations: truncate both to the shorter signal's duration
    # Rule: IMU_LEN * (AUDIO_FS / IMU_FS) = AUDIO_LEN
    ratio = AUDIO_FS_TARGET // IMU_FS  # 80
    imu_len = len(imu_data)
    audio_len_from_imu = imu_len * ratio
    audio_len = min(len(audio), audio_len_from_imu)
    imu_len = audio_len // ratio
    audio_len = imu_len * ratio  # Ensure exact alignment

    audio = audio[:audio_len]
    imu_data = imu_data[:imu_len]

    # Generate header files into subject subfolder
    audio_file = generate_audio_header(audio, audio_len, suffix, subj_dir)
    imu_file = generate_imu_header(imu_data, imu_len, suffix, subj_dir)
    bio_file = generate_bio_header(
        os.path.join(dataset_path, subj_id, 'biodata.json'), subj_id, subj_dir)

    return suffix, f"{subj_id}/{audio_file}", f"{subj_id}/{imu_file}", f"{subj_id}/{bio_file}"


def transform_subject(subj_id, dataset_path, output_dir,
                      trials=TRIALS, movements=MOVEMENTS,
                      noises=NOISES, sounds=SOUNDS):
    """Transform all recordings for a given subject."""
    results = []
    for trial in trials:
        for mov in movements:
            for noise in noises:
                for sound in sounds:
                    result = transform_recording(
                        subj_id, trial, mov, noise, sound,
                        dataset_path, output_dir)
                    if result is not None:
                        results.append(result)
    return results


def transform_all(dataset_path, output_dir, subjects=None):
    """Transform all subjects (or a specified subset) from the public dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if subjects is None:
        subjects = sorted(os.listdir(dataset_path))
        subjects = [s for s in subjects if os.path.isdir(os.path.join(dataset_path, s))]

    all_results = {}
    for subj_id in subjects:
        print(f"Transforming subject {subj_id}...")
        results = transform_subject(subj_id, dataset_path, output_dir)
        all_results[subj_id] = results
        print(f"  Generated {len(results)} recording(s)")

    total = sum(len(v) for v in all_results.values())
    print(f"\nTotal: {total} recordings transformed for {len(subjects)} subjects")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform public dataset into C header files for Cough-E C application")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the public_dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the C application input_data directory")
    parser.add_argument("--subjects", nargs="+", type=str, default=None,
                        help="Specific subject IDs to transform (default: all)")
    args = parser.parse_args()

    transform_all(args.dataset_path, args.output_dir, args.subjects)
