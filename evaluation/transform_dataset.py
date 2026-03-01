"""
Transform public dataset recordings into C header files for the Cough-E C application.

Converts WAV audio + CSV IMU + JSON biodata into the .h format expected by input_data/.
Files are organized into per-subject subfolders: input_data/{subj_id}/

Usage:
    python transform_dataset.py --dataset_path /path/to/public_dataset \
                                --output_dir /path/to/C_application/input_data
    python transform_dataset.py --dataset_path /path/to/public_dataset \
                                --output_dir /path/to/C_application/input_data \
                                --subjects 55502 14287
"""

import argparse
import importlib.util
import json
import os

import numpy as np

# Import helpers from ML_methodology/src (bypassing __init__.py which pulls heavy deps)
_helpers_path = os.path.join(os.path.dirname(__file__), '..', 'ML_methodology', 'src', 'helpers.py')
_spec = importlib.util.spec_from_file_location("helpers", _helpers_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

load_audio = _helpers.load_audio
load_imu = _helpers.load_imu
FS_AUDIO = _helpers.FS_AUDIO
FS_IMU = _helpers.FS_IMU
Sound = _helpers.Sound
Noise = _helpers.Noise
Movement = _helpers.Movement
Trial = _helpers.Trial

# Constants
AUDIO_FS_TARGET = 8000
IMU_FS = FS_IMU  # 100 Hz

# Experimental conditions (matches no_bystander_cough_training test conditions)
SOUNDS = [s.value for s in Sound]
NOISES = [n.value for n in Noise]
MOVEMENTS = [m.value for m in Movement]
TRIALS = [t.value for t in Trial]


# ──────────────────────────────────────────────
#  Header file generation
# ──────────────────────────────────────────────

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
    """Generate biodata C header file. Skips if already exists (one per subject)."""
    filename = f"bio_input_{subj_id}.h"
    filepath = os.path.join(output_dir, filename)

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


# ──────────────────────────────────────────────
#  Recording & subject transformation
# ──────────────────────────────────────────────

def transform_recording(subj_id, trial, mov, noise, sound, dataset_path, output_dir):
    """
    Transform a single recording from the public dataset into C header files.

    Returns:
        (suffix, audio_relpath, imu_relpath, bio_relpath) or None if recording doesn't exist.
        Relpaths are relative to output_dir (e.g. "{subj_id}/audio_input_{suffix}.h").
    """
    rec_path = os.path.join(dataset_path, subj_id,
                            f'trial_{trial}', f'mov_{mov}',
                            f'background_noise_{noise}', sound)

    if not os.path.exists(rec_path) or len(os.listdir(rec_path)) == 0:
        return None

    suffix = make_recording_suffix(subj_id, trial, mov, noise, sound)
    subj_dir = os.path.join(output_dir, subj_id)
    os.makedirs(subj_dir, exist_ok=True)

    # Skip if already transformed (idempotent)
    audio_file = f"audio_input_{suffix}.h"
    imu_file = f"imu_input_{suffix}.h"
    bio_file = f"bio_input_{subj_id}.h"

    if os.path.exists(os.path.join(subj_dir, audio_file)):
        return suffix, f"{subj_id}/{audio_file}", f"{subj_id}/{imu_file}", f"{subj_id}/{bio_file}"

    # Load and downsample audio
    try:
        audio_air, _ = load_audio(dataset_path + '/', subj_id, AUDIO_FS_TARGET,
                                  trial, mov, noise, sound)
    except Exception:
        return None

    # Load IMU data
    imu = load_imu(dataset_path + '/', subj_id, trial, mov, noise, sound)
    if imu == 0:
        return None

    imu_data = np.stack((imu.x, imu.y, imu.z, imu.Y, imu.P, imu.R), axis=1)
    audio = audio_air

    # Align durations: truncate to shorter signal, enforcing IMU_LEN * 80 = AUDIO_LEN
    ratio = AUDIO_FS_TARGET // IMU_FS  # 80
    imu_len = len(imu_data)
    audio_len = min(len(audio), imu_len * ratio)
    imu_len = audio_len // ratio
    audio_len = imu_len * ratio  # ensure exact alignment

    audio = audio[:audio_len]
    imu_data = imu_data[:imu_len]

    # Generate header files
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
        subjects = sorted([
            s for s in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, s))
        ])

    all_results = {}
    for subj_id in subjects:
        print(f"Transforming subject {subj_id}...")
        results = transform_subject(subj_id, dataset_path, output_dir)
        all_results[subj_id] = results
        print(f"  Generated {len(results)} recording(s)")

    total = sum(len(v) for v in all_results.values())
    print(f"\nTotal: {total} recordings transformed for {len(subjects)} subjects")
    return all_results


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform public dataset into C header files for Cough-E")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the public_dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to C application input_data directory")
    parser.add_argument("--subjects", nargs="+", type=str, default=None,
                        help="Specific subject IDs (default: all)")
    args = parser.parse_args()

    transform_all(args.dataset_path, args.output_dir, args.subjects)
