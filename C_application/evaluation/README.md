# Evaluation Pipeline

This folder contains scripts for systematically evaluating the Cough-E C application against the test set (`full_dataset_test`). The pipeline automates the full process: converting dataset recordings into C header files, compiling and running the C application for each recording, parsing the output, and scoring the results using event-based metrics.

## Scripts

#### `transform_dataset.py`
Converts WAV audio, CSV IMU, and JSON biodata files from the public dataset into C header files (`.h`) expected by the C application's `input_data/` directory. Uses `load_audio()` and `load_imu()` directly from `ML_methodology/src/helpers.py` to ensure identical data preparation as the Python ML pipeline. Generated files are organized into per-subject subfolders and are idempotent (skipped if already present).

#### `evaluate.py`
Full evaluation pipeline for float or fixed-point runtime mode. For each recording, it updates `main.h` to point to the generated headers, compiles the C application, runs it, parses the output, and scores the predictions against ground truth using event-based scoring from the [timescoring](https://github.com/esl-epfl/epilepsy_performance_metrics) library with parameters matching `ML_methodology/config/scoring/default.yaml`:

| Parameter | Value |
|---|---|
| `toleranceStart` | 0.25 s |
| `toleranceEnd` | 0.25 s |
| `minOverlap` | 0.125 |
| `maxEventDuration` | 0.6 s |
| `minDurationBetweenEvents` | 0 s |

## Usage

All commands are run from the repository root (`Cough-E/`).

#### Full pipeline (transform + evaluate, all subjects)
```
python C_application/evaluation/evaluate.py
```

#### Transform dataset only
```
python C_application/evaluation/transform_dataset.py --dataset_path /path/to/full_dataset_test --output_dir C_application/input_data
```

#### Run fixed-point evaluation
```
python C_application/evaluation/evaluate.py --mode fxp --twiddle 32
```

## Fixed-Point Harness

`evaluation/fxp/fxp_harness.py` reports fixed-point kernel error metrics against
the float baseline. `evaluate.py` remains the production end-to-end runner for
float/FxP ML metrics.

Run from the repository root:

```
python C_application/evaluation/fxp/fxp_harness.py
```

The script emits one `FXP_ERROR` row per measured kernel/function:

```
FXP_ERROR,block=audio,kernel=...,stage=...,qformat=...,n=...,rmse_pct=...,max_abs_pct=...
```

## Output files

- `results_float.csv` / `results_fxp.csv`: Per-recording event metrics.
- `summary_float.json` / `summary_fxp.json`: Aggregate and per-subject summaries.

## Dependencies

The evaluation scripts rely on:
- The private test dataset (`full_dataset_test`).
- `ML_methodology/src/helpers.py` for data loading functions
- The [timescoring](https://github.com/esl-epfl/epilepsy_performance_metrics) library for event-based scoring
- `numpy`, `scipy` (used by helpers.py for audio decimation)
