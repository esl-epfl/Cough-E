# Evaluation pipeline

This folder contains the scripts used to evaluate the Cough-E C application against the test dataset. The pipeline converts dataset recordings into C headers, compiles the C application, runs it for each recording, parses the emitted cough segments, and scores the output against ground truth.

Evaluation commands should be run from this folder:

```sh
cd C_application/evaluation
```


#### Scripts

`transform_dataset.py` converts WAV audio, CSV IMU, and JSON biodata files into C headers expected by `../input_data`.

The conversion uses the same `load_audio()` and `load_imu()` helpers as the Python ML pipeline so that the C application receives data prepared in the same way as the reference pipeline. Generated files are placed in per-subject subfolders and existing files are skipped.

`evaluate.py` runs the end-to-end evaluation. For each recording, it updates the three input includes in `main.h`, compiles the requested C runtime, executes the application, parses `COUGH_SEG` output lines, and scores predictions with event-based metrics.

`fxp/` contains the fixed-point stage harnesses used by `evaluate.py --mode fxp-error` and by progressive block evaluation. The harnesses compare fixed-point kernels against the floating-point reference code.


#### Dataset location

By default, `evaluate.py` expects the private test dataset at:

```text
Datasets/full_dataset_test
```

relative to the repository root. This default is resolved by the script, so it does not depend on the current working directory.

To transform a dataset stored somewhere else, pass the path explicitly to `transform_dataset.py`. The main `evaluate.py` script currently uses the default dataset path.


#### Transform input headers

Transform all subjects:

```sh
python3 transform_dataset.py \
  --dataset_path /path/to/full_dataset_test \
  --output_dir ../input_data
```

Transform selected subjects:

```sh
python3 transform_dataset.py \
  --dataset_path /path/to/full_dataset_test \
  --output_dir ../input_data \
  --subjects 55502 55503
```


#### End-to-end evaluation

Run the floating-point C application:

```sh
python3 evaluate.py --mode float
```

`--mode float` is the default, so this is equivalent:

```sh
python3 evaluate.py
```

Run the fixed-point C application:

```sh
python3 evaluate.py --mode fxp --twiddle 32
```

The fixed-point runtime currently uses 32-bit KissFFT twiddles, so `--twiddle 32` is the supported setting.


#### Progressive FxP evaluation

Progressive evaluation replaces one feature block with fixed-point outputs while keeping the rest of the pipeline in the reference path. This is useful for isolating which block changes the final event metrics.

Example:

```sh
python3 evaluate.py --fxp-block audio_mel
```

Supported blocks:

```text
audio_fft
audio_psd
audio_mel
audio_scalar
audio_all
imu_raw
imu_l2
imu_all
```

`--fxp-block` is its own evaluation mode and should not be combined with `--mode fxp` or `--mode fxp-error`.


#### Fixed-point kernel error

The fixed-point error mode compares FxP kernels against the floating-point baseline and reports absolute deviations. RMSE and MaxAbs are not relative percentages.

Run:

```sh
python3 evaluate.py --mode fxp-error
```

The stage harness emits accumulator rows like:

```text
FXP_KERNEL_ACC,block=audio,kernel=...,n=...,sum_sq_err=...,max_abs_err=...,nonfinite=...
```

`evaluate.py` aggregates those rows into per-kernel RMSE and MaxAbs values.


#### Output files

End-to-end modes write:

- `results_float.csv` and `summary_float.json`
- `results_fxp.csv` and `summary_fxp.json`

Progressive block modes write:

- `results_fxp_block_<block>.csv`
- `summary_fxp_block_<block>.json`

Fixed-point error mode prints kernel error summaries to stdout.


#### Scoring parameters

Event scoring uses the `timescoring` library with parameters matching `ML_methodology/config/scoring/default.yaml`.

| Parameter | Value |
|---|---:|
| `toleranceStart` | 0.25 s |
| `toleranceEnd` | 0.25 s |
| `minOverlap` | 0.125 |
| `maxEventDuration` | 0.6 s |
| `minDurationBetweenEvents` | 0 s |


#### Dependencies

The evaluation scripts require:

- The private `full_dataset_test` dataset.
- The Python ML helper modules under `ML_methodology/src`.
- `timescoring` for event-based metrics.
- `numpy` and `scipy` for data loading and preprocessing.
- A local C compiler and `make`.

The scripts temporarily modify `C_application/main.h` while evaluating recordings and restore it after the run.
