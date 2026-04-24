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

#### Specific subjects only
```
python C_application/evaluation/evaluate.py --subjects 14287 14342
```

#### Custom dataset path
```
python C_application/evaluation/evaluate.py --dataset-path /path/to/full_dataset_test
```

#### Transform dataset only
```
python C_application/evaluation/transform_dataset.py --dataset_path /path/to/full_dataset_test --output_dir C_application/input_data
```

#### Run evaluation only (assumes headers already generated)
```
python C_application/evaluation/evaluate.py --dataset-path /path/to/full_dataset_test --skip-transform
```

#### Run fixed-point evaluation
```
python C_application/evaluation/evaluate.py --mode fxp --twiddle 32 --dataset-path /path/to/full_dataset_test
```

#### Float-vs-fixed comparison
```
python C_application/evaluation/evaluate.py --compare --twiddle 32
```

## Fixed-Point Harness

`evaluation/fxp/fxp_harness.py` is the single entry point for fixed-point error
analysis. It owns harness builds, static FxP audits, per-stage percentage
metrics, KissFFT precision checks, Q-format sweeps, regression baselines, and
runtime hybrid experiments. `evaluate.py` remains the production end-to-end
runner.

Run all harness commands from the repository root. Two global options are useful
for most commands:

- `--twiddle {16,32}` selects the KissFFT fixed-point twiddle width used when
  building the harness. The default is `16`.
- `--max-windows N` limits the number of audio/IMU windows processed by the C
  stage harness. The default is `4`; use `1` for quick smoke tests.

Example:

```
python C_application/evaluation/fxp/fxp_harness.py --twiddle 32 --max-windows 1 block --block imu
```

Harness metric rows use a shared schema:

```
block,kernel,stage,backend,qformat,n,rmse,rel_rmse_pct,wape_pct,max_abs_pct,saturation_count,overflow_count
```

Hybrid mode currently switches the feature and model blocks independently for
audio and IMU. Lower-level FFT, Mel, periodogram, and postprocessing internals
are represented in the descriptor table as extension points until their fixed
kernels expose named trace hooks.

### Harness Commands

#### `audit`
Static check for accidental floating-point fallback in fixed-point runtime
slices. It scans the fixed-point feature extraction, audio/IMU pipelines, model
files, postprocessing, and fixed KissFFT sections for banned float tokens.

Use this after fixed-point code changes:

```
python C_application/evaluation/fxp/fxp_harness.py audit
```

Expected result:

```
FxP float fallback audit passed for runtime fixed-point slices.
```

#### `descriptors`
Prints the current block/kernel/stage registry. This shows which processing
stages are measurable today and which are placeholders for future trace hooks.

```
python C_application/evaluation/fxp/fxp_harness.py descriptors
```

Use this when adding a new trace point or checking whether a block is already
represented by the harness.

#### `single-kernel`
Prints percentage error rows for selected final feature outputs. These are
kernel-level output comparisons, not full internal traces.

Common usage:

```
python C_application/evaluation/fxp/fxp_harness.py single-kernel
python C_application/evaluation/fxp/fxp_harness.py single-kernel --block imu
python C_application/evaluation/fxp/fxp_harness.py single-kernel --block audio --kernel MEL
python C_application/evaluation/fxp/fxp_harness.py single-kernel --block imu --kernel RMS
```

Options:

- `--block {audio,imu}` filters to one feature family.
- `--kernel TEXT` filters rows whose kernel/stage name contains `TEXT`.

#### `block`
Prints conversion, intermediate, block-level, and model-logit metric rows for a
selected block.

```
python C_application/evaluation/fxp/fxp_harness.py block
python C_application/evaluation/fxp/fxp_harness.py block --block audio
python C_application/evaluation/fxp/fxp_harness.py block --block imu
```

Options:

- `--block {audio,imu,kissfft,postprocessing}` filters by block. `kissfft` and
  `postprocessing` are descriptor-level extension points; use `kissfft` for the
  active KissFFT precision mode.

#### `hybrid`
Runs runtime hybrid experiments where feature and model blocks can be float or
fixed independently. This is for A/B quantization analysis, not production
evaluation.

```
python C_application/evaluation/fxp/fxp_harness.py hybrid \
  --audio-features-backend float --audio-model-backend fxp \
  --imu-features-backend fxp --imu-model-backend float
```

Options:

- `--audio-features-backend {float,fxp}`
- `--audio-model-backend {float,fxp}`
- `--imu-features-backend {float,fxp}`
- `--imu-model-backend {float,fxp}`
- `--trace-limit N` also emits trace rows for the first `N` windows.

Hybrid output includes explicit `hybrid-bridge` rows when model inputs cross
between float and fixed-point formats.

#### `trace`
Prints detailed trace rows for early windows plus the normal stage metrics.
Current trace rows cover selected feature outputs; deeper FFT, Mel,
periodogram, and postprocessing internals will appear here as their fixed
kernels gain named trace hooks.

```
python C_application/evaluation/fxp/fxp_harness.py trace
python C_application/evaluation/fxp/fxp_harness.py trace --trace-limit 3
```

#### `q-sensitivity`
Sweeps candidate Q-format fractional widths at marked boundaries and reports
percentage impact on model-logit metrics.

```
python C_application/evaluation/fxp/fxp_harness.py q-sensitivity
python C_application/evaluation/fxp/fxp_harness.py --max-windows 1 q-sensitivity
```

Current sweep locations are audio input, IMU input, and model-feature pipes.

#### `stage-all`
Prints all stage rows. Add `--sweep` for Q-format sweep rows and
`--trace-limit N` for trace rows.

```
python C_application/evaluation/fxp/fxp_harness.py stage-all
python C_application/evaluation/fxp/fxp_harness.py stage-all --sweep
python C_application/evaluation/fxp/fxp_harness.py stage-all --trace-limit 2 --sweep
```

This is the best command when collecting raw harness output for inspection.

#### `kissfft`
Runs the integrated KissFFT Q15-vs-Q31 precision study. It compares output bins
for deterministic input signals and reports magnitude RMSE, relative RMSE, and
gain-corrected metrics.

```
python C_application/evaluation/fxp/fxp_harness.py kissfft
python C_application/evaluation/fxp/fxp_harness.py kissfft --nffts 900 --signals impulse
python C_application/evaluation/fxp/fxp_harness.py kissfft --nffts 900 2048 --signals tone_bin7 noise --write-csv
```

Options:

- `--nffts N ...` selects FFT lengths. Defaults are `900 2048 6400`.
- `--signals NAME ...` selects deterministic test signals.
- `--write-csv` writes case and summary CSV files.
- `--output-dir PATH` chooses where CSV files are written.

Valid signals are `impulse`, `tone_bin7`, `dual_5_37`, `chirp`, and `noise`.

#### `end-to-end`
Hands off to `evaluate.py --compare` for full float-vs-fixed event-level
evaluation. Use this for ML metrics, not low-level numeric tracing.

```
python C_application/evaluation/fxp/fxp_harness.py end-to-end
python C_application/evaluation/fxp/fxp_harness.py end-to-end --skip-transform
python C_application/evaluation/fxp/fxp_harness.py end-to-end --subjects 14287 --sounds clean --noises none
```

Options are forwarded to `evaluate.py` where applicable:

- `--subjects ID ...`
- `--sounds NAME ...`
- `--noises NAME ...`
- `--skip-transform`

#### `regression`
Compares current stage metrics against the accepted baseline JSON.

```
python C_application/evaluation/fxp/fxp_harness.py regression
```

To refresh the baseline after intentionally accepted numeric changes:

```
python C_application/evaluation/fxp/fxp_harness.py regression --write-baseline
```

Options:

- `--baseline PATH` chooses a non-default baseline file.
- `--rel-tolerance FLOAT` allows relative growth over baseline metrics.
- `--abs-tolerance-pct FLOAT` sets a minimum absolute percentage tolerance.

## Output files

- `results.csv`: Per-recording results with event-based metrics (TP, FP, FN, SE, PR, F1)
- `summary.json`: Aggregate and per-subject metrics summary

## Dependencies

The evaluation scripts rely on:
- The private test dataset (`full_dataset_test`).
- `ML_methodology/src/helpers.py` for data loading functions
- The [timescoring](https://github.com/esl-epfl/epilepsy_performance_metrics) library for event-based scoring
- `numpy`, `scipy` (used by helpers.py for audio decimation)
