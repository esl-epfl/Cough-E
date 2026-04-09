# Range Analysis

This folder contains the infrastructure to characterize the numerical ranges of intermediate values throughout the C application. It is used to inform fixed-point design decisions by identifying the min, max, and absolute maximum of key signals and intermediate results across the full dataset.

## Files

#### `range_analysis.h`
Header-only instrumentation library. Defines macros (`RA_LOG_ARRAY`, `RA_LOG_SCALAR`, `RA_IMU_LOG_ARRAY`, `RA_IMU_LOG_SCALAR`) that emit `RANGE|...` lines to stdout when the application is compiled with `-DRANGE_ANALYSIS`. All macros compile away to nothing without that flag, so there is no runtime cost in production builds.

Output line format:
```
RANGE|<section>|<function>|<variable>|<len>|<min>|<max>|<absmax>
```

Sections used across the pipeline: `IMU_RAW`, `IMU_L2_ACCEL`, `IMU_L2_GYRO`, `AUDIO_FFT`, `AUDIO_PSD`, `AUDIO_MEL`, `AUDIO_EEPD`, `CLASSIFY`, `POSTPROC`.

#### `run_range_analysis.py`
Runs the application over the full dataset and aggregates the logged ranges. For each recording it updates `main.h`, compiles with `-DRANGE_ANALYSIS`, runs the binary, and collects all `RANGE|...` lines. After all recordings are processed, it aggregates the global min, max, and absmax per `(section, function, variable)` triple and writes the results to `range_results.csv`.

## Usage

All commands are run from the repository root (`Cough-E/`).

#### Full run (all subjects)
```
python C_application/range_analysis/run_range_analysis.py
```

#### Specific subjects only
```
python C_application/range_analysis/run_range_analysis.py --subjects 14287 14342
```

#### Custom dataset path
```
python C_application/range_analysis/run_range_analysis.py --dataset_path /path/to/full_dataset_test
```

## Output files

- `range_results.csv`: Aggregated range statistics per `(section, function, variable)` with columns `section`, `function`, `variable`, `len`, `global_min`, `global_max`, `global_absmax`, `always_positive`, `n_observations`.

## Dependencies

- The private test dataset (`full_dataset_test`).
- `C_application/evaluation/transform_dataset.py` for dataset-to-header conversion (reused directly).
