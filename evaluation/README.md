# Evaluation Pipeline

This folder contains scripts for systematically evaluating the Cough-E C application against the public dataset. The pipeline automates the full process: converting dataset recordings into C header files, compiling and running the C application for each recording, parsing the output, and scoring the results using event-based metrics.

## Changes made to the C application

To enable automated evaluation, three `printf` statements were added to `C_application/Src/main.c` to output detected cough segments (`COUGH_SEG`) and the number of peaks (`N_PEAKS FINAL`) after each postprocessing period. These outputs are parsed by the evaluation pipeline to extract predicted cough boundaries.

## Scripts

#### `transform_dataset.py`
Converts WAV audio, CSV IMU, and JSON biodata files from the public dataset into C header files (`.h`) expected by the C application's `input_data/` directory. Uses `load_audio()` and `load_imu()` directly from `ML_methodology/src/helpers.py` to ensure identical data preparation as the Python ML pipeline.

#### `evaluate.py`
Full evaluation pipeline with multiple subcommands. For each recording, it updates `main.h` to point to the generated headers, compiles the C application, runs it, parses the output, and scores the predictions against ground truth using event-based scoring from the [timescoring](https://github.com/esl-epfl/epilepsy_performance_metrics) library with parameters matching `ML_methodology/config/scoring/default.yaml`.

## Usage

All commands are run from the repository root (`Cough-E/`).

#### Full pipeline (transform + evaluate)
```
python evaluation/evaluate.py full --dataset_path /path/to/public_dataset
```

#### Transform dataset only
```
python evaluation/transform_dataset.py --dataset_path /path/to/public_dataset --output_dir C_application/input_data
```

#### Run evaluation only (assumes headers already generated)
```
python evaluation/evaluate.py run --dataset_path /path/to/public_dataset
```

#### Aggregate metrics from an existing results CSV
```
python evaluation/evaluate.py aggregate --csv evaluation/results.csv
```

## Output files

- `results.csv`: Per-recording results with event-based metrics (TP, FP, FN, SE, PR, F1)
- `summary.json`: Aggregate and per-subject metrics summary

## Dependencies

The evaluation scripts rely on:
- The public dataset (downloadable from [Zenodo](https://zenodo.org/records/7562332))
- `ML_methodology/src/helpers.py` for data loading functions
- The [timescoring](https://github.com/esl-epfl/epilepsy_performance_metrics) library for event-based scoring
- `numpy`, `scipy` (used by helpers.py for audio decimation)
