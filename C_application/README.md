# Embedded application

This folder contains the C implementation of the Cough-E embedded application, together with the fixed-point runtime and the scripts used to compile and execute it locally.

The default build produces a `build/cough-e` executable. It can be compiled either as the original floating-point application or as the fixed-point application by passing the corresponding compiler flags.


#### Folder structure

The `Inc` folder contains the public headers for the floating-point feature extractors, model stages, post-processing, and application configuration.

The `Src` folder contains the floating-point feature extraction pipeline, model wrappers, filtering, helper functions, and cough post-processing code.

The `FxP` folder contains the fixed-point runtime:

- `FxP/core`: shared fixed-point types, Q-format aliases, arithmetic helpers, and log/exp approximations.
- `FxP/audio`: fixed-point audio kernels for FFT features, Welch PSD features, mel features, and scalar audio features.
- `FxP/imu`: fixed-point IMU kernels for raw-axis, accel-norm, and gyro-norm features.

The `kiss_fftr` folder contains the KissFFT real FFT backend and the generator for fixed-point twiddle tables.

The `input_data` folder contains C header files with audio, IMU, and biodata samples used by `main.h`.

The `evaluation` folder contains the dataset transformation and evaluation scripts. More detailed evaluation instructions are kept in `evaluation/README.md`.


#### Main settings

The `main.h` file contains the main application settings.

The post-processing period is controlled by `TIME_DEADLINE_OUTPUT_NUM`, `TIME_DEADLINE_OUTPUT_DEN`, and the derived `TIME_DEADLINE_OUTPUT_TICKS` macros. This setting determines how often the model reports the estimated cough segments.

The modality configuration is controlled by `RUN_ONLY_AUD`, `RUN_ONLY_IMU`, and `RUN_MIXED`.

The input data headers are also selected in `main.h`. These include one audio header, one IMU header, and one biodata header from `input_data`.

Feature-selection settings are defined in `audio_features.h` and `imu_features.h`.


#### Compile and execute

Run these commands from this folder:

```sh
cd C_application
```

Build the default floating-point application:

```sh
make
```

Build and run the default floating-point application:

```sh
make run
```

Build the fixed-point application:

```sh
make CFLAGS="-DFXP_MODE -DFIXED_POINT=32"
```

Build and run the fixed-point application:

```sh
make run CFLAGS="-DFXP_MODE -DFIXED_POINT=32"
```

Clean the local build output:

```sh
make clean
```

The Makefile automatically generates the fixed-point KissFFT twiddle tables when needed:

- `kiss_fftr/twiddles_win08_fs8000_q15.h`
- `kiss_fftr/twiddles_win08_fs8000_q31.h`


#### Input data

The application expects three input headers: audio, IMU, and biodata.

The example headers in `input_data` contain a short local test recording and are included by default through `main.h`.

Dataset recordings can be converted into the same header format from the `evaluation` folder with:

```sh
cd evaluation
python3 transform_dataset.py \
  --dataset_path /path/to/full_dataset_test \
  --output_dir ../input_data
```

The full evaluation pipeline updates the `main.h` includes automatically for each recording, so manual edits are only needed for ad-hoc local runs.
