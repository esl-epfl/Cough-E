# Embedded application

This folder contains the C implementation of the Cough-E embedded application, together with the fixed-point runtime and the scripts used to compile and execute it locally.

The default build produces a `build/cough-e` executable. It can be compiled as the original floating-point application, or as the fixed-point or posit application by passing the corresponding compiler flags.

**NOTE**: `g++` is BROKEN while using templated constant values with external linkage, such as `hann_mfcc_wind`. We get "explicit instantiation but no definition available" when compiling `constants.cpp`. This issue does not exist on `clang++`.

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

Build the universal posit application:

```sh
make CC="clang++" CFLAGS="-DUPOS_MODE"
```

Build and run the universal posit application:

```sh
make run CC="clang++" CFLAGS="-DUPOS_MODE"
```

Clean the local build output:

```sh
make clean
```

The Makefile automatically generates the fixed-point KissFFT twiddle tables when needed:

- `kiss_fftr/twiddles_win08_fs8000_q15.cpp`
- `kiss_fftr/twiddles_win08_fs8000_q31.cpp`


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

#### HEEPatia

To compile cough-e with libposit on HEEPatia, use:

```bash
make app PROJECT=cough-e TARGET=zcu104 LINKER=flash_load BOOT_MODE=flash COMPILER_FLAGS="-DTARGET_BYPASS_FLL -DHEEPATIA_MODE -DEVALUATION_MODE -DDEBUG -DLPOS_MODE -DPOSIT_SIZE=16" TOOLCHAIN=POS RISCV_POS=/shares/eslfiler1/common/esl/HEEPatia/compilers/rv32imfcxposit_patched
```

Note: If compiling for 32-bit posit, add `-mavoid-pmv` compiler flag as a hardware hazard workaround.

FLASH, CARUS and GCRAM are utilized due to limited main memory in HEEPatia. Note that we use virtual GCRAM as native support on FPGA is not assumed. Also, we explicitly configure linker to use `ALIGN(4)` for `carus`. The modified [memory layout](/config/mcu-gen-system.hjson) and [linker](/sw/linker/link_flash_load.ld.tpl) must be used.

We override `_sbrk` to reroute heap data to virtual GCRAM because main `data` only has 32kB space - this requires modifying `CMakeLists.txt` by adding `-Wl,--wrap=_sbrk` linker flag. We do not use GCRAM for constant data and we keep stack data in main `data` section.

CARUS are marked as NOLOAD in linker and so we store certain constant data in FLASH and manually load them into CARUS on startup. It is not possible to remove NOLOAD flag because CARUS has high address and linker would automatically pad unused addresses, creating a huge binary that cannot be loaded into FPGA.

Twiddles for `kiss-fftr`, cosine LUT for `dct-linear`, and input data are stored in FLASH and loaded on demand. Other constant data is mostly stored in CARUS and INTERLEAVED data, if not in main `data`, and the usage are as follows:

- CARUS0: 54400 bytes / 65536 bytes (audio model)
- CARUS1: 60400 bytes / 65536 bytes (imu model)
- INTERLEAVED: 121680 bytes / 131072 bytes (other constant)
