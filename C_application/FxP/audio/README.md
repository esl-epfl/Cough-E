# Audio FxP Module

This directory contains audio-domain fixed-point code only. Shared arithmetic and
type primitives remain in `FxP/core/`.

Current contents:

- `audio_fft_bridge.{h,c}`:
  - float<->FxP bridge utilities for FFT-domain arrays
  - Q-format conversions for magnitudes/frequencies/sum
- `audio_fft_kernels.{h,c}`:
  - modular fixed-point kernels for:
    - spectral rolloff
    - spectral centroid
    - spectral spread
    - spectral kurtosis
- `audio_fft_block.{h,c}`:
  - high-level FxP entry points used by `fft_based_features()`
  - `fxp_audio_fft_features_from_signal(...)` computes FFT-domain features from
    signal input through a fixed-point FFT path
  - `fxp_audio_fft_features_hybrid(...)` keeps the previous bridge-based kernel
    regression path
- `audio_periodogram_bridge.{h,c}`:
  - float<->FxP bridge for periodogram-domain arrays
  - PSD proxy reconstruction using scale-cancellation described in `OVERLEAF.md`
- `audio_periodogram_kernels.{h,c}`:
  - modular fixed-point kernels for:
    - dominant frequency
    - spectral flatness
    - normalized band powers
- `audio_periodogram_block.{h,c}`:
  - high-level FxP entry points used by `periodogram_based_features()`
  - `fxp_audio_periodogram_features_from_signal(...)` computes periodogram-domain
    features from signal input through fixed-point Welch/FFT arithmetic
  - `fxp_audio_periodogram_features_hybrid(...)` keeps the previous bridge-based
    kernel regression path
  - implementation note:
    - for `FIXED_POINT=32`, this block uses an internal high-precision
      periodogram input format (`Q1.30`) before the KISS FFT to reduce
      low-energy bin quantization error (important for spectral flatness)
    - this change is intentionally local to the periodogram block
      (`audio_periodogram_block.c`) and does not modify the FFT feature block
      (`audio_fft_block.c`), which keeps its existing int16/Q14 path
    - no 128-bit arithmetic is used in the periodogram path
- `audio_mel_block.{h,c}`:
  - high-level FxP entry point used by `mel_spectrogram_features()`
  - `fxp_audio_mel_features_from_signal(...)` computes selected Mel feature
    families (mean/std/max/entropy per required Mel bins) from signal input
    through a fixed-point STFT + Mel projection path
  - output write-back remains float to preserve the current downstream
    classifier interface

Latest regression snapshot (`2026-04-20`):

- command: `python C_application/evaluation/evaluate.py error-audio-psd --kissfft-fixed 32`
- `N=14306` for all kernels

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `DOMINANT_FREQUENCY` | `9.7355` | `2.447%` | `1164.44` |
| `PSD_BAND_1` | `0.00076611` | `0.248%` | `0.0388142` |
| `PSD_BAND_2` | `0.0001911` | `0.161%` | `0.00533526` |
| `PSD_BAND_3` | `0.00040162` | `0.146%` | `0.00728337` |
| `SPECTRAL_FLATNESS` | `0.0045786` | `10.678%` | `0.44565` |

Latest Mel regression snapshot (`2026-04-20`):

- command: `python C_application/evaluation/evaluate.py error-audio-mel --kissfft-fixed 32`
- `N=915584` for all kernels

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `MEL_ENTROPY` | `0.55937` | `28.197%` | `2.56237` |
| `MEL_MAX` | `2.1756` | `9.565%` | `15.9926` |
| `MEL_MEAN` | `3.0621` | `10.167%` | `16.6487` |
| `MEL_STD` | `1.3962` | `20.366%` | `30.385` |

Latest Mel isolated sanity snapshot after FxP precision fixes (`2026-04-20`):

- command: `make -B -C C_application/test fxp_audio_mel_regression FFT_MODE=-DFIXED_POINT=32 && C_application/test/fxp_audio_mel_regression`
- `N=320` per kernel (synthetic suite)

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `MEL_MEAN` | `0.002771` | `0.007179%` | `0.0234032` |
| `MEL_STD` | `0.001706` | `0.022186%` | `0.0164583` |
| `MEL_MAX` | `0.005474` | `0.016545%` | `0.0601082` |
| `MEL_ENTROPY` | `0.187008` | `12.034413%` | `0.825341` |

Latest ML snapshot after periodogram port (`2026-04-20`):

- run context: FxP evaluation (`--fxp --kissfft-fixed 32`)
- dataset coverage: `594 recordings` (`1.688 hrs`)
- overall metrics:
  - `SE=0.5892`
  - `PR=0.7941`
  - `F1=0.6765`
  - `FP/hr=98.9`
  - `TP=644`, `FP=167`, `FN=449`
- artifacts:
  - `C_application/evaluation/results.csv`
  - `C_application/evaluation/summary.json`

Latest ML snapshot with full FxP feature pipeline (`2026-04-20`):

- run context: FxP evaluation (`--fxp --kissfft-fixed 32`)
- dataset coverage: `594 recordings` (`1.688 hrs`)
- overall metrics:
  - `SE=0.5819`
  - `PR=0.7980`
  - `F1=0.6730`
  - `FP/hr=95.4`
  - `TP=636`, `FP=161`, `FN=457`
- artifacts:
  - `C_application/evaluation/results.csv`
  - `C_application/evaluation/summary.json`
- quick delta vs previous snapshot above:
  - `SE`: `-0.0073`
  - `PR`: `+0.0039`
  - `F1`: `-0.0035`
  - `FP/hr`: `-3.5`
  - interpretation: only small net change overall; slightly fewer false positives but slightly more missed positives.

ML execution in FxP mode:

- command:
  - `python C_application/evaluation/evaluate.py run --fxp --kissfft-fixed 32`
- compile flags injected by `evaluate.py`:
  - `-DFXP_MODE -DFIXED_POINT=32`
- safety default:
  - if `--fxp` is set and `--kissfft-fixed` is omitted in `run`/`compare`, `evaluate.py` now defaults to `-DFIXED_POINT=32`.
- audio block behavior in this mode:
  - FFT block dispatch (`fxp_audio_fft_features_from_signal`) is active for selected FFT kernels.
  - Periodogram block dispatch (`fxp_audio_periodogram_features_from_signal`) is active for selected PSD kernels.
  - Mel block dispatch (`fxp_audio_mel_features_from_signal`) is active for selected Mel bins/families.
- model selector note (`Inc/audio_model.h`):
  - selected FFT/PSD kernels are all among the FxP-ported set (`SPECTRAL_ROLLOFF`, `SPECTRAL_SPREAD`, `SPECTRAL_KURTOSIS`, `SPECTRAL_FLATNESS`, `DOMINANT_FREQUENCY`, `PSD_BAND_1..3`).
  - float-only fallback kernels in these families (`SPECTRAL_DECREASE`, `SPECTRAL_SLOPE`, `SPECTRAL_SKEW`, `SPECTRAL_STD`, `SPECTRAL_ENTROPY`) are not selected by the current model feature selector.

Planned extensions:

- none (current audio feature families now have FxP compute paths)
