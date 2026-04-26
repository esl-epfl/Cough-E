# Audio FxP Module

This directory contains audio-domain fixed-point code only. Shared arithmetic and
type primitives remain in `FxP/core/`.

Current contents:

- `audio_pipeline_fxp.c`:
  - consolidated fixed-point audio runtime implementation (FFT, periodogram, MEL).
  - no per-processing-block split files remain.
- `audio_pipeline_fxp.h`:
  - public FxP audio entrypoints:
    - `audio_fft_features(...)`
    - `audio_psd_features(...)`
    - `audio_mel_features(...)`

Latest regression snapshot (`2026-04-20`):

- replacement command: `python C_application/evaluation/fxp/fxp_harness.py --twiddle 32 block`
- `N=14306` for all kernels

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `DOMINANT_FREQUENCY` | `9.7355` | `2.447%` | `1164.44` |
| `PSD_BAND_1` | `0.00076611` | `0.248%` | `0.0388142` |
| `PSD_BAND_2` | `0.0001911` | `0.161%` | `0.00533526` |
| `PSD_BAND_3` | `0.00040162` | `0.146%` | `0.00728337` |
| `SPECTRAL_FLATNESS` | `0.0045786` | `10.678%` | `0.44565` |

Latest Mel regression snapshot (`2026-04-20`):

- replacement command: `python C_application/evaluation/fxp/fxp_harness.py --twiddle 32 block`
- `N=915584` for all kernels

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `MEL_ENTROPY` | `0.55937` | `28.197%` | `2.56237` |
| `MEL_MAX` | `2.1756` | `9.565%` | `15.9926` |
| `MEL_MEAN` | `3.0621` | `10.167%` | `16.6487` |
| `MEL_STD` | `1.3962` | `20.366%` | `30.385` |

Latest Mel isolated sanity snapshot after FxP precision fixes (`2026-04-20`):

- command: `python C_application/evaluation/fxp/fxp_harness.py --twiddle 32 single-kernel`
- `N=320` per kernel (synthetic suite)

| Kernel | RMSE | RelRMSE | MaxAbs |
|---|---:|---:|---:|
| `MEL_MEAN` | `0.002771` | `0.007179%` | `0.0234032` |
| `MEL_STD` | `0.001706` | `0.022186%` | `0.0164583` |
| `MEL_MAX` | `0.005474` | `0.016545%` | `0.0601082` |
| `MEL_ENTROPY` | `0.187008` | `12.034413%` | `0.825341` |

Latest ML snapshot after periodogram port (`2026-04-20`):

- run context: FxP evaluation (`--mode fxp --twiddle 32`)
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

- run context: FxP evaluation (`--mode fxp --twiddle 32`)
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
  - `python C_application/evaluation/evaluate.py --mode fxp --twiddle 32`
- compile flags injected by `evaluate.py`:
  - `-DFXP_MODE -DFIXED_POINT=32`
- safety default:
  - if `--mode fxp` is set and `--twiddle` is omitted, `evaluate.py` defaults to `-DFIXED_POINT=32`.
- audio block behavior in this mode:
  - FFT block dispatch (`audio_fft_features`) is active for selected FFT kernels.
  - Periodogram block dispatch (`audio_psd_features`) is active for selected PSD kernels.
  - Mel block dispatch (`audio_mel_features`) is active for selected Mel bins/families.
- model selector note (`Inc/audio_model.h`):
  - selected FFT/PSD kernels are all among the FxP-ported set (`SPECTRAL_ROLLOFF`, `SPECTRAL_SPREAD`, `SPECTRAL_KURTOSIS`, `SPECTRAL_FLATNESS`, `DOMINANT_FREQUENCY`, `PSD_BAND_1..3`).
  - float-only fallback kernels in these families (`SPECTRAL_DECREASE`, `SPECTRAL_SLOPE`, `SPECTRAL_SKEW`, `SPECTRAL_STD`, `SPECTRAL_ENTROPY`) are not selected by the current model feature selector.

Planned extensions:

- none (current audio feature families now have FxP compute paths)
