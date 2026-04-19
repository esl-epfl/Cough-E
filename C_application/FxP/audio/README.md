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
  - high-level mixed-precision entry point used by `fft_based_features()`
  - runs the fixed-point kernels and writes feature outputs in float format

Planned extensions:

- periodogram-domain FxP kernels
- mel-spectrogram-domain FxP kernels
