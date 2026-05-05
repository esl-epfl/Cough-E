# FxP Module Layout

This folder is split by the remaining fixed-point runtime pieces.

- `core/`: generic fixed-point infrastructure shared across domains.
  - `fxp_core.h`: consolidated core helpers (Q-format constants, types, saturation, arithmetic, conversion, backend carriers, Q16 pipeline helpers).
- `imu/`: IMU-only kernels.
  - `imu_pipeline.h`: typed IMU API surface and FxP kernel declarations.
  - `imu_pipeline.c`: consolidated IMU dispatch + kernel implementations.
- `audio/`: audio-domain FxP kernels/adapters for FFT, PSD, mel, and selected scalar features.
  - `audio_tables_q15.h`: consolidated generated Q15 audio tables (Welch Hann, STFT Hann, Mel basis).
