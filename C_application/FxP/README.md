# FxP Module Layout

This folder is split by responsibility so fixed-point support can scale cleanly.

- `core/`: generic fixed-point infrastructure shared across domains.
  - `fxp_core.h`: consolidated core helpers (types, saturation, arithmetic, Q16 pipeline helpers).
  - `fxp_convert.h`: boundary converters and float/fixed helpers.
  - `fxp_qformats.h`: fixed-point Q-format constants.
- `imu/`: IMU-only kernels and IMU-specific Q-format notes.
  - `imu_pipeline.h`: typed IMU API surface and FxP kernel declarations.
  - `imu_pipeline.c`: consolidated IMU dispatch + kernel implementations.
- `audio/`: audio-domain FxP kernels/adapters (currently FFT-based features).
