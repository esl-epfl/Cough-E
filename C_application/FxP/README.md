# FxP Module Layout

This folder is split by responsibility so fixed-point support can scale cleanly.

- `core/`: generic fixed-point infrastructure shared across domains.
  - `fxp_core.h`: consolidated core helpers (types, saturation, arithmetic, Q16 pipeline helpers).
  - `fxp_convert.h`: boundary converters and float/fixed helpers.
  - `qformats.yaml`: single source of truth for Q-format metadata.
  - `fxp_qformats.h`: generated constants from `qformats.yaml`.
- `imu/`: IMU-only kernels and IMU-specific Q-format notes.
  - `imu_pipeline.h`: typed IMU API surface and FxP kernel declarations.
  - `imu_pipeline.c`: consolidated IMU dispatch + kernel implementations.
- `audio/`: audio-domain FxP kernels/adapters (currently FFT-based features).

## Regenerating Q-format constants

```sh
python3 C_application/FxP/tools/generate_qformats.py
```

Run this after editing `core/qformats.yaml`.
