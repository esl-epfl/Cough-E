# FxP Module Layout

This folder is split by responsibility so fixed-point support can scale cleanly.

- `core/`: generic fixed-point infrastructure shared across domains.
  - `fxp_types.h`: type aliases and compile-time width checks.
  - `fxp_convert.h`: boundary converters and float/fixed helpers.
  - `fxp_sat.h`: saturating cast helpers.
  - `fxp_math.h`: arithmetic primitives (mul/div/sqrt/abs).
  - `qformats.yaml`: single source of truth for Q-format metadata.
  - `fxp_qformats.h`: generated constants from `qformats.yaml`.
- `imu/`: IMU-only kernels and IMU-specific Q-format notes.
  - `imu_kernels.h`: IMU kernel API surface.
  - `imu_dispatch.h`: typed IMU feature dispatch API.
- `audio/`: audio-domain FxP kernels/adapters (currently FFT-based features).

## Regenerating Q-format constants

```sh
python3 C_application/FxP/tools/generate_qformats.py
```

Run this after editing `core/qformats.yaml`.
