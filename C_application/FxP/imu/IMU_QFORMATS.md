# IMU Q-Formats

This table is derived from `FxP/core/qformats.yaml`.

| Signal / Feature | Type | Q-format | Frac bits |
|---|---|---|---|
| Raw IMU axis | `q11_5_t` | Q11.5 | `FXP_FRAC_IMU_RAW` |
| L2 accel norm | `uq10_6_t` | UQ10.6 | `FXP_FRAC_IMU_L2A` |
| L2 gyro norm | `uq5_11_t` | UQ5.11 | `FXP_FRAC_IMU_L2G` |
| RMS (raw) | `uq16_16_t` | UQ16.16 | `FXP_FRAC_IMU_RMS_RAW` |
| RMS (L2A) | `uq13_3_t` | UQ13.3 | `FXP_FRAC_IMU_RMS_L2A` |
| RMS (L2G) | `uq7_9_t` | UQ7.9 | `FXP_FRAC_IMU_RMS_L2G` |
| Line length (raw) | `uq9_23_t` | UQ9.23 | `FXP_FRAC_IMU_LINE_LENGTH_RAW` |
| Line length (L2G) | `uq7_9_t` | UQ7.9 | `FXP_FRAC_IMU_LINE_LENGTH_L2G` |
| Kurtosis output | `q34_30_t` | Q34.30 | `FXP_FRAC_IMU_KURTOSIS_RAW` |
| Crest factor (L2G) | `uq2_14_t` | UQ2.14 | `FXP_FRAC_IMU_CREST_L2G` |

Kernel implementations live in one file per family:

- `imu_kernel_rms.c`
- `imu_kernel_line_length.c`
- `imu_kernel_kurtosis.c`
- `imu_kernel_norm.c`
- `imu_kernel_peak.c`
- `imu_kernel_azc.c`
