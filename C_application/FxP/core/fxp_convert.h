#pragma once

#include <limits.h>
#include <stdint.h>

#include <core/fxp_core.h>
#include <core/fxp_qformats.h>

static inline int32_t fxp_from_float_signed(float x, uint8_t frac_bits)
{
    float scale = (float)(1ULL << frac_bits);
    float scaled = x * scale;
    scaled += (scaled >= 0.0f) ? 0.5f : -0.5f;

    if (scaled > (float)INT32_MAX) return INT32_MAX;
    if (scaled < (float)INT32_MIN) return INT32_MIN;
    return (int32_t)scaled;
}

static inline uint32_t fxp_from_float_unsigned(float x, uint8_t frac_bits)
{
    if (x <= 0.0f) return 0;

    float scale = (float)(1ULL << frac_bits);
    float scaled = x * scale + 0.5f;

    if (scaled > (float)UINT32_MAX) return UINT32_MAX;
    return (uint32_t)scaled;
}

static inline float fxp_to_float(int64_t x, uint8_t frac_bits)
{
    float scale = (float)(1ULL << frac_bits);
    return (float)x / scale;
}

// Compatibility macros used across the existing codebase.
#define FXP_FROM_FLOAT(x, f)   (fxp_from_float_signed((float)(x), (uint8_t)(f)))
#define FXP_FROM_FLOAT_U(x, f) (fxp_from_float_unsigned((float)(x), (uint8_t)(f)))
#define FXP_TO_FLOAT(x, f)     (fxp_to_float((int64_t)(x), (uint8_t)(f)))

static inline q11_5_t fxp_imu_raw_from_float(float x)
{
    return fxp_sat_s16_from_s32(FXP_FROM_FLOAT(x, FXP_FRAC_IMU_RAW));
}

static inline uq10_6_t fxp_imu_l2a_from_float(float x)
{
    return fxp_sat_u16_from_u32(FXP_FROM_FLOAT_U(x, FXP_FRAC_IMU_L2A));
}

static inline uq5_11_t fxp_imu_l2g_from_float(float x)
{
    return fxp_sat_u16_from_u32(FXP_FROM_FLOAT_U(x, FXP_FRAC_IMU_L2G));
}

static inline int16_t fxp_audio_from_float(float x)
{
    return fxp_sat_s16_from_s32(FXP_FROM_FLOAT(x, FXP_FRAC_AUDIO_INPUT));
}

#define FXP_IMU_RAW_FROM_FLOAT(x) (fxp_imu_raw_from_float((float)(x)))
#define FXP_IMU_L2A_FROM_FLOAT(x) (fxp_imu_l2a_from_float((float)(x)))
#define FXP_IMU_L2G_FROM_FLOAT(x) (fxp_imu_l2g_from_float((float)(x)))
#define FXP_AUDIO_FROM_FLOAT(x)   (fxp_audio_from_float((float)(x)))

static inline void fxp_convert_imu_raw_buffer(const float *in, q11_5_t *out, int16_t len)
{
    for (int16_t i = 0; i < len; i++) {
        out[i] = FXP_IMU_RAW_FROM_FLOAT(in[i]);
    }
}
