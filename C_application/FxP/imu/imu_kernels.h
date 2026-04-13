#pragma once

#include <stdint.h>

#include <core/fxp_convert.h>
#include <core/fxp_math.h>
#include <core/fxp_qformats.h>
#include <core/fxp_types.h>

#define FXP_KURT_FISHER_Q34_30 ((int64_t)3 << FXP_FRAC_IMU_KURTOSIS_RAW)

// Tiny shared helpers stay inline in headers; heavy kernels live in .c files.
static inline uq2_14_t fxp_cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    if (rms == 0) return 0;
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

static inline float fxp_compute_zcr_raw(const q11_5_t *sig, int16_t len)
{
    if (len <= 1) return 0.0f;

    int crossings = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        if ((sig[i] ^ sig[i + 1]) < 0) crossings++;
    }
    return (float)crossings / (float)(len - 1);
}

static inline uint32_t fxp_azc_eps_raw(float eps)
{
    return FXP_FROM_FLOAT_U(eps, FXP_FRAC_IMU_RAW);
}

static inline uint32_t fxp_azc_eps_l2a(float eps)
{
    return FXP_FROM_FLOAT_U(eps, FXP_FRAC_IMU_L2A);
}

static inline uint32_t fxp_azc_eps_l2g(float eps)
{
    return FXP_FROM_FLOAT_U(eps, FXP_FRAC_IMU_L2G);
}

uq16_16_t fxp_get_rms_raw(const q11_5_t *sig, int16_t len);
uq13_3_t  fxp_get_rms_l2a(const uq10_6_t *sig, int16_t len);
uq7_9_t   fxp_get_rms_l2g(const uq5_11_t *sig, int16_t len);

uq9_23_t fxp_get_line_length_raw(const q11_5_t *sig, int16_t len);
uq7_9_t  fxp_get_line_length_l2g(const uq5_11_t *sig, int16_t len);

q34_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len);

uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len);

uq10_6_t fxp_l2_norm_accel_from_raw(q11_5_t ax, q11_5_t ay, q11_5_t az);
uq5_11_t fxp_l2_norm_gyro_from_raw(q11_5_t gx, q11_5_t gy, q11_5_t gz);
uq10_6_t fxp_l2_norm_accel(float ax, float ay, float az);
uq5_11_t fxp_l2_norm_gyro(float gx, float gy, float gz);

int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, float epsilon);
