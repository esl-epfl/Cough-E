#pragma once

#include <inttypes.h>

#include <imu_features.h>

#ifdef FXP_MODE
#include <core/fxp_core.h>
#endif

/* -------------------------------------------------------------------------- */
/*  Typed IMU signal carriers                                                 */
/* -------------------------------------------------------------------------- */

typedef struct { float *data; int16_t len; } imu_sig_float_t;

#ifdef FXP_MODE
typedef struct { q11_5_t *data; int16_t len; } imu_sig_raw_t;
typedef struct { uq10_6_t *data; int16_t len; } imu_sig_l2a_t;
typedef struct { uq5_11_t *data; int16_t len; } imu_sig_l2g_t;
#endif

typedef enum {
    IMU_SIG_KIND_FLOAT = 0,
#ifdef FXP_MODE
    IMU_SIG_KIND_RAW,
    IMU_SIG_KIND_L2A,
    IMU_SIG_KIND_L2G,
#endif
} imu_sig_kind_t;

typedef struct {
    imu_sig_kind_t kind;
    int16_t len;
    union {
        const float *float_data;
#ifdef FXP_MODE
        const q11_5_t  *raw_data;
        const uq10_6_t *l2a_data;
        const uq5_11_t *l2g_data;
#endif
    } data;
} imu_sig_view_t;

static inline imu_sig_view_t imu_view_from_float(imu_sig_float_t s)
{
    imu_sig_view_t v;
    v.kind = IMU_SIG_KIND_FLOAT;
    v.len = s.len;
    v.data.float_data = s.data;
    return v;
}

#ifdef FXP_MODE
static inline imu_sig_view_t imu_view_from_raw(imu_sig_raw_t s)
{
    imu_sig_view_t v;
    v.kind = IMU_SIG_KIND_RAW;
    v.len = s.len;
    v.data.raw_data = s.data;
    return v;
}

static inline imu_sig_view_t imu_view_from_l2a(imu_sig_l2a_t s)
{
    imu_sig_view_t v;
    v.kind = IMU_SIG_KIND_L2A;
    v.len = s.len;
    v.data.l2a_data = s.data;
    return v;
}

static inline imu_sig_view_t imu_view_from_l2g(imu_sig_l2g_t s)
{
    imu_sig_view_t v;
    v.kind = IMU_SIG_KIND_L2G;
    v.len = s.len;
    v.data.l2g_data = s.data;
    return v;
}
#endif

/* -------------------------------------------------------------------------- */
/*  IMU feature dispatch entry points                                          */
/* -------------------------------------------------------------------------- */

void imu_run_feature_table(const int8_t *features_selector, imu_sig_view_t sig, float *feats);

#ifdef FXP_MODE
void imu_run_feature_table_q16(const int8_t *features_selector, imu_sig_view_t sig, fxp_q16_t *feats_q16);

/* -------------------------------------------------------------------------- */
/*  FxP IMU kernels used by dispatch and feature extraction                    */
/* -------------------------------------------------------------------------- */

#define FXP_KURT_FISHER_Q34_30 ((int64_t)3 << FXP_FRAC_IMU_KURTOSIS_RAW)

static inline uq2_14_t fxp_cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    if (rms == 0) return 0;
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

static inline uq0_16_t fxp_compute_zcr_raw_q16(const q11_5_t *sig, int16_t len)
{
    if (len <= 1) return 0U;

    int crossings = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        if ((sig[i] ^ sig[i + 1]) < 0) crossings++;
    }
    return (uq0_16_t)fxp_uq0_16_ratio((uint32_t)crossings, (uint32_t)(len - 1));
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

int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, uint32_t epsilon_q5);
int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, uint32_t epsilon_q6);
int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, uint32_t epsilon_q11);

#endif
