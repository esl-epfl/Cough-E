#pragma once

#include <inttypes.h>

#include <imu_features.h>

#ifdef FXP_MODE
#include <core/fxp_core.h>

/* -------------------------------------------------------------------------- */
/*  Typed IMU signal carriers                                                 */
/* -------------------------------------------------------------------------- */

typedef struct { q11_5_t *data; int16_t len; } imu_sig_raw_t;
typedef struct { uq10_6_t *data; int16_t len; } imu_sig_l2a_t;
typedef struct { uq5_11_t *data; int16_t len; } imu_sig_l2g_t;

typedef enum {
    IMU_SIG_KIND_RAW = 0,
    IMU_SIG_KIND_L2A,
    IMU_SIG_KIND_L2G,
} imu_sig_kind_t;

typedef struct {
    imu_sig_kind_t kind;
    int16_t len;
    union {
        const q11_5_t  *raw_data;
        const uq10_6_t *l2a_data;
        const uq5_11_t *l2g_data;
    } data;
} imu_sig_view_t;

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

/* -------------------------------------------------------------------------- */
/*  IMU feature dispatch entry points                                          */
/* -------------------------------------------------------------------------- */

void imu_run_features_q16(const int8_t *features_selector, imu_sig_view_t sig, fxp_q16_t *feats_q16);

uq10_6_t imu_l2_norm_accel_from_raw(q11_5_t ax, q11_5_t ay, q11_5_t az);
uq5_11_t imu_l2_norm_gyro_from_raw(q11_5_t gx, q11_5_t gy, q11_5_t gz);

#endif
