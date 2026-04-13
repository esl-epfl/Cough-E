#pragma once

#include <inttypes.h>
#include <imu_features.h>

#ifdef FXP_MODE
#include <imu/imu_kernels.h>
#endif

// Table-driven typed IMU signal carriers.
typedef struct { float    *data; int16_t len; } imu_sig_float_t;

#ifdef FXP_MODE
typedef struct { q11_5_t  *data; int16_t len; } imu_sig_raw_t;
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

// Runs the typed kernel registry for one signal view.
// Fails fast if a requested feature has no valid kernel row for the signal kind.
void imu_run_feature_table(const int8_t *features_selector, imu_sig_view_t sig, float *feats);
