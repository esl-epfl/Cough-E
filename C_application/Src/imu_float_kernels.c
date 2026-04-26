#include <stdlib.h>

#include <azc.h>
#include <helpers.h>
#include <imu_float_kernels.h>
#include <time_domain_feat.h>

#ifndef FXP_MODE

typedef void (*imu_float_kernel_fn)(const imu_float_view_t *sig, float *out, float param);

typedef struct {
    uint8_t feature_idx;
    imu_float_kernel_fn fn;
    float param;
} imu_float_kernel_desc_t;

#define AZC_IDX(i) ((uint8_t)(APPROXIMATE_ZERO_CROSSING + (i)))
#define AZC_EPS(i) ((float)(EPSILON_START + (EPSILON_STEP * (i))))

static void _line_length(const imu_float_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_line_length(sig->data, sig->len);
}

static void _zcr(const imu_float_view_t *sig, float *out, float param)
{
    (void)param;
    *out = compute_zrc(sig->data, sig->len);
}

static void _kurtosis(const imu_float_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_kurtosis(sig->data, sig->len);
}

static void _rms(const imu_float_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_rms(sig->data, sig->len);
}

static void _crest(const imu_float_view_t *sig, float *out, float param)
{
    (void)param;
    float rms = get_rms(sig->data, sig->len);
    float peak = get_max(sig->data, sig->len);
    *out = (rms > 0.0f) ? (peak / rms) : 0.0f;
}

static void _azc(const imu_float_view_t *sig, float *out, float param)
{
    *out = (float)azc_computation(sig->data, sig->len, param);
}

static const imu_float_kernel_desc_t k_float_table[] = {
    {LINE_LENGTH, _line_length, 0.0f},
    {ZERO_CROSSING_RATE_IMU, _zcr, 0.0f},
    {KURTOSIS, _kurtosis, 0.0f},
    {ROOT_MEANS_SQUARED_IMU, _rms, 0.0f},
    {CREST_FACTOR_IMU, _crest, 0.0f},
    {AZC_IDX(0), _azc, AZC_EPS(0)},
    {AZC_IDX(1), _azc, AZC_EPS(1)},
    {AZC_IDX(2), _azc, AZC_EPS(2)},
    {AZC_IDX(3), _azc, AZC_EPS(3)},
    {AZC_IDX(4), _azc, AZC_EPS(4)},
    {AZC_IDX(5), _azc, AZC_EPS(5)},
    {AZC_IDX(6), _azc, AZC_EPS(6)},
    {AZC_IDX(7), _azc, AZC_EPS(7)},
};

void imu_run_float_features(const int8_t *features_selector,
                            imu_float_view_t sig,
                            float *feats)
{
    for (size_t i = 0; i < sizeof(k_float_table) / sizeof(k_float_table[0]); i++) {
        const imu_float_kernel_desc_t *row = &k_float_table[i];
        if (features_selector[row->feature_idx] != 1) continue;
        row->fn(&sig, &feats[row->feature_idx], row->param);
    }
}

#endif
