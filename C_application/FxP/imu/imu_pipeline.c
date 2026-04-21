#include <stdio.h>
#include <stdlib.h>

#include <imu/imu_pipeline.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>

typedef void (*imu_kernel_fn)(const imu_sig_view_t *sig, float *out, float param);

typedef struct {
    uint8_t feature_idx;
    imu_kernel_fn fn;
    float param;
} imu_kernel_desc_t;

/* -------------------------------------------------------------------------- */
/*  Float kernels (legacy/non-FxP path)                                       */
/* -------------------------------------------------------------------------- */

static void kern_float_line_length(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_line_length((float *)sig->data.float_data, sig->len);
}

static void kern_float_zcr(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = compute_zrc((float *)sig->data.float_data, sig->len);
}

static void kern_float_kurtosis(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_kurtosis((float *)sig->data.float_data, sig->len);
}

static void kern_float_rms(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_rms((float *)sig->data.float_data, sig->len);
}

static void kern_float_crest(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    float rms = get_rms((float *)sig->data.float_data, sig->len);
    float peak = get_max((float *)sig->data.float_data, sig->len);
    *out = (rms > 0.0f) ? (peak / rms) : 0.0f;
}

static void kern_float_azc(const imu_sig_view_t *sig, float *out, float param)
{
    *out = (float)azc_computation((float *)sig->data.float_data, sig->len, param);
}

#define AZC_IDX(i) ((uint8_t)(APPROXIMATE_ZERO_CROSSING + (i)))
#define AZC_EPS(i) ((float)(EPSILON_START + (EPSILON_STEP * (i))))

static const imu_kernel_desc_t k_float_table[] = {
    {LINE_LENGTH, kern_float_line_length, 0.0f},
    {ZERO_CROSSING_RATE_IMU, kern_float_zcr, 0.0f},
    {KURTOSIS, kern_float_kurtosis, 0.0f},
    {ROOT_MEANS_SQUARED_IMU, kern_float_rms, 0.0f},
    {CREST_FACTOR_IMU, kern_float_crest, 0.0f},
    {AZC_IDX(0), kern_float_azc, AZC_EPS(0)},
    {AZC_IDX(1), kern_float_azc, AZC_EPS(1)},
    {AZC_IDX(2), kern_float_azc, AZC_EPS(2)},
    {AZC_IDX(3), kern_float_azc, AZC_EPS(3)},
    {AZC_IDX(4), kern_float_azc, AZC_EPS(4)},
    {AZC_IDX(5), kern_float_azc, AZC_EPS(5)},
    {AZC_IDX(6), kern_float_azc, AZC_EPS(6)},
    {AZC_IDX(7), kern_float_azc, AZC_EPS(7)},
};

#ifdef FXP_MODE

/* -------------------------------------------------------------------------- */
/*  FxP kernels                                                               */
/* -------------------------------------------------------------------------- */

static inline uint64_t fxp_shift_u64(uint64_t value, int shift)
{
    return (shift >= 0) ? (value << shift) : (value >> (-shift));
}

static inline uint32_t fxp_shift_u32(uint32_t value, int shift)
{
    return (shift >= 0) ? (value << shift) : (value >> (-shift));
}

uq16_16_t fxp_get_rms_raw(const q11_5_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uint64_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uint32_t sq = (uint32_t)fxp_mul_s32((int32_t)sig[i], (int32_t)sig[i]);
        sum += (uint64_t)sq;
    }

    uint64_t mean = sum / (uint64_t)len;
    uint64_t shifted = fxp_shift_u64(mean, 22);
    return (uq16_16_t)fxp_sat_u32_from_u64(fxp_sqrt64(shifted));
}

uq13_3_t fxp_get_rms_l2a(const uq10_6_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uint32_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uint32_t sq = fxp_mul_u32((uint32_t)sig[i], (uint32_t)sig[i]);
        sum += (sq >> 5);
    }

    uint32_t mean = sum / (uint32_t)len;
    uint32_t shifted = fxp_shift_u32(mean, -1);
    return (uq13_3_t)fxp_sat_u16_from_u32(fxp_sqrt32(shifted));
}

uq7_9_t fxp_get_rms_l2g(const uq5_11_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uint32_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uint32_t sq = fxp_mul_u32((uint32_t)sig[i], (uint32_t)sig[i]);
        sum += (sq >> 3);
    }

    uint32_t mean = sum / (uint32_t)len;
    uint32_t shifted = fxp_shift_u32(mean, -1);
    return (uq7_9_t)fxp_sat_u16_from_u32(fxp_sqrt32(shifted));
}

static inline uint32_t fxp_linelen_diff_to_accum(int32_t diff, uint8_t sr)
{
    return ((uint32_t)fxp_abs_s32(diff)) >> sr;
}

static inline uint32_t fxp_linelen_result(uint32_t accum, int16_t denom_len, uint8_t sl)
{
    if (denom_len <= 0) return 0;
    return (uint32_t)(((uint64_t)accum << sl) / (uint32_t)denom_len);
}

uq9_23_t fxp_get_line_length_raw(const q11_5_t *sig, int16_t len)
{
    if (len <= 1) return 0;

    uint32_t accum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        int32_t diff = (int32_t)sig[i + 1] - (int32_t)sig[i];
        accum += fxp_linelen_diff_to_accum(diff, 0);
    }
    return (uq9_23_t)fxp_linelen_result(accum, len - 1, 18);
}

uq7_9_t fxp_get_line_length_l2g(const uq5_11_t *sig, int16_t len)
{
    if (len <= 1) return 0;

    uint32_t accum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        int32_t diff = (int32_t)sig[i + 1] - (int32_t)sig[i];
        accum += fxp_linelen_diff_to_accum(diff, 2);
    }
    return (uq7_9_t)fxp_sat_u16_from_u32(fxp_linelen_result(accum, len - 1, 0));
}

static inline int32_t fxp_kurt_mean_q10(int32_t accum_q5, int16_t n)
{
    if (n <= 0) return 0;
    return (accum_q5 * 32) / (int32_t)n;
}

static inline int16_t fxp_kurt_centred_q5(q11_5_t sample_q5, int32_t mean_q10)
{
    return (int16_t)((((int32_t)sample_q5 * 32) - mean_q10 + 16) >> 5);
}

q34_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    int32_t sum_mean = 0;
    for (int16_t i = 0; i < len; i++) {
        sum_mean += (int32_t)sig[i];
    }
    int32_t mean_q10 = fxp_kurt_mean_q10(sum_mean, len);

    uint64_t sum_var = 0;
    for (int16_t i = 0; i < len; i++) {
        int16_t centered = fxp_kurt_centred_q5(sig[i], mean_q10);
        uint32_t c2_q10 = (uint32_t)fxp_mul_s32(centered, centered);
        sum_var += ((uint64_t)c2_q10 << 12);
    }

    uq10_22_t variance = (uq10_22_t)(sum_var / (uint64_t)len);
    uq5_11_t stddev = (uq5_11_t)fxp_sqrt32(variance);

    uint32_t std2 = fxp_mul_u32(stddev, stddev);
    uq20_44_t std4 = fxp_mul_u64(std2, std2);
    uq20_44_t denom = (uq20_44_t)((uint64_t)len * std4);
    if (denom == 0) return 0;

    uint64_t sum_x4 = 0;
    for (int16_t i = 0; i < len; i++) {
        int16_t centered = fxp_kurt_centred_q5(sig[i], mean_q10);
        uint64_t c2 = (uint64_t)fxp_mul_s32(centered, centered);
        sum_x4 += fxp_mul_u64(c2, c2);
    }

    /* sum_x4 is UQ16.20 and denom is UQ20.44; shift by 24 to land in Q34.30. */
    uint64_t denom_shifted = (uint64_t)denom >> FXP_FRAC_IMU_KURTOSIS_RAW;
    if (denom_shifted == 0) return 0;

    q34_30_t normalized = (q34_30_t)((sum_x4 << 24) / denom_shifted);
    return normalized - FXP_KURT_FISHER_Q34_30;
}

uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uq5_11_t max = sig[0];
    for (int16_t i = 1; i < len; i++) {
        if (sig[i] > max) max = sig[i];
    }
    return max;
}

uq10_6_t fxp_l2_norm_accel_from_raw(q11_5_t ax, q11_5_t ay, q11_5_t az)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(ax, ax)
                 + (uint32_t)fxp_mul_s32(ay, ay)
                 + (uint32_t)fxp_mul_s32(az, az);

    return (uq10_6_t)fxp_sat_u16_from_u32(fxp_sqrt32(sum << 2));
}

uq5_11_t fxp_l2_norm_gyro_from_raw(q11_5_t gx, q11_5_t gy, q11_5_t gz)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(gx, gx)
                 + (uint32_t)fxp_mul_s32(gy, gy)
                 + (uint32_t)fxp_mul_s32(gz, gz);

    return (uq5_11_t)fxp_sat_u16_from_u32((uint32_t)fxp_sqrt64((uint64_t)sum << 12));
}

typedef struct {
    int16_t first;
    int16_t last;
} fxp_azc_segment_t;

static inline int32_t fxp_azc_diff(int32_t a, int32_t b, int16_t gap)
{
    return (gap == 0) ? 0 : (b - a) / (int32_t)gap;
}

static void fxp_azc_interp(int16_t len, int16_t xf, int32_t yf,
                           int16_t xl, int32_t yl, int32_t *res)
{
    res[0] = yf;
    res[len - 1] = yl;

    int32_t dy = yl - yf;
    int16_t dx = xl - xf;
    for (int16_t i = 1; i < len - 1; i++) {
        res[i] = yf + (dy * i) / dx;
    }
}

static uint32_t fxp_azc_max_vdist(const int32_t *sig, int16_t first,
                                  int16_t last, int16_t *idx)
{
    if (first == last) {
        *idx = first;
        return 0;
    }

    int16_t len = last - first + 1;
    int32_t *intrp = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (intrp == NULL) {
        *idx = first;
        return 0;
    }

    fxp_azc_interp(len, first, sig[first], last, sig[last], intrp);

    uint32_t max_dist = 0;
    *idx = first;
    for (int16_t i = 0; i < len; i++) {
        uint32_t d = (uint32_t)fxp_abs_s32(sig[first + i] - intrp[i]);
        if (d > max_dist) {
            max_dist = d;
            *idx = first + i;
        }
    }

    free(intrp);
    return max_dist;
}

static int16_t *fxp_azc_polygonal_approx(const int32_t *sig, int16_t len,
                                         uint32_t eps_fxp, int16_t *res_len)
{
    int16_t *res = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    fxp_azc_segment_t *stack = (fxp_azc_segment_t *)malloc((size_t)len * sizeof(fxp_azc_segment_t));
    if (res == NULL || stack == NULL) {
        free(res);
        free(stack);
        *res_len = 0;
        return NULL;
    }

    int16_t found = 0;
    stack[0].first = 0;
    stack[0].last = len - 1;
    int16_t next = 0;

    while (next >= 0) {
        int16_t first = stack[next].first;
        int16_t last = stack[next].last;
        next--;

        int16_t mid;
        uint32_t max_dist = fxp_azc_max_vdist(sig, first, last, &mid);

        if (max_dist > eps_fxp) {
            stack[next + 1].first = first;
            stack[next + 1].last = mid;
            stack[next + 2].first = mid;
            stack[next + 2].last = last;
            next += 2;
        } else {
            int16_t add_first = 1;
            int16_t add_last = 1;

            for (int16_t j = 0; j < found; j++) {
                if (first == res[j]) add_first = 0;
                if (last == res[j]) add_last = 0;
            }

            if (add_first) res[found++] = first;
            if (add_last) res[found++] = last;
        }
    }

    free(stack);
    *res_len = found;
    return res;
}

static int fxp_azc_qsort_cmp(const void *a, const void *b)
{
    return (int)(*(const int16_t *)a) - (int)(*(const int16_t *)b);
}

static int16_t fxp_azc_impl(const int32_t *sig, int16_t len, uint32_t eps_fxp)
{
    if (len <= 1) return 0;

    int16_t approx_len = 0;
    int16_t *idxs = fxp_azc_polygonal_approx(sig, len, eps_fxp, &approx_len);
    if (idxs == NULL || approx_len <= 0) {
        free(idxs);
        return 0;
    }

    qsort(idxs, (size_t)approx_len, sizeof(int16_t), fxp_azc_qsort_cmp);

    int16_t azc = 0;
    if (approx_len > 2) {
        int32_t prev = fxp_azc_diff(sig[idxs[0]], sig[idxs[1]], (int16_t)(idxs[1] - idxs[0]));
        for (int16_t i = 1; i < approx_len - 1; i++) {
            int32_t cur = fxp_azc_diff(sig[idxs[i]], sig[idxs[i + 1]], (int16_t)(idxs[i + 1] - idxs[i]));
            if ((prev > 0 && cur < 0) || (prev < 0 && cur > 0)) azc++;
            prev = cur;
        }
    }

    free(idxs);
    return azc;
}

int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, uint32_t epsilon_q5)
{
    if (len <= 0) return 0;

    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, epsilon_q5);
    free(wide);
    return result;
}

int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, uint32_t epsilon_q6)
{
    if (len <= 0) return 0;

    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, epsilon_q6);
    free(wide);
    return result;
}

int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, uint32_t epsilon_q11)
{
    if (len <= 0) return 0;

    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, epsilon_q11);
    free(wide);
    return result;
}

/* -------------------------------------------------------------------------- */
/*  FxP dispatch wrappers                                                      */
/* -------------------------------------------------------------------------- */

typedef void (*imu_kernel_fn_q16)(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q);

typedef struct {
    uint8_t feature_idx;
    imu_kernel_fn_q16 fn;
    uint32_t param_q;
} imu_kernel_desc_q16_t;

/*
 * Quantized epsilon values for AZC thresholds:
 *   eps = [0.3, 0.4, ..., 1.0]
 * encoded per signal carrier Q-format.
 */
static const uint32_t k_azc_eps_raw_q5[8] = {10U, 13U, 16U, 19U, 22U, 26U, 29U, 32U};
static const uint32_t k_azc_eps_l2a_q6[8] = {19U, 26U, 32U, 38U, 45U, 51U, 58U, 64U};
static const uint32_t k_azc_eps_l2g_q11[8] = {614U, 819U, 1024U, 1229U, 1434U, 1638U, 1843U, 2048U};

static void kern_raw_line_length_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_get_line_length_raw(sig->data.raw_data, sig->len), FXP_FRAC_IMU_LINE_LENGTH_RAW);
}

static void kern_raw_zcr_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_compute_zcr_raw_q16(sig->data.raw_data, sig->len), 16U);
}

static void kern_raw_kurtosis_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_s64(fxp_get_kurtosis_raw(sig->data.raw_data, sig->len), FXP_FRAC_IMU_KURTOSIS_RAW);
}

static void kern_raw_rms_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_get_rms_raw(sig->data.raw_data, sig->len), FXP_FRAC_IMU_RMS_RAW);
}

static void kern_raw_azc_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    int16_t azc = fxp_azc_computation_raw(sig->data.raw_data, sig->len, param_q);
    *out = fxp_q16_from_int((int32_t)azc);
}

static void kern_l2a_rms_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_get_rms_l2a(sig->data.l2a_data, sig->len), FXP_FRAC_IMU_RMS_L2A);
}

static void kern_l2a_azc_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    int16_t azc = fxp_azc_computation_l2a(sig->data.l2a_data, sig->len, param_q);
    *out = fxp_q16_from_int((int32_t)azc);
}

static void kern_l2g_line_length_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_get_line_length_l2g(sig->data.l2g_data, sig->len), FXP_FRAC_IMU_LINE_LENGTH_L2G);
}

static void kern_l2g_rms_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    *out = fxp_q16_from_u32(fxp_get_rms_l2g(sig->data.l2g_data, sig->len), FXP_FRAC_IMU_RMS_L2G);
}

static void kern_l2g_crest_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    (void)param_q;
    uq7_9_t rms = fxp_get_rms_l2g(sig->data.l2g_data, sig->len);
    uq5_11_t peak = fxp_get_max_l2g(sig->data.l2g_data, sig->len);
    uq2_14_t cf = (rms > 0U) ? fxp_cf_l2g_result(peak, rms) : 0U;
    *out = fxp_q16_from_u32((uint32_t)cf, FXP_FRAC_IMU_CREST_L2G);
}

static void kern_l2g_azc_q16(const imu_sig_view_t *sig, fxp_q16_t *out, uint32_t param_q)
{
    int16_t azc = fxp_azc_computation_l2g(sig->data.l2g_data, sig->len, param_q);
    *out = fxp_q16_from_int((int32_t)azc);
}

static const imu_kernel_desc_q16_t k_raw_table_q16[] = {
    {LINE_LENGTH, kern_raw_line_length_q16, 0U},
    {ZERO_CROSSING_RATE_IMU, kern_raw_zcr_q16, 0U},
    {KURTOSIS, kern_raw_kurtosis_q16, 0U},
    {ROOT_MEANS_SQUARED_IMU, kern_raw_rms_q16, 0U},
    {AZC_IDX(0), kern_raw_azc_q16, k_azc_eps_raw_q5[0]},
    {AZC_IDX(1), kern_raw_azc_q16, k_azc_eps_raw_q5[1]},
    {AZC_IDX(2), kern_raw_azc_q16, k_azc_eps_raw_q5[2]},
    {AZC_IDX(3), kern_raw_azc_q16, k_azc_eps_raw_q5[3]},
    {AZC_IDX(4), kern_raw_azc_q16, k_azc_eps_raw_q5[4]},
    {AZC_IDX(5), kern_raw_azc_q16, k_azc_eps_raw_q5[5]},
    {AZC_IDX(6), kern_raw_azc_q16, k_azc_eps_raw_q5[6]},
    {AZC_IDX(7), kern_raw_azc_q16, k_azc_eps_raw_q5[7]},
};

static const imu_kernel_desc_q16_t k_l2a_table_q16[] = {
    {ROOT_MEANS_SQUARED_IMU, kern_l2a_rms_q16, 0U},
    {AZC_IDX(0), kern_l2a_azc_q16, k_azc_eps_l2a_q6[0]},
    {AZC_IDX(1), kern_l2a_azc_q16, k_azc_eps_l2a_q6[1]},
    {AZC_IDX(2), kern_l2a_azc_q16, k_azc_eps_l2a_q6[2]},
    {AZC_IDX(3), kern_l2a_azc_q16, k_azc_eps_l2a_q6[3]},
    {AZC_IDX(4), kern_l2a_azc_q16, k_azc_eps_l2a_q6[4]},
    {AZC_IDX(5), kern_l2a_azc_q16, k_azc_eps_l2a_q6[5]},
    {AZC_IDX(6), kern_l2a_azc_q16, k_azc_eps_l2a_q6[6]},
    {AZC_IDX(7), kern_l2a_azc_q16, k_azc_eps_l2a_q6[7]},
};

static const imu_kernel_desc_q16_t k_l2g_table_q16[] = {
    {LINE_LENGTH, kern_l2g_line_length_q16, 0U},
    {ROOT_MEANS_SQUARED_IMU, kern_l2g_rms_q16, 0U},
    {CREST_FACTOR_IMU, kern_l2g_crest_q16, 0U},
    {AZC_IDX(0), kern_l2g_azc_q16, k_azc_eps_l2g_q11[0]},
    {AZC_IDX(1), kern_l2g_azc_q16, k_azc_eps_l2g_q11[1]},
    {AZC_IDX(2), kern_l2g_azc_q16, k_azc_eps_l2g_q11[2]},
    {AZC_IDX(3), kern_l2g_azc_q16, k_azc_eps_l2g_q11[3]},
    {AZC_IDX(4), kern_l2g_azc_q16, k_azc_eps_l2g_q11[4]},
    {AZC_IDX(5), kern_l2g_azc_q16, k_azc_eps_l2g_q11[5]},
    {AZC_IDX(6), kern_l2g_azc_q16, k_azc_eps_l2g_q11[6]},
    {AZC_IDX(7), kern_l2g_azc_q16, k_azc_eps_l2g_q11[7]},
};

#endif

/* -------------------------------------------------------------------------- */
/*  Dispatch                                                                  */
/* -------------------------------------------------------------------------- */

static void imu_get_table(imu_sig_kind_t kind, const imu_kernel_desc_t **table, size_t *table_len)
{
    switch (kind) {
        case IMU_SIG_KIND_FLOAT:
            *table = k_float_table;
            *table_len = sizeof(k_float_table) / sizeof(k_float_table[0]);
            return;
        default:
            fprintf(stderr, "IMU dispatch (float): unsupported signal kind %d.\n", (int)kind);
            abort();
    }
}

#ifdef FXP_MODE
static void imu_get_table_q16(imu_sig_kind_t kind, const imu_kernel_desc_q16_t **table, size_t *table_len)
{
    switch (kind) {
        case IMU_SIG_KIND_RAW:
            *table = k_raw_table_q16;
            *table_len = sizeof(k_raw_table_q16) / sizeof(k_raw_table_q16[0]);
            return;
        case IMU_SIG_KIND_L2A:
            *table = k_l2a_table_q16;
            *table_len = sizeof(k_l2a_table_q16) / sizeof(k_l2a_table_q16[0]);
            return;
        case IMU_SIG_KIND_L2G:
            *table = k_l2g_table_q16;
            *table_len = sizeof(k_l2g_table_q16) / sizeof(k_l2g_table_q16[0]);
            return;
        default:
            fprintf(stderr, "IMU dispatch (Q16): unsupported signal kind %d.\n", (int)kind);
            abort();
    }
}
#endif

void imu_run_feature_table(const int8_t *features_selector, imu_sig_view_t sig, float *feats)
{
    const imu_kernel_desc_t *table = NULL;
    size_t table_len = 0;
    imu_get_table(sig.kind, &table, &table_len);

    for (size_t i = 0; i < table_len; i++) {
        const imu_kernel_desc_t *row = &table[i];
        if (features_selector[row->feature_idx] != 1) continue;
        row->fn(&sig, &feats[row->feature_idx], row->param);
    }
}

#ifdef FXP_MODE
void imu_run_feature_table_q16(const int8_t *features_selector, imu_sig_view_t sig, fxp_q16_t *feats_q16)
{
    const imu_kernel_desc_q16_t *table = NULL;
    size_t table_len = 0;
    imu_get_table_q16(sig.kind, &table, &table_len);

    for (size_t i = 0; i < table_len; i++) {
        const imu_kernel_desc_q16_t *row = &table[i];
        if (features_selector[row->feature_idx] != 1) continue;
        row->fn(&sig, &feats_q16[row->feature_idx], row->param_q);
    }
}
#endif
