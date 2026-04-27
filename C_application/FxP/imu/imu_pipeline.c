#include <stdio.h>
#include <stdlib.h>

#include <imu/imu_pipeline.h>

#ifdef FXP_MODE

/* -------------------------------------------------------------------------- */
/*  FxP kernels                                                               */
/* -------------------------------------------------------------------------- */

#define FXP_KURT_FISHER_Q34_30 ((int64_t)3 << FXP_FRAC_IMU_KURTOSIS_RAW)

static inline uq2_14_t _cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    if (rms == 0) return 0;
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

static inline uint64_t _shift_u64(uint64_t value, int shift)
{
    return (shift >= 0) ? (value << shift) : (value >> (-shift));
}

static int32_t _sample_as_s32(const void *sig, int16_t idx, uint8_t width_bits, uint8_t is_signed)
{
    if (width_bits == 16U) {
        return is_signed
            ? (int32_t)((const int16_t *)sig)[idx]
            : (int32_t)((const uint16_t *)sig)[idx];
    }
    return is_signed
        ? ((const int32_t *)sig)[idx]
        : (int32_t)((const uint32_t *)sig)[idx];
}

static uint32_t _rms(const void *sig, int16_t len,
                        uint8_t width_bits, uint8_t is_signed,
                        uint8_t pre_square_shift, int8_t post_mean_shift,
                        uint8_t result_bits)
{
    if (len <= 0) return 0;

    uint64_t sum64 = 0;
    uint32_t sum32 = 0;
    for (int16_t i = 0; i < len; i++) {
        int32_t sample = _sample_as_s32(sig, i, width_bits, is_signed);
        uint32_t sq = is_signed
            ? (uint32_t)fxp_mul_s32(sample, sample)
            : fxp_mul_u32((uint32_t)sample, (uint32_t)sample);
        sq >>= pre_square_shift;
        if (result_bits > 16U) {
            sum64 += (uint64_t)sq;
        } else {
            sum32 += sq;
        }
    }

    if (result_bits > 16U) {
        uint64_t mean = sum64 / (uint64_t)len;
        uint64_t shifted = _shift_u64(mean, post_mean_shift);
        return fxp_sat_u32_from_u64(fxp_sqrt64(shifted));
    }

    uint32_t mean = sum32 / (uint32_t)len;
    uint32_t shifted = (uint32_t)_shift_u64(mean, post_mean_shift);
    return fxp_sat_u16_from_u32(fxp_sqrt32(shifted));
}

static uint32_t _line_length(const void *sig, int16_t len,
                                uint8_t width_bits, uint8_t is_signed,
                                uint8_t diff_shift, uint8_t result_shift)
{
    if (len <= 1) return 0;

    uint32_t accum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        int32_t a = _sample_as_s32(sig, i + 1, width_bits, is_signed);
        int32_t b = _sample_as_s32(sig, i, width_bits, is_signed);
        accum += ((uint32_t)fxp_abs_s32(a - b)) >> diff_shift;
    }
    return (uint32_t)(((uint64_t)accum << result_shift) / (uint32_t)(len - 1));
}

static inline int32_t _kurt_mean(int32_t accum_q5, int16_t n)
{
    if (n <= 0) return 0;
    return (accum_q5 * 32) / (int32_t)n;
}

static inline int16_t _kurt_centered(q11_5_t sample_q5, int32_t mean_q10)
{
    return (int16_t)((((int32_t)sample_q5 * 32) - mean_q10 + 16) >> 5);
}

static q34_30_t _kurtosis_raw(const q11_5_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    int32_t sum_mean = 0;
    for (int16_t i = 0; i < len; i++) {
        sum_mean += (int32_t)sig[i];
    }
    int32_t mean_q10 = _kurt_mean(sum_mean, len);

    uint64_t sum_var = 0;
    for (int16_t i = 0; i < len; i++) {
        int16_t centered = _kurt_centered(sig[i], mean_q10);
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
        int16_t centered = _kurt_centered(sig[i], mean_q10);
        uint64_t c2 = (uint64_t)fxp_mul_s32(centered, centered);
        sum_x4 += fxp_mul_u64(c2, c2);
    }

    /* sum_x4 is UQ16.20 and denom is UQ20.44; shift by 24 to land in Q34.30. */
    uint64_t denom_shifted = (uint64_t)denom >> FXP_FRAC_IMU_KURTOSIS_RAW;
    if (denom_shifted == 0) return 0;

    q34_30_t normalized = (q34_30_t)((sum_x4 << 24) / denom_shifted);
    return normalized - FXP_KURT_FISHER_Q34_30;
}

static uq5_11_t _max_l2g(const uq5_11_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uq5_11_t max = sig[0];
    for (int16_t i = 1; i < len; i++) {
        if (sig[i] > max) max = sig[i];
    }
    return max;
}

uq10_6_t imu_l2_norm_accel_from_raw(q11_5_t ax, q11_5_t ay, q11_5_t az)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(ax, ax)
                 + (uint32_t)fxp_mul_s32(ay, ay)
                 + (uint32_t)fxp_mul_s32(az, az);

    return (uq10_6_t)fxp_sat_u16_from_u32(fxp_sqrt32(sum << 2));
}

uq5_11_t imu_l2_norm_gyro_from_raw(q11_5_t gx, q11_5_t gy, q11_5_t gz)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(gx, gx)
                 + (uint32_t)fxp_mul_s32(gy, gy)
                 + (uint32_t)fxp_mul_s32(gz, gz);

    return (uq5_11_t)fxp_sat_u16_from_u32((uint32_t)fxp_sqrt64((uint64_t)sum << 12));
}

typedef struct {
    int16_t first;
    int16_t last;
} azc_segment_t;

static inline int32_t _azc_diff(int32_t a, int32_t b, int16_t gap)
{
    return (gap == 0) ? 0 : (b - a) / (int32_t)gap;
}

static void _azc_interp(int16_t len, int16_t xf, int32_t yf,
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

static uint32_t _azc_max_vdist(const int32_t *sig, int16_t first,
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

    _azc_interp(len, first, sig[first], last, sig[last], intrp);

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

static int16_t *_azc_polygonal_approx(const int32_t *sig, int16_t len,
                                         uint32_t eps_fxp, int16_t *res_len)
{
    int16_t *res = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    azc_segment_t *stack = (azc_segment_t *)malloc((size_t)len * sizeof(azc_segment_t));
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
        uint32_t max_dist = _azc_max_vdist(sig, first, last, &mid);

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

static int _azc_qsort_cmp(const void *a, const void *b)
{
    return (int)(*(const int16_t *)a) - (int)(*(const int16_t *)b);
}

static int16_t _azc(const int32_t *sig, int16_t len, uint32_t eps_fxp)
{
    if (len <= 1) return 0;

    int16_t approx_len = 0;
    int16_t *idxs = _azc_polygonal_approx(sig, len, eps_fxp, &approx_len);
    if (idxs == NULL || approx_len <= 0) {
        free(idxs);
        return 0;
    }

    qsort(idxs, (size_t)approx_len, sizeof(int16_t), _azc_qsort_cmp);

    int16_t azc = 0;
    if (approx_len > 2) {
        int32_t prev = _azc_diff(sig[idxs[0]], sig[idxs[1]], (int16_t)(idxs[1] - idxs[0]));
        for (int16_t i = 1; i < approx_len - 1; i++) {
            int32_t cur = _azc_diff(sig[idxs[i]], sig[idxs[i + 1]], (int16_t)(idxs[i + 1] - idxs[i]));
            if ((prev > 0 && cur < 0) || (prev < 0 && cur > 0)) azc++;
            prev = cur;
        }
    }

    free(idxs);
    return azc;
}

static int16_t _azc_from_signal(const void *sig, int16_t len,
                                   uint8_t width_bits, uint8_t is_signed,
                                   uint32_t epsilon_q)
{
    if (len <= 0) return 0;

    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) {
        wide[i] = _sample_as_s32(sig, i, width_bits, is_signed);
    }
    int16_t result = _azc(wide, len, epsilon_q);
    free(wide);
    return result;
}

/*
 * Quantized epsilon values for AZC thresholds:
 *   eps = [0.3, 0.4, ..., 1.0]
 * encoded per signal carrier Q-format.
 */
static const uint32_t k_azc_eps_raw_q5[8] = {10U, 13U, 16U, 19U, 22U, 26U, 29U, 32U};
static const uint32_t k_azc_eps_l2a_q6[8] = {19U, 26U, 32U, 38U, 45U, 51U, 58U, 64U};
static const uint32_t k_azc_eps_l2g_q11[8] = {614U, 819U, 1024U, 1229U, 1434U, 1638U, 1843U, 2048U};

static void _run_raw_feature(const int8_t *features_selector,
                             const imu_sig_view_t *sig,
                             uint8_t local,
                             fxp_q16_t *out)
{
    switch (local) {
        case LINE_LENGTH:
            *out = fxp_q16_from_u32(
                _line_length(sig->data.raw_data, sig->len, 16U, 1U, 0U, 18U),
                FXP_FRAC_IMU_LINE_LENGTH_RAW);
            return;
        case KURTOSIS:
            *out = fxp_q16_from_s64(_kurtosis_raw(sig->data.raw_data, sig->len), FXP_FRAC_IMU_KURTOSIS_RAW);
            return;
        case ROOT_MEANS_SQUARED_IMU:
            *out = fxp_q16_from_u32(
                _rms(sig->data.raw_data, sig->len, 16U, 1U, 0U, 22, 32U),
                FXP_FRAC_IMU_RMS_RAW);
            return;
        default:
            if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
                uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
                int16_t azc = _azc_from_signal(sig->data.raw_data, sig->len, 16U, 1U, k_azc_eps_raw_q5[idx]);
                *out = fxp_q16_from_int((int32_t)azc);
            }
            (void)features_selector;
            return;
    }
}

static void _run_l2a_feature(const imu_sig_view_t *sig, uint8_t local, fxp_q16_t *out)
{
    if (local == ROOT_MEANS_SQUARED_IMU) {
        *out = fxp_q16_from_u32(
            _rms(sig->data.l2a_data, sig->len, 16U, 0U, 5U, -1, 16U),
            FXP_FRAC_IMU_RMS_L2A);
        return;
    }

    if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
        uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
        int16_t azc = _azc_from_signal(sig->data.l2a_data, sig->len, 16U, 0U, k_azc_eps_l2a_q6[idx]);
        *out = fxp_q16_from_int((int32_t)azc);
    }
}

static void _run_l2g_feature(const imu_sig_view_t *sig, uint8_t local, fxp_q16_t *out)
{
    switch (local) {
        case LINE_LENGTH:
            *out = fxp_q16_from_u32(
                fxp_sat_u16_from_u32(_line_length(sig->data.l2g_data, sig->len, 16U, 0U, 2U, 0U)),
                FXP_FRAC_IMU_LINE_LENGTH_L2G);
            return;
        case ROOT_MEANS_SQUARED_IMU:
            *out = fxp_q16_from_u32(
                _rms(sig->data.l2g_data, sig->len, 16U, 0U, 3U, -1, 16U),
                FXP_FRAC_IMU_RMS_L2G);
            return;
        case CREST_FACTOR_IMU: {
            uq7_9_t rms = (uq7_9_t)_rms(sig->data.l2g_data, sig->len, 16U, 0U, 3U, -1, 16U);
            uq5_11_t peak = _max_l2g(sig->data.l2g_data, sig->len);
            uq2_14_t cf = (rms > 0U) ? _cf_l2g_result(peak, rms) : 0U;
            *out = fxp_q16_from_u32((uint32_t)cf, FXP_FRAC_IMU_CREST_L2G);
            return;
        }
        default:
            if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
                uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
                int16_t azc = _azc_from_signal(sig->data.l2g_data, sig->len, 16U, 0U, k_azc_eps_l2g_q11[idx]);
                *out = fxp_q16_from_int((int32_t)azc);
            }
            return;
    }
}

void imu_run_features_q16(const int8_t *features_selector, imu_sig_view_t sig, fxp_q16_t *feats_q16)
{
    for (uint8_t local = 0U; local < Num_imu_feat_families; local++) {
        if (features_selector[local] != 1) continue;

        switch (sig.kind) {
            case IMU_SIG_KIND_RAW:
                _run_raw_feature(features_selector, &sig, local, &feats_q16[local]);
                break;
            case IMU_SIG_KIND_L2A:
                _run_l2a_feature(&sig, local, &feats_q16[local]);
                break;
            case IMU_SIG_KIND_L2G:
                _run_l2g_feature(&sig, local, &feats_q16[local]);
                break;
            default:
                fprintf(stderr, "IMU dispatch (Q16): unsupported signal kind %d.\n", (int)sig.kind);
                abort();
        }
    }
}
#endif
