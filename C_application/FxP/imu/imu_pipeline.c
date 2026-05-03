#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <imu/imu_pipeline.h>

#ifdef FXP_MODE

/* -------------------------------------------------------------------------- */
/*  FxP kernels                                                               */
/* -------------------------------------------------------------------------- */

#define FXP_KURT_FISHER_Q10_22 ((int32_t)3 << FXP_FRAC_IMU_KURTOSIS_RAW)

static inline uq2_14_t _cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    if (rms == 0) return 0;
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
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

static uint16_t _rms16(const void *sig, int16_t len,
                       uint8_t width_bits, uint8_t is_signed,
                       uint8_t pre_square_shift)
{
    if (len <= 0) return 0;

    uint32_t sum32 = 0;
    for (int16_t i = 0; i < len; i++) {
        int32_t sample = _sample_as_s32(sig, i, width_bits, is_signed);
        uint32_t sq = is_signed
            ? (uint32_t)fxp_mul_s32(sample, sample)
            : fxp_mul_u32((uint32_t)sample, (uint32_t)sample);
        sq >>= pre_square_shift;
        sum32 = fxp_sat_add_u32(sum32, sq);
    }

    uint32_t sqrt_input = (sum32 / (uint16_t)len) >> 1U;
    return fxp_sat_u16_from_u32(fxp_sqrt32(sqrt_input));
}

static uq13_3_t _rms_raw(const q11_5_t *sig, int16_t len)
{
    return (uq13_3_t)_rms16(sig, len, 16U, 1U, 3U);
}

static uq13_3_t _rms_l2a(const uq10_6_t *sig, int16_t len)
{
    return (uq13_3_t)_rms16(sig, len, 16U, 0U, 5U);
}

static uq7_9_t _rms_l2g(const uq5_11_t *sig, int16_t len)
{
    return (uq7_9_t)_rms16(sig, len, 16U, 0U, 3U);
}

static uq7_9_t _line_length(const void *sig, int16_t len,
                            uint8_t is_signed, uint8_t input_frac)
{
    if (len <= 1) return 0;

    const int16_t *sig_s16 = (const int16_t *)sig;
    const uint16_t *sig_u16 = (const uint16_t *)sig;
    uint16_t accum_q9 = 0;
    uint8_t shift;

    if (input_frac < FXP_FRAC_IMU_LINE_LENGTH_RAW) {
        shift = (uint8_t)(FXP_FRAC_IMU_LINE_LENGTH_RAW - input_frac);
    } else {
        shift = (uint8_t)(input_frac - FXP_FRAC_IMU_LINE_LENGTH_RAW);
    }

    for (int16_t i = 0; i < len - 1; i++) {
        uint16_t diff = is_signed
            ? fxp_abs_delta_s16(sig_s16[i + 1], sig_s16[i])
            : fxp_abs_delta_u16(sig_u16[i + 1], sig_u16[i]);
        uint16_t diff_q9 = (input_frac < FXP_FRAC_IMU_LINE_LENGTH_RAW)
            ? fxp_sat_sl_u16(diff, shift)
            : (uint16_t)(diff >> shift);
        accum_q9 = fxp_sat_add_u16(accum_q9, diff_q9);
    }

    return (uq7_9_t)(accum_q9 / (uint16_t)(len - 1));
}

static inline q11_5_t _kurt_mean(q11_5_t accum_q5, int16_t n)
{
    if (n <= 0) return 0;
    return (q11_5_t)(accum_q5 / n);
}

static inline q11_5_t _kurt_centered(q11_5_t sample_q5, q11_5_t mean_q5)
{
    return (q11_5_t)(sample_q5 - mean_q5);
}

static q10_22_t _kurtosis(const q11_5_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    q11_5_t sum_mean = 0;
    for (int16_t i = 0; i < len; i++) {
        sum_mean = fxp_sat_add_s16(sum_mean, sig[i]);
    }
    q11_5_t mean_q5 = _kurt_mean(sum_mean, len);

    uq10_22_t sum_var = 0;
    for (int16_t i = 0; i < len; i++) {
        int16_t centered = _kurt_centered(sig[i], mean_q5);
        uint32_t c2_q10 = (uint32_t)fxp_mul_s32(centered, centered);
        uint32_t c2_q22 = fxp_sat_sl_u32(c2_q10, 12U);
        sum_var = fxp_sat_add_u32(sum_var, c2_q22);
    }

    uq10_22_t variance = (uq10_22_t)(sum_var / (uint16_t)len);
    uq5_11_t stddev = (uq5_11_t)fxp_sqrt32(variance);

    uint32_t std2 = fxp_mul_u32(stddev, stddev);
    uq20_44_t std4 = fxp_mul_u64(std2, std2);
    uq20_44_t denom = (uq20_44_t)((uint64_t)len * std4);
    if (denom == 0) return 0;

    uint64_t sum_x4 = 0;
    for (int16_t i = 0; i < len; i++) {
        int16_t centered = _kurt_centered(sig[i], mean_q5);
        uint64_t c2 = (uint64_t)fxp_mul_s32(centered, centered);
        sum_x4 += fxp_mul_u64(c2, c2);
    }

    uint64_t denom_shifted = (uint64_t)denom >> 30U;
    if (denom_shifted == 0) return 0;

    uint64_t normalized = (sum_x4 << 16) / denom_shifted;
    if (normalized > (uint64_t)INT32_MAX) normalized = (uint64_t)INT32_MAX;
    return (q10_22_t)((int32_t)normalized - FXP_KURT_FISHER_Q10_22);
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

    if (sum > (UINT32_MAX >> 12U)) {
        return UINT16_MAX;
    }
    return (uq5_11_t)fxp_sat_u16_from_u32(fxp_sqrt32(sum << 12U));
}

typedef struct {
    int16_t first;
    int16_t last;
} azc_segment_t;

static inline int32_t _azc_sample(const void *sig, int16_t idx, uint8_t is_signed)
{
    return is_signed ? (int32_t)((const int16_t *)sig)[idx]
                     : (int32_t)((const uint16_t *)sig)[idx];
}

static inline int8_t _azc_slope_sign(const void *sig, int16_t a_idx, int16_t b_idx,
                                     uint8_t is_signed)
{
    if (a_idx == b_idx) return 0;
    int32_t a = _azc_sample(sig, a_idx, is_signed);
    int32_t b = _azc_sample(sig, b_idx, is_signed);
    if (b > a) return 1;
    if (b < a) return -1;
    return 0;
}

static uint32_t _azc_max_vdist(const void *sig, int16_t first, int16_t last,
                               uint8_t is_signed, int16_t *idx)
{
    if (first == last) {
        *idx = first;
        return 0;
    }

    int16_t dx = last - first;
    int32_t yf = _azc_sample(sig, first, is_signed);
    int32_t dy = _azc_sample(sig, last, is_signed) - yf;
    uint32_t max_dist = 0;
    *idx = first;
    for (int16_t i = 0; i <= dx; i++) {
        int32_t interp = yf + (dy * i) / dx;
        int32_t sample = _azc_sample(sig, (int16_t)(first + i), is_signed);
        uint32_t d = (uint32_t)fxp_abs_s32(sample - interp);
        if (d > max_dist) {
            max_dist = d;
            *idx = first + i;
        }
    }

    return max_dist;
}

static int16_t *_azc_polygonal_approx(const void *sig, int16_t len,
                                      uint8_t is_signed, uint32_t eps_fxp,
                                      int16_t *res_len)
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
        uint32_t max_dist = _azc_max_vdist(sig, first, last, is_signed, &mid);

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

static int16_t _azc(const void *sig, int16_t len, uint8_t is_signed, uint32_t eps_fxp)
{
    if (len <= 1) return 0;

    int16_t approx_len = 0;
    int16_t *idxs = _azc_polygonal_approx(sig, len, is_signed, eps_fxp, &approx_len);
    if (idxs == NULL || approx_len <= 0) {
        free(idxs);
        return 0;
    }

    qsort(idxs, (size_t)approx_len, sizeof(int16_t), _azc_qsort_cmp);

    int16_t azc = 0;
    if (approx_len > 2) {
        int8_t prev = _azc_slope_sign(sig, idxs[0], idxs[1], is_signed);
        for (int16_t i = 1; i < approx_len - 1; i++) {
            int8_t cur = _azc_slope_sign(sig, idxs[i], idxs[i + 1], is_signed);
            if ((prev > 0 && cur < 0) || (prev < 0 && cur > 0)) azc++;
            prev = cur;
        }
    }

    free(idxs);
    return azc;
}

static int16_t _azc_from_signal(const void *sig, int16_t len,
                                uint8_t is_signed, uint32_t epsilon_q)
{
    if (len <= 0) return 0;

    return _azc(sig, len, is_signed, epsilon_q);
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
                             fxp_feat_t *out)
{
    switch (local) {
        case LINE_LENGTH:
            *out = (fxp_feat_t)_line_length(sig->data.raw_data, sig->len,
                                            1U, FXP_FRAC_IMU_RAW);
            return;
        case KURTOSIS:
            *out = (fxp_feat_t)_kurtosis(sig->data.raw_data, sig->len);
            return;
        case ROOT_MEANS_SQUARED_IMU:
            *out = (fxp_feat_t)_rms_raw(sig->data.raw_data, sig->len);
            return;
        default:
            if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
                uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
                int16_t azc = _azc_from_signal(sig->data.raw_data, sig->len, 1U,
                                               k_azc_eps_raw_q5[idx]);
                *out = (fxp_feat_t)azc;
            }
            (void)features_selector;
            return;
    }
}

static void _run_l2a_feature(const imu_sig_view_t *sig, uint8_t local, fxp_feat_t *out)
{
    if (local == ROOT_MEANS_SQUARED_IMU) {
        *out = (fxp_feat_t)_rms_l2a(sig->data.l2a_data, sig->len);
        return;
    }

    if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
        uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
        int16_t azc = _azc_from_signal(sig->data.l2a_data, sig->len, 0U,
                                       k_azc_eps_l2a_q6[idx]);
        *out = (fxp_feat_t)azc;
    }
}

static void _run_l2g_feature(const imu_sig_view_t *sig, uint8_t local, fxp_feat_t *out)
{
    switch (local) {
        case LINE_LENGTH:
            *out = (fxp_feat_t)_line_length(sig->data.l2g_data, sig->len,
                                            0U, 11U);
            return;
        case ROOT_MEANS_SQUARED_IMU:
            *out = (fxp_feat_t)_rms_l2g(sig->data.l2g_data, sig->len);
            return;
        case CREST_FACTOR_IMU: {
            uq7_9_t rms = _rms_l2g(sig->data.l2g_data, sig->len);
            uq5_11_t peak = _max_l2g(sig->data.l2g_data, sig->len);
            uq2_14_t cf = (rms > 0U) ? _cf_l2g_result(peak, rms) : 0U;
            *out = (fxp_feat_t)cf;
            return;
        }
        default:
            if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
                uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
                int16_t azc = _azc_from_signal(sig->data.l2g_data, sig->len, 0U,
                                               k_azc_eps_l2g_q11[idx]);
                *out = (fxp_feat_t)azc;
            }
            return;
    }
}

void imu_run_features_native(const int8_t *features_selector, imu_sig_view_t sig, fxp_feat_t *feats)
{
    for (uint8_t local = 0U; local < Num_imu_feat_families; local++) {
        if (features_selector[local] != 1) continue;

        switch (sig.kind) {
            case IMU_SIG_KIND_RAW:
                _run_raw_feature(features_selector, &sig, local, &feats[local]);
                break;
            case IMU_SIG_KIND_L2A:
                _run_l2a_feature(&sig, local, &feats[local]);
                break;
            case IMU_SIG_KIND_L2G:
                _run_l2g_feature(&sig, local, &feats[local]);
                break;
            default:
                fprintf(stderr, "IMU dispatch: unsupported signal kind %d.\n", (int)sig.kind);
                abort();
        }
    }
}
#endif
