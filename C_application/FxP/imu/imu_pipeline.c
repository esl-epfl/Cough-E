#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <imu/imu_pipeline.h>

#ifdef FXP_MODE

/* -------------------------------------------------------------------------- */
/*  FxP kernels : IMU                                                             */
/* -------------------------------------------------------------------------- */

#define FXP_KURT_FISHER_Q10_22 ((int32_t)3 << FXP_FRAC_IMU_KURTOSIS_RAW)

// Crest factor for the gyro L2 norm signal: peak / RMS.
// peak is UQ5.11, RMS is UQ7.9, and the result is UQ2.14.
static inline uq2_14_t _crest_factor_l2g(uq5_11_t peak, uq7_9_t rms) {
    if (rms == 0) return 0;

    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

/* RMS helpers keep the sqrt input at an even number of fractional bits,
 * so integer sqrt lands directly on the desired output Q format.
 */

// RMS for the RAW signal, input is Q11.5, output is UQ13.3.
static uq13_3_t _rms_raw(const q11_5_t *sig, int16_t len) {
    if (len <= 0) return 0;

    uq25_7_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uq22_10_t sig_squared = (uq22_10_t)((int32_t)sig[i] * (int32_t)sig[i]);
        sum += (uq25_7_t)(sig_squared >> 3U);
    }

    uq26_6_t sqrt_input = (uq26_6_t)((sum / (uint16_t)len) >> 1U);
    return (uq13_3_t)fxp_sqrt32(sqrt_input);
}

// RMS for the L2A signal, input is UQ10.6, output is UQ13.3.
static uq13_3_t _rms_l2a(const uq10_6_t *sig, int16_t len) {
    if (len <= 0) return 0;

    uq25_7_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uq20_12_t sig_squared = (uq20_12_t)((uint32_t)sig[i] * (uint32_t)sig[i]);
        sum += (uq25_7_t)(sig_squared >> 5U);
    }

    uq26_6_t sqrt_input = (uq26_6_t)((sum / (uint16_t)len) >> 1U);
    return (uq13_3_t)fxp_sqrt32(sqrt_input);
}
// RMS for the L2G signal, input is UQ5.11, output is UQ7.9.
static uq7_9_t _rms_l2g(const uq5_11_t *sig, int16_t len) {
    if (len <= 0) return 0;

    uq13_19_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uq10_22_t sig_squared = (uq10_22_t)((uint32_t)sig[i] * (uint32_t)sig[i]);
        sum += (uq13_19_t)(sig_squared >> 3U);
    }

    uq14_18_t sqrt_input = (uq14_18_t)((sum / (uint16_t)len) >> 1U);
    return (uq7_9_t)fxp_sqrt32(sqrt_input);
}

// Line length for the RAW signal, input is Q11.5, output is UQ7.9.
static uq7_9_t _line_length_raw(const q11_5_t *sig, int16_t len) {
    if (len <= 1) return 0;

    uq7_9_t sum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        uq11_5_t diff = (uq11_5_t)_abs_delta_raw(sig[i + 1], sig[i]);
        sum += (uq7_9_t)(diff << 4U);
    }

    return (sum / (uint16_t)(len - 1));
}

// Line length for the L2G signal, input is UQ5.11, output is UQ7.9.
static uq7_9_t _line_length_l2g(const uq5_11_t *sig, int16_t len) {
    if (len <= 1) return 0;

    uq7_9_t sum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        // Absolute sample-to-sample difference, converted from UQ5.11 to UQ7.9.
        uq6_10_t diff = (uq6_10_t)_abs_delta_l2g(sig[i + 1], sig[i]) >> 1U;
        sum += (uq7_9_t)(diff >> 1U);
    }

    return (sum / (uint16_t)(len - 1));
}
static inline q11_5_t _kurtosis_mean(q11_5_t sum, int16_t len) {
    if (len <= 0) return 0;
    return (q11_5_t)(sum / len);
}

static inline q11_5_t _kurtosis_dev(q11_5_t sig, q11_5_t mean) { return (q11_5_t)(sig - mean); }

/* Kurtosis is computed as sum(dev^4) / (len * stddev^4) - 3.
 * The numerator and denominator use different Q formats, so the final
 * division applies a net Q22 scaling before subtracting Fisher's 3.
 */
static q10_22_t _kurtosis(const q11_5_t *sig, int16_t len) {
    if (len <= 0) return 0;

    q11_5_t sum_mean = 0;
    for (int16_t i = 0; i < len; i++) {
        sum_mean += sig[i];
    }
    q11_5_t mean = _kurtosis_mean(sum_mean, len);

    uq10_22_t sum_var = 0;
    uq44_20_t sum_dev_4 = 0;
    for (int16_t i = 0; i < len; i++) {
        q11_5_t dev = _kurtosis_dev(sig[i], mean);

        // this one is used for the numerator of kurtosis
        uq22_10_t dev_2_num = (uq22_10_t)(((int32_t)dev * (int32_t)dev));
        uq44_20_t dev_4_num = (uq44_20_t)(((uint64_t)dev_2_num * (uint64_t)dev_2_num));
        sum_dev_4 += dev_4_num;

        // this one is for the denominator of kurtosis
        uq10_22_t dev_2_denom = (uq10_22_t)(dev_2_num << 12U);
        sum_var += dev_2_denom;
    }
    // all denominator related
    uq10_22_t variance = (uq10_22_t)(sum_var / (uint16_t)len);
    uq5_11_t stddev = (uq5_11_t)fxp_sqrt32(variance);
    uq10_22_t stddev_2 = (uq10_22_t)((uint32_t)stddev * (uint32_t)stddev);
    uq20_44_t stddev_4 = (uq20_44_t)((uint64_t)stddev_2 * (uint64_t)stddev_2);
    uq20_44_t denom = (uq20_44_t)((uint64_t)len * stddev_4);

    // sum_dev_4 is UQ44.20 and denom is UQ20.44.
    // To return Q10.22, the ratio needs a net left shift of 46 bits:
    // 44 denominator frac bits - 20 numerator frac bits + 22 output frac bits.
    // We split that as numerator << 16 and denominator >> 30 to stay in 64-bit.
    uint64_t denom_shifted = (uint64_t)denom >> 30U;
    if (denom_shifted == 0) return 0;
    uint64_t normalized = (sum_dev_4 << 16U) / denom_shifted;

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
// Norm calculator for L2A: raw Q11.5 accel axes -> UQ10.6.
uq10_6_t imu_l2a(q11_5_t ax, q11_5_t ay, q11_5_t az) {
    uq22_10_t sum = (uq22_10_t)((int32_t)ax * (int32_t)ax) +
                    (uq22_10_t)((int32_t)ay * (int32_t)ay) +
                    (uq22_10_t)((int32_t)az * (int32_t)az);

    return (uq10_6_t)fxp_sqrt32(sum << 2U);
}

// Norm calculator for L2G: raw Q11.5 gyro axes -> UQ5.11.
uq5_11_t imu_l2g(q11_5_t gx, q11_5_t gy, q11_5_t gz) {
    uq22_10_t sum = (uq22_10_t)((int32_t)gx * (int32_t)gx) +
                    (uq22_10_t)((int32_t)gy * (int32_t)gy) +
                    (uq22_10_t)((int32_t)gz * (int32_t)gz);

    return (uq5_11_t)fxp_sqrt32(sum << 12U);
}


typedef struct {
    int16_t first;
    int16_t last;
} azc_segment_t;

/* AZC uses fixed-point sample values only for distance comparisons.
 * The returned feature is an integer count, so its fractional width is 0.
 */
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

static int16_t *_azc_polygonal_approx(const void *sig, int16_t len, uint8_t is_signed,
                                      uint32_t eps_fxp, int16_t *res_len) {
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

/* Per-signal feature dispatch. */
static void _run_raw_feature(const q11_5_t *sig, int16_t len, uint8_t local, fxp_feat_t *out)
{
    switch (local) {
    case LINE_LENGTH:
        *out = (fxp_feat_t)_line_length_raw(sig, len);
        return;
    case KURTOSIS:
        *out = (fxp_feat_t)_kurtosis(sig, len);
        return;
    case ROOT_MEANS_SQUARED_IMU:
        *out = (fxp_feat_t)_rms_raw(sig, len);
        return;
    default:
        if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
            uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
            int16_t azc = _azc_from_signal(sig, len, 1U, k_azc_eps_raw_q5[idx]);
            *out = (fxp_feat_t)azc;
            return;
        }
        fprintf(stderr, "FXP IMU runtime: unsupported raw feature %u.\n", (unsigned)local);
        abort();
    }
}

static void _run_l2a_feature(const uq10_6_t *sig, int16_t len, uint8_t local, fxp_feat_t *out)
{
    if (local == ROOT_MEANS_SQUARED_IMU) {
        *out = (fxp_feat_t)_rms_l2a(sig, len);
        return;
    }

    if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
        uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
        int16_t azc = _azc_from_signal(sig, len, 0U, k_azc_eps_l2a_q6[idx]);
        *out = (fxp_feat_t)azc;
        return;
    }

    fprintf(stderr, "FXP IMU runtime: unsupported accel-combo feature %u.\n", (unsigned)local);
    abort();
}

static void _run_l2g_feature(const uq5_11_t *sig, int16_t len, uint8_t local, fxp_feat_t *out)
{
    switch (local) {
    case LINE_LENGTH:
        *out = (fxp_feat_t)_line_length_l2g(sig, len);
        return;
    case ROOT_MEANS_SQUARED_IMU:
        *out = (fxp_feat_t)_rms_l2g(sig, len);
        return;
    case CREST_FACTOR_IMU: {
        uq7_9_t rms = _rms_l2g(sig, len);
        uq5_11_t peak = _max_l2g(sig, len);
        uq2_14_t crest_factor = (rms > 0U) ? _crest_factor_l2g(peak, rms) : 0U;
        *out = (fxp_feat_t)crest_factor;
        return;
    }
    default:
        if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
            uint8_t idx = (uint8_t)(local - APPROXIMATE_ZERO_CROSSING);
            int16_t azc = _azc_from_signal(sig, len, 0U, k_azc_eps_l2g_q11[idx]);
            *out = (fxp_feat_t)azc;
            return;
        }
        fprintf(stderr, "FXP IMU runtime: unsupported gyro-combo feature %u.\n", (unsigned)local);
        abort();
    }
}

void imu_run_raw_features(const int8_t *features_selector,
                          const q11_5_t *sig,
                          int16_t len,
                          fxp_feat_t *feats)
{
    for (uint8_t local = 0U; local < Num_imu_feat_families; local++) {
        if (features_selector[local] != 1) continue;
        _run_raw_feature(sig, len, local, &feats[local]);
    }
}

void imu_run_l2a_features(const int8_t *features_selector,
                          const uq10_6_t *sig,
                          int16_t len,
                          fxp_feat_t *feats)
{
    for (uint8_t local = 0U; local < Num_imu_feat_families; local++) {
        if (features_selector[local] != 1) continue;
        _run_l2a_feature(sig, len, local, &feats[local]);
    }
}

void imu_run_l2g_features(const int8_t *features_selector,
                          const uq5_11_t *sig,
                          int16_t len,
                          fxp_feat_t *feats)
{
    for (uint8_t local = 0U; local < Num_imu_feat_families; local++) {
        if (features_selector[local] != 1) continue;
        _run_l2g_feature(sig, len, local, &feats[local]);
    }
}
#endif
