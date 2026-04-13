#include <imu/imu_kernels.h>

#ifdef FXP_MODE

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

    // sum_x4 is UQ16.20 and denom is UQ20.44; shift by 24 to land in Q34.30.
    uint64_t denom_shifted = (uint64_t)denom >> FXP_FRAC_IMU_KURTOSIS_RAW;
    if (denom_shifted == 0) return 0;

    q34_30_t normalized = (q34_30_t)((sum_x4 << 24) / denom_shifted);
    return normalized - FXP_KURT_FISHER_Q34_30;
}

#endif
