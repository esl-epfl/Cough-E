#include <imu/imu_kernels.h>

#ifdef FXP_MODE

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

#endif
