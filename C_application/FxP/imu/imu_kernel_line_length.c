#include <imu/imu_kernels.h>

#ifdef FXP_MODE

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

#endif
