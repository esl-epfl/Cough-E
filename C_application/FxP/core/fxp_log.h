#pragma once

#include <stdint.h>

#include <audio/audio_periodogram_lut.h>

#define FXP_LN2_Q24 ((int32_t)11629080)

/* Natural logarithm on unsigned integer input, result in Q11. */
static inline int32_t fxp_ln_u64_q11(uint64_t x)
{
    if (x == 0ULL) x = 1ULL;

    uint32_t msb = 63U - (uint32_t)__builtin_clzll(x);
    uint64_t base = 1ULL << msb;
    uint64_t diff = x - base;

    uint32_t frac_q24;
    if (msb <= 24U) {
        frac_q24 = (uint32_t)(diff << (24U - msb));
    } else {
        uint32_t shift = msb - 24U;
        frac_q24 = (uint32_t)((diff + (1ULL << (shift - 1U))) >> shift);
    }

    uint32_t idx = frac_q24 >> 16;
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = frac_q24 & 0xFFFFU;

    int32_t y0 = fxp_ln_lut_q24[idx];
    int32_t y1 = fxp_ln_lut_q24[idx + 1];
    int32_t y = y0 + (int32_t)((((int64_t)(y1 - y0) * (int64_t)alpha) + (1LL << 15)) >> 16);

    int64_t ln_x_q24 = (int64_t)msb * (int64_t)FXP_LN2_Q24 + (int64_t)y;
    return (int32_t)((ln_x_q24 + (1LL << 12)) >> 13);
}
