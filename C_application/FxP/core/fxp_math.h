#pragma once

#include <limits.h>
#include <stdint.h>

#include <core/fxp_sat.h>

#define FXP_SR(x, n) ((x) >> (n))
#define FXP_SL(x, n) ((x) << (n))

static inline int32_t  fxp_mul_s32(int32_t a, int32_t b)    { return a * b; }
static inline uint32_t fxp_mul_u32(uint32_t a, uint32_t b)  { return a * b; }
static inline int64_t  fxp_mul_s64(int64_t a, int64_t b)    { return a * b; }
static inline uint64_t fxp_mul_u64(uint64_t a, uint64_t b)  { return a * b; }

static inline int32_t fxp_div_s32(int32_t num, int32_t denom, int extra)
{
    if (denom == 0) return (num >= 0) ? INT32_MAX : INT32_MIN;
    int64_t scaled = ((int64_t)num) << extra;
    return fxp_sat_s32_from_s64(scaled / denom);
}

static inline uint32_t fxp_div_u32(uint32_t num, uint32_t denom, int extra)
{
    if (denom == 0) return UINT32_MAX;
    uint64_t scaled = ((uint64_t)num) << extra;
    return fxp_sat_u32_from_u64(scaled / denom);
}

static inline uint32_t _fxp_isqrt32(uint32_t x)
{
    if (x == 0) return 0;
    uint32_t bits = 32U - (uint32_t)__builtin_clz(x);
    uint32_t r = (uint32_t)1U << ((bits + 1U) >> 1U);
    for (;;) {
        uint32_t q = x / r;
        if (r <= q) break;
        r = (r + q) >> 1U;
    }
    return r;
}

static inline uint64_t _fxp_isqrt64(uint64_t x)
{
    if (x == 0) return 0;
    uint32_t hi = (uint32_t)(x >> 32);
    uint64_t r;
    if (hi != 0) {
        r = ((uint64_t)_fxp_isqrt32(hi) + 1ULL) << 16;
    } else {
        r = (uint64_t)_fxp_isqrt32((uint32_t)x) + 1ULL;
    }

    for (;;) {
        uint64_t q = x / r;
        if (r <= q) break;
        r = (r + q) >> 1;
    }
    return r;
}

static inline uint32_t fxp_sqrt32(uint32_t x)
{
    uint32_t r = _fxp_isqrt32(x);
    uint64_t d = (uint64_t)x - (uint64_t)r * r;
    if (d > r) r++;
    return r;
}

static inline uint64_t fxp_sqrt64(uint64_t x)
{
    uint64_t r = _fxp_isqrt64(x);
    uint64_t d = x - r * r;
    if (d > r) r++;
    return r;
}

static inline int32_t fxp_abs_s32(int32_t x)
{
    if (x == INT32_MIN) return INT32_MAX;
    return x < 0 ? -x : x;
}

static inline int16_t fxp_abs_s16(int16_t x)
{
    if (x == INT16_MIN) return INT16_MAX;
    return x < 0 ? -x : x;
}
