#pragma once

#include <stdint.h>

// =============================================================================
// fxp_math.h — Generic fixed-point arithmetic primitives
// =============================================================================

// =============================================================================
// Section 1 — Shift primitives
// =============================================================================

#define FXP_SR(x, n) ((x) >> (n))
#define FXP_SL(x, n) ((x) << (n))

// =============================================================================
// Section 2 — Multiply and divide
// =============================================================================

static inline int32_t  fxp_mul_s32(int32_t a,   int32_t b)  { return a * b; }
static inline uint32_t fxp_mul_u32(uint32_t a,  uint32_t b) { return a * b; }
static inline int64_t  fxp_mul_s64(int64_t a,   int64_t b)  { return a * b; }
static inline uint64_t fxp_mul_u64(uint64_t a,  uint64_t b) { return a * b; }

// Division with precision pre-scaling: numerator is shifted left by `extra`
// fractional bits before dividing, so the result lands in the target Q-format
// without a separate post-shift.
static inline int32_t fxp_div_s32(int32_t num, int32_t denom, int extra)
{
    return (num << extra) / denom;
}
static inline uint32_t fxp_div_u32(uint32_t num, uint32_t denom, int extra)
{
    return (num << extra) / denom;
}

// =============================================================================
// Section 3 — Integer square root  (Newton-Raphson, no floating point)
//
// Rounding (public wrappers fxp_sqrt32 / fxp_sqrt64):
//   After N-R converges to r = floor(sqrt(x)), let d = x - r^2.
//   Round up iff d > r, because:
//     d > r  <=>  x - r^2 > (r+1)^2 - x  <=>  r+1 is closer to sqrt(x)
//   d < 2r+1, so it always fits in the same bit width.
// =============================================================================

// Internal floor-only helpers — not for direct use outside this file.
static inline uint32_t _fxp_isqrt32(uint32_t x)
{
    if (x == 0) return 0;
    uint32_t bits = 32 - __builtin_clz(x);
    uint32_t r = (uint32_t)1 << ((bits + 1) >> 1);  // 2^ceil(bits/2) >= sqrt(x)
    for (;;) {
        uint32_t q = x / r;
        if (r <= q) break;
        r = (r + q) >> 1;
    }
    return r;  // floor(sqrt(x))
}

static inline uint64_t _fxp_isqrt64(uint64_t x)
{
    if (x == 0) return 0;
    uint32_t hi = (uint32_t)(x >> 32);
    uint64_t r;
    if (hi != 0) {
        // Seed from floor(sqrt(hi))+1, shifted left 16: guaranteed overestimate
        r = ((uint64_t)_fxp_isqrt32(hi) + 1) << 16;
    } else {
        r = (uint64_t)_fxp_isqrt32((uint32_t)x) + 1;
    }
    for (;;) {
        uint64_t q = x / r;
        if (r <= q) break;
        r = (r + q) >> 1;
    }
    return r;  // floor(sqrt(x))
}

// Public round-to-nearest wrappers
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
    // d < 2r+1 and r < 2^32, so d fits safely in uint64_t
    uint64_t d = x - r * r;
    if (d > r) r++;
    return r;
}

// =============================================================================
// Section 4 — Absolute value
// =============================================================================

static inline int32_t fxp_abs_s32(int32_t x) { return x < 0 ? -x : x; }
static inline int16_t fxp_abs_s16(int16_t x) { return x < 0 ? -x : x; }
