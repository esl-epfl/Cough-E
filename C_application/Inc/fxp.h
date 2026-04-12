#pragma once

#include <stdint.h>
#include <stdlib.h>

// =============================================================================
// fxp.h — Fixed-point library for Cough-E IMU feature block
//
// Q-format notation:
//   Qm.f  : signed,   m integer bits (1 sign + m-1 magnitude), f fractional bits
//   UQm.f : unsigned, m integer bits, f fractional bits
//   Total register width = m + f bits
//
// Design split:
//   - Macros:         shift primitives, float<->fixed conversions (need constant f),
//                     and generator macros (must be macros: produce named functions)
//   - static inline:  all arithmetic ops and per-kernel op helpers
//                     (type-safe, debuggable, no double-evaluation)
//
// Compile with -DFXP_MODE to activate fixed-point kernels.
// Without FXP_MODE, original float code compiles unchanged.
//
// All Q-formats from OVERLEAF.md Section 4, table: tab:imu-all-qfmt.
// =============================================================================

// =============================================================================
// Section 1 — Type aliases
// =============================================================================

typedef int16_t q11_5_t;   // RAW axis signal,             Q11.5  (16-bit)
typedef uint16_t uq10_6_t; // L2_A (accel norm),           UQ10.6 (16-bit)
typedef uint16_t uq5_11_t; // L2_G (gyro norm),            UQ5.11 (16-bit)

typedef int16_t q4_5_t;     // kurtosis mean / centred,     Q4.5   (16-bit, effective)
typedef uint32_t uq10_22_t; // variance accumulator,        UQ10.22 (32-bit)
typedef uint32_t uq4_22_t;  // variance,                    UQ4.22 (32-bit)
typedef uint16_t uq2_11_t;  // standard deviation,          UQ2.11 (16-bit)
typedef uint64_t uq8_44_t;  // std^4,                       UQ8.44 (64-bit)
typedef uint64_t uq14_44_t; // N * std^4,                   UQ14.44 (64-bit)
// Kurtosis result widened from OVERLEAF's Q5.22 to Q6.30 for model fidelity.
typedef int64_t q6_30_t;    // kurtosis result,             Q6.30  (64-bit)

// RMS result types.
// RAW is widened from OVERLEAF's UQ13.3 to UQ11.16 for model fidelity (more
// fractional bits preserve small-signal RMS values that the classifier uses).
typedef uint32_t uq11_16_t; // RMS result (RAW),            UQ11.16 (32-bit)
typedef uint16_t uq13_3_t;  // RMS result (L2_A),           UQ13.3 (16-bit)
typedef uint16_t uq7_9_t;   // RMS result (L2_G),           UQ7.9  (16-bit)

typedef uint32_t uq2_14_t; // crest factor result,         UQ2.14 (32-bit)
// Line length RAW widened from OVERLEAF's UQ2.9 to UQ3.23 for model fidelity.
typedef uint32_t uq3_23_t; // line length result (RAW),    UQ3.23 (32-bit)
typedef uint16_t uq2_9_t;  // line length result (L2_G),   UQ2.9  (16-bit)

// =============================================================================
// Section 3 — Shift primitives  (macros: applied to any integer type)
// =============================================================================

#define FXP_SR(x, n) ((x) >> (n))
#define FXP_SL(x, n) ((x) << (n))

// =============================================================================
// Section 4 — Float <-> fixed-point conversions  (macros: f must be a constant)
// =============================================================================

// Signed: handles both positive and negative floats correctly
#define FXP_FROM_FLOAT(x, f)                                 \
    ((int32_t)(((float)(x) >= 0.0f)                          \
                   ? ((float)(x) * (float)(1 << (f)) + 0.5f) \
                   : ((float)(x) * (float)(1 << (f)) - 0.5f)))

// Unsigned variant (strictly positive values only)
#define FXP_FROM_FLOAT_U(x, f) \
    ((uint32_t)((float)(x) * (float)(1 << (f)) + 0.5f))

// Fixed to float (verification / printf)
#define FXP_TO_FLOAT(x, f) ((float)(x) / (float)(1 << (f)))

// Signal-entry convenience converters
#define FXP_IMU_RAW_FROM_FLOAT(x) ((q11_5_t)FXP_FROM_FLOAT((x), 5))
#define FXP_IMU_L2A_FROM_FLOAT(x) ((uq10_6_t)FXP_FROM_FLOAT_U((x), 6))
#define FXP_IMU_L2G_FROM_FLOAT(x) ((uq5_11_t)FXP_FROM_FLOAT_U((x), 11))

// =============================================================================
// Section 5 — Arithmetic primitives  (static inline: type-safe, no side-effects)
// =============================================================================

static inline int32_t fxp_mul_s32(int32_t a, int32_t b) { return a * b; }
static inline uint32_t fxp_mul_u32(uint32_t a, uint32_t b) { return a * b; }
static inline int64_t fxp_mul_s64(int64_t a, int64_t b) { return a * b; }
static inline uint64_t fxp_mul_u64(uint64_t a, uint64_t b) { return a * b; }

// Division with precision pre-scaling: SL(num, extra) / denom
static inline int32_t fxp_div_s32(int32_t num, int32_t denom, int extra)
{
    return (num << extra) / denom;
}
static inline uint32_t fxp_div_u32(uint32_t num, uint32_t denom, int extra)
{
    return (num << extra) / denom;
}

// Integer square root — Newton-Raphson, no floating point, no __uint128_t.
//
// Internal floor-only helpers (not for external use).
// Public fxp_sqrt32 / fxp_sqrt64 round to nearest (matching sqrtf behaviour).
//
// Inputs are Q-format integer accumulators; isqrt keeps the result in the
// integer domain without a float conversion round-trip.
//
// Rounding without wider types:
//   After N-R converges, r = floor(sqrt(x)).  Let d = x - r^2.
//   Round up iff d > (r+1)^2 - x = 2r+1-d, i.e. 2d > 2r+1, i.e. d > r.
//   d < 2r+1, so it fits in 64 bits — no overflow.
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
        // seed from floor(sqrt(hi))+1, shifted left 16: overestimates sqrt(x)
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

static inline uint32_t fxp_sqrt32(uint32_t x)
{
    uint32_t r = _fxp_isqrt32(x);
    // round to nearest: d = x - r^2; round up iff d > r (see comment above)
    uint64_t d = (uint64_t)x - (uint64_t)r * r;
    if (d > r) r++;
    return r;
}

static inline uint64_t fxp_sqrt64(uint64_t x)
{
    uint64_t r = _fxp_isqrt64(x);
    // round to nearest: d = x - r^2; round up iff d > r
    // d < 2r+1, r < 2^32, so d fits safely in uint64_t
    uint64_t d = x - r * r;
    if (d > r) r++;
    return r;
}

static inline int32_t fxp_abs_s32(int32_t x) { return x < 0 ? -x : x; }
static inline int16_t fxp_abs_s16(int16_t x) { return x < 0 ? -x : x; }

// Note: use fxp_abs_s32 / fxp_abs_s16 instead of a macro to avoid
// double-evaluation. The old FXP_ABS macro has been removed.

// =============================================================================
// Section 6 — Per-kernel operation helpers  (static inline)
//
// These encode the Q-format arithmetic for each step of each IMU kernel,
// exactly as specified in OVERLEAF.md tab:imu-all-qfmt.
// =============================================================================

// ── Line Length ─────────────────────────────────────────────────────────────
// RAW (Q11.5): accumulate |diff| values (Q11.5 integer units) without shifting.
//   After loop: (uint64)(accum) << 18 / (N-1) -> UQ3.23.
//   The <<18 promotes from 5 frac bits to 23 frac bits (gain of 18).
//   NOTE: OVERLEAF specifies UQ2.9; implementation widens to UQ3.23 for model fidelity.
static inline uint32_t fxp_linelen_raw_diff_to_accum(int32_t diff)
{
    return (uint32_t)fxp_abs_s32(diff);
} // no early shift

static inline uq3_23_t fxp_linelen_raw_result(uint32_t accum, int16_t N)
{
    return (uq3_23_t)(((uint64_t)accum << 18) / (uint32_t)N);
}

// L2_G (UQ5.11): diff is in UQ5.11 integer units (11 frac bits).
//   SR(2): 11 - 2 = 9 frac bits remaining -> accumulator has 9 frac bits.
//   Result cast to UQ2.9 is then consistent (9 frac bits).
static inline uint32_t fxp_linelen_l2g_diff_to_accum(int32_t diff)
{
    return (uint32_t)fxp_abs_s32(diff) >> 2;
}

// L2_G: accum / N -> UQ2.9
static inline uq2_9_t fxp_linelen_l2g_result(uint32_t accum, int16_t N)
{
    return (uq2_9_t)(accum / (uint32_t)N);
}

// ── Kurtosis helpers (RAW only) ─────────────────────────────────────────────
#define FXP_KURT_FISHER_Q6_30 ((int64_t)3 << 30) // 3.0 in Q6.30

// Mean: compute in Q11.10 (5× finer than Q11.5) to avoid systematic bias in centred values.
//   mean_q10 = (sum_q5 * 32) / N  (int32, Q11.10)
// Centred values are then rounded back to Q11.5:
//   centred_q5 = ((sig_q5 << 5) - mean_q10 + 16) >> 5
// The +16 (= 0.5 LSB at Q11.10) gives nearest-integer rounding to Q11.5.
// This reduces mean-quantisation error in kurtosis by ~28× vs plain truncation.
static inline int32_t fxp_kurt_mean_q10(int32_t accum, int16_t N)
{
    return (accum * 32) / (int32_t)N;
}
static inline int16_t fxp_kurt_centred(q11_5_t sig, int32_t mean_q10)
{
    return (int16_t)(((int32_t)sig * 32 - mean_q10 + 16) >> 5);
}

// Centred^2: Q4.5 * Q4.5 -> Q8.10; SL(12) -> UQ10.22.
// sq can reach 291^2=84681; sq<<12 can reach 3.5e8, fits uint32.
// But sum of 50 such terms can reach 50*3.5e8=1.7e10 > uint32 -- use uint64.
static inline uint32_t fxp_kurt_sq_dev(int16_t c)
{
    return (uint32_t)fxp_mul_s32(c, c);
}
static inline uint64_t fxp_kurt_sq_to_var_accum(uint32_t sq)
{
    return (uint64_t)(sq) << 12;
}

// Variance: sum / N -> UQ4.22;  std: sqrt(UQ4.22) -> UQ2.11
static inline uq4_22_t fxp_kurt_variance(uint64_t sum, int16_t N)
{
    return (uq4_22_t)(sum / (uint64_t)N);
}
static inline uq2_11_t fxp_kurt_std(uq4_22_t variance)
{
    return (uq2_11_t)fxp_sqrt32(variance);
}

// std^4: UQ2.11 -> UQ4.22 -> UQ8.44 (64-bit)
static inline uint32_t fxp_kurt_std2(uq2_11_t std)
{
    return fxp_mul_u32(std, std);
}
static inline uint64_t fxp_kurt_std4(uint32_t std2)
{
    return fxp_mul_u64(std2, std2);
}
static inline uq14_44_t fxp_kurt_denom(uint64_t std4, int16_t N)
{
    return (uq14_44_t)((uint64_t)N * std4);
}

// Fourth moment: centred^2 = UQ8.10; centred^4 = UQ16.20
// c2 can reach ~84000 (c=291), so c4 = c2*c2 can reach ~7e9 > uint32_t.
// Cast c2 to uint64_t before squaring to avoid overflow.
// sum_c4 can exceed uint32 (50 * max_c4 ~ 3.6e11), accumulate in uint64.
static inline uint64_t fxp_kurt_c2(int16_t c)
{
    return (uint64_t)fxp_mul_s32(c, c);
}
static inline uint64_t fxp_kurt_c4(uint64_t c2)
{
    return fxp_mul_u64(c2, c2);
}

// Result in Q6.30:
//   kurtosis_float = (sum_x4 / 2^20) / (denom / 2^44) = sum_x4 * 2^24 / denom
//   result_Q6.30   = kurtosis_float * 2^30 = (sum_x4 << 24) / (denom >> 30)
//   sum_x4 << 24 fits uint64 (max ~2^61); denom >> 30 retains ~27 bits of precision.
//   Subtract 3.0 in Q6.30 for Fisher correction.
static inline q6_30_t fxp_kurt_result(uint64_t sum_x4, uq14_44_t denom)
{
    /* denom is in Q14.44. We shift right 30 to reduce to a usable divisor.
       If (denom >> 30) is zero (denom < 2^30) the division would be by
       zero, causing SIGFPE. Guard against that case and return 0 (same
       semantic as earlier std==0 guard) to indicate undefined kurtosis. */
    uint64_t _denom_shift = (uint64_t)denom >> 30;
    if (_denom_shift == 0)
        return 0;
    return (q6_30_t)((sum_x4 << 24) / _denom_shift) - FXP_KURT_FISHER_Q6_30;
}

// ── RMS ─────────────────────────────────────────────────────────────────────
// RAW: Q11.5^2 -> UQ22.10 (sq in uint32).  Accumulate in uint64 without
//   early right-shift so we preserve all bits for the wider result format.
//   mean = sum / N (Q22.10).  Shift left 22 -> Q22.32; sqrt -> UQ11.16.
static inline uint32_t fxp_rms_raw_sq(q11_5_t x)
{
    return (uint32_t)fxp_mul_s32(x, x);
}
static inline uint64_t fxp_rms_raw_sq_to_accum(uint32_t sq)
{
    return (uint64_t)sq;
} // no early shift; accumulate full Q22.10 values
// (uint64 sum) / N -> Q22.10 mean; << 22 -> Q22.32; sqrt -> UQ11.16
static inline uq11_16_t fxp_rms_raw_result(uint64_t sum, int16_t N)
{
    return (uq11_16_t)fxp_sqrt64((sum / (uint64_t)N) << 22);
}

// L2_A: UQ10.6^2 -> UQ20.12; SR(5) -> UQ25.7 accum; /N; SR(1); sqrt -> UQ13.3
static inline uint32_t fxp_rms_l2a_sq(uq10_6_t x)
{
    return fxp_mul_u32(x, x);
}
static inline uint32_t fxp_rms_l2a_sq_to_accum(uint32_t sq)
{
    return sq >> 5;
}
static inline uq13_3_t fxp_rms_l2a_result(uint32_t sum, int16_t N)
{
    return (uq13_3_t)fxp_sqrt32((sum / (uint32_t)N) >> 1);
}

// L2_G: UQ5.11^2 -> UQ10.22; SR(3) -> UQ13.19 accum; /N; SR(1); sqrt -> UQ7.9
static inline uint32_t fxp_rms_l2g_sq(uq5_11_t x)
{
    return fxp_mul_u32(x, x);
}
static inline uint32_t fxp_rms_l2g_sq_to_accum(uint32_t sq)
{
    return sq >> 3;
}
static inline uq7_9_t fxp_rms_l2g_result(uint32_t sum, int16_t N)
{
    return (uq7_9_t)fxp_sqrt32((sum / (uint32_t)N) >> 1);
}

// ── Crest Factor (L2_G only) ─────────────────────────────────────────────────
// peak UQ5.11; widen -> SL(12) -> UQ9.23; / rms UQ7.9 -> UQ2.14
static inline uq2_14_t fxp_cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

// ── ZCR (Zero Crossing Rate) ────────────────────────────────────────────────
// For signed signals (RAW/Q11.5): count sign changes between consecutive
// samples.  No float conversion needed — just check sign bits.
// For unsigned signals (L2_A, L2_G): always positive, ZCR = 0 by definition.
// NOTE: returns float — ZCR is inherently a ratio in [0,1]; the model
// consumes it as float regardless of mode.  See pipeline audit notes.
static inline float fxp_compute_zcr_raw(const q11_5_t *sig, int16_t len)
{
    int sum = 0;
    for (int16_t i = 0; i < len - 1; i++)
    {
        // Sign change: one negative, other non-negative (or vice versa)
        if ((sig[i] ^ sig[i + 1]) < 0)
            sum++;
    }
    return (float)sum / (float)(len - 1);
}

// ── AZC signal-value helpers ─────────────────────────────────────────────────
static inline uint32_t fxp_azc_dist(int32_t a, int32_t b)
{
    return (uint32_t)fxp_abs_s32(a - b);
}
static inline int32_t fxp_azc_diff(int32_t a, int32_t b, int16_t gap)
{
    return (gap == 0) ? 0 : (b - a) / (int32_t)gap;
}

// Epsilon to FxP (for each signal type)
#define FXP_AZC_EPS_RAW(eps) ((uint32_t)FXP_FROM_FLOAT((eps), 5))
#define FXP_AZC_EPS_L2A(eps) ((uint32_t)FXP_FROM_FLOAT_U((eps), 6))
#define FXP_AZC_EPS_L2G(eps) ((uint32_t)FXP_FROM_FLOAT_U((eps), 11))

// =============================================================================
// Section 7 — Generator macros  (must be macros: stamp out named functions)
//
// Instantiate in the relevant .c file under #ifdef FXP_MODE.
// All generated functions are static; names are fxp_<kernel>_<suffix>.
// =============================================================================

#ifdef FXP_MODE

// ── 7.1  get_rms ─────────────────────────────────────────────────────────────
// Parameters:
//   SUFFIX     : raw / l2a / l2g
//   sample_t   : input element type
//   result_t   : output type (uq11_16_t for RAW; uq13_3_t for L2A; uq7_9_t for L2G)
//   accum_t    : accumulator type (uint64_t for RAW; uint32_t for L2A/L2G)
//   SQ         : per-sample square inline fn
//   SQ_TO_ACC  : per-term accumulator contribution (returns accum_t)
//   RESULT     : (accum_t, len) -> result_t
#define FXP_DEFINE_GET_RMS(SUFFIX, sample_t, result_t, accum_t, SQ, SQ_TO_ACC, RESULT) \
    result_t fxp_get_rms_##SUFFIX(const sample_t *sig, int16_t len)                    \
    {                                                                                  \
        accum_t _sum = 0;                                                              \
        for (int16_t _i = 0; _i < len; _i++)                                           \
        {                                                                              \
            _sum += SQ_TO_ACC(SQ(sig[_i]));                                            \
        }                                                                              \
        return RESULT(_sum, len);                                                      \
    }

// ── 7.2  get_line_length ─────────────────────────────────────────────────────
// Parameters:
//   SUFFIX       : raw / l2g
//   result_t     : output type (uq3_23_t for RAW; uq2_9_t for L2G)
//   sample_t     : input element type
//   DIFF_TO_ACC  : per-diff accumulator step inline fn (returns uint32_t)
//   RESULT       : (uint32_t accum, int16_t N) -> result_t
#define FXP_DEFINE_GET_LINE_LENGTH(SUFFIX, result_t, sample_t, DIFF_TO_ACC, RESULT) \
    result_t fxp_get_line_length_##SUFFIX(const sample_t *sig, int16_t len)         \
    {                                                                               \
        uint32_t _accum = 0;                                                        \
        for (int16_t _i = 0; _i < len - 1; _i++)                                    \
        {                                                                           \
            int32_t _diff = (int32_t)sig[_i + 1] - (int32_t)sig[_i];                \
            _accum += DIFF_TO_ACC(_diff);                                           \
        }                                                                           \
        return RESULT(_accum, len - 1);                                             \
    }

// ── 7.3  get_kurtosis (RAW only) ─────────────────────────────────────────────
// Spelled out (not parameterised) — the Q-format chain is unique to RAW.
// sum_x4 uses uint64 (50 * max_x4 ~ 1.2e11, overflows uint32).
#define FXP_DEFINE_GET_KURTOSIS_RAW()                                            \
    q6_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len)                \
    {                                                                            \
        /* Step 1: mean in Q11.10 (rounded) to minimise centred-value bias */    \
        int32_t _sum_mean = 0;                                                   \
        for (int16_t _i = 0; _i < len; _i++)                                     \
            _sum_mean += (int32_t)sig[_i];                                       \
        int32_t _mean_q10 = fxp_kurt_mean_q10(_sum_mean, len);                   \
        /* Step 2: variance -> std (centred rounded to Q11.5 via Q11.10 mean) */ \
        uint64_t _sum_var = 0;                                                   \
        for (int16_t _i = 0; _i < len; _i++)                                     \
        {                                                                        \
            int16_t _c = fxp_kurt_centred(sig[_i], _mean_q10);                   \
            _sum_var += fxp_kurt_sq_to_var_accum(fxp_kurt_sq_dev(_c));           \
        }                                                                        \
        uq4_22_t _var = fxp_kurt_variance(_sum_var, len);                        \
        uq2_11_t _std = fxp_kurt_std(_var);                                      \
        /* Step 3: denominator = N * std^4 (64-bit) */                           \
        uint32_t _std2 = fxp_kurt_std2(_std);                                    \
        uint64_t _std4 = fxp_kurt_std4(_std2);                                   \
        uq14_44_t _denom = fxp_kurt_denom(_std4, len);                           \
        if (_denom == 0)                                                         \
            return 0; /* guard: std==0, undefined */                             \
        /* Step 4: fourth moment accumulation (uint64: sum can exceed uint32) */ \
        uint64_t _sum_x4 = 0;                                                    \
        for (int16_t _i = 0; _i < len; _i++)                                     \
        {                                                                        \
            int16_t _c = fxp_kurt_centred(sig[_i], _mean_q10);                   \
            _sum_x4 += fxp_kurt_c4(fxp_kurt_c2(_c));                             \
        }                                                                        \
        return fxp_kurt_result(_sum_x4, _denom);                                 \
    }

// ── 7.4  get_max (L2_G only) ─────────────────────────────────────────────────
#define FXP_DEFINE_GET_MAX_L2G()                               \
    uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len) \
    {                                                          \
        uq5_11_t _max = sig[0];                                \
        for (int16_t _i = 1; _i < len; _i++)                   \
        {                                                      \
            if (sig[_i] > _max)                                \
                _max = sig[_i];                                \
        }                                                      \
        return _max;                                           \
    }

// ── 7.5  L2 norm (per-sample: 3 float axes -> one FxP output sample) ─────────
// Accel: sqrt(ax^2 + ay^2 + az^2) in Q11.5 intermediate -> UQ10.6 output
//   sum of squares in Q22.10; shift sum left 2 -> Q22.12; sqrt -> UQ11.6 -> cast UQ10.6
#define FXP_DEFINE_L2_NORM_ACCEL()                                                                                           \
    uq10_6_t fxp_L2_norm_accel(float ax, float ay, float az)                                                                 \
    {                                                                                                                        \
        q11_5_t _ax = FXP_IMU_RAW_FROM_FLOAT(ax);                                                                            \
        q11_5_t _ay = FXP_IMU_RAW_FROM_FLOAT(ay);                                                                            \
        q11_5_t _az = FXP_IMU_RAW_FROM_FLOAT(az);                                                                            \
        uint32_t _sum = (uint32_t)fxp_mul_s32(_ax, _ax) + (uint32_t)fxp_mul_s32(_ay, _ay) + (uint32_t)fxp_mul_s32(_az, _az); \
        return (uq10_6_t)fxp_sqrt32(_sum << 2);                                                                              \
    }

// Gyro: sqrt(gx^2 + gy^2 + gz^2); sum in Q22.10; shift left 12 -> Q22.22; sqrt -> UQ11.11
// UQ11.11 truncated to UQ5.11 (upper bits zero for observed gyro range [3,29])
#define FXP_DEFINE_L2_NORM_GYRO()                                                                                            \
    uq5_11_t fxp_L2_norm_gyro(float gx, float gy, float gz)                                                                  \
    {                                                                                                                        \
        q11_5_t _gx = FXP_IMU_RAW_FROM_FLOAT(gx);                                                                            \
        q11_5_t _gy = FXP_IMU_RAW_FROM_FLOAT(gy);                                                                            \
        q11_5_t _gz = FXP_IMU_RAW_FROM_FLOAT(gz);                                                                            \
        uint32_t _sum = (uint32_t)fxp_mul_s32(_gx, _gx) + (uint32_t)fxp_mul_s32(_gy, _gy) + (uint32_t)fxp_mul_s32(_gz, _gz); \
        return (uq5_11_t)fxp_sqrt64((uint64_t)_sum << 12);                                                                   \
    }

// ── 7.6  AZC sub-generators ──────────────────────────────────────────────────
// Linear interpolation in fixed-point (integer arithmetic only: dy*i/dx)
#define FXP_DEFINE_AZC_INTERP(SUFFIX, sample_t)                             \
    static void fxp_interp_##SUFFIX(int16_t len, int16_t xf, sample_t yf,   \
                                    int16_t xl, sample_t yl, sample_t *res) \
    {                                                                       \
        res[0] = yf;                                                        \
        res[len - 1] = yl;                                                  \
        int32_t _dy = (int32_t)yl - (int32_t)yf;                            \
        int16_t _dx = xl - xf;                                              \
        for (int16_t _i = 1; _i < len - 1; _i++)                            \
        {                                                                   \
            res[_i] = (sample_t)((int32_t)yf + (_dy * _i) / _dx);           \
        }                                                                   \
    }

// Max vertical distance from segment (returns FxP distance in signal units)
#define FXP_DEFINE_AZC_MAX_VDIST(SUFFIX, sample_t)                                     \
    static uint32_t fxp_max_vdist_##SUFFIX(const sample_t *sig, int16_t first,         \
                                           int16_t last, int16_t *idx)                 \
    {                                                                                  \
        if (first == last)                                                             \
        {                                                                              \
            *idx = first;                                                              \
            return 0;                                                                  \
        }                                                                              \
        int16_t _len = last - first + 1;                                               \
        sample_t *_intrp = (sample_t *)malloc((size_t)_len * sizeof(sample_t));        \
        fxp_interp_##SUFFIX(_len, first, sig[first], last, sig[last], _intrp);         \
        uint32_t _max_d = 0;                                                           \
        *idx = first;                                                                  \
        for (int16_t _i = 0; _i < _len; _i++)                                          \
        {                                                                              \
            uint32_t _d = fxp_azc_dist((int32_t)sig[first + _i], (int32_t)_intrp[_i]); \
            if (_d > _max_d)                                                           \
            {                                                                          \
                _max_d = _d;                                                           \
                *idx = first + _i;                                                     \
            }                                                                          \
        }                                                                              \
        free(_intrp);                                                                  \
        return _max_d;                                                                 \
    }

// Douglas-Peucker polygonal approximation returning index array
#define FXP_DEFINE_AZC_POLYGONAL_APPROX(SUFFIX, sample_t)                           \
    static int16_t *fxp_polygonal_approx_##SUFFIX(const sample_t *sig, int16_t len, \
                                                  uint32_t eps_fxp,                 \
                                                  int16_t *res_len)                 \
    {                                                                               \
        int16_t *_res = (int16_t *)malloc((size_t)len * sizeof(int16_t));           \
        int16_t _found = 0;                                                         \
        typedef struct                                                              \
        {                                                                           \
            int16_t first;                                                          \
            int16_t last;                                                           \
        } _seg_t;                                                                   \
        _seg_t *_stk = (_seg_t *)malloc((size_t)len * sizeof(_seg_t));              \
        _stk[0].first = 0;                                                          \
        _stk[0].last = len - 1;                                                     \
        int16_t _next = 0;                                                          \
        while (_next >= 0)                                                          \
        {                                                                           \
            int16_t _f = _stk[_next].first;                                         \
            int16_t _l = _stk[_next].last;                                          \
            _next--;                                                                \
            int16_t _mid;                                                           \
            uint32_t _md = fxp_max_vdist_##SUFFIX(sig, _f, _l, &_mid);              \
            if (_md > eps_fxp)                                                      \
            {                                                                       \
                _stk[_next + 1].first = _f;                                         \
                _stk[_next + 1].last = _mid;                                        \
                _stk[_next + 2].first = _mid;                                       \
                _stk[_next + 2].last = _l;                                          \
                _next += 2;                                                         \
            }                                                                       \
            else                                                                    \
            {                                                                       \
                int16_t _af = 1, _al = 1;                                           \
                for (int16_t _j = 0; _j < _found; _j++)                             \
                {                                                                   \
                    if (_f == _res[_j])                                             \
                        _af = 0;                                                    \
                    if (_l == _res[_j])                                             \
                        _al = 0;                                                    \
                }                                                                   \
                if (_af)                                                            \
                    _res[_found++] = _f;                                            \
                if (_al)                                                            \
                    _res[_found++] = _l;                                            \
            }                                                                       \
        }                                                                           \
        free(_stk);                                                                 \
        *res_len = _found;                                                          \
        return _res;                                                                \
    }

// qsort comparator (shared across all AZC suffix variants; defined once, inline)
static inline int _fxp_qsort_cmp(const void *a, const void *b)
{
    return (int)(*(const int16_t *)a) - (int)(*(const int16_t *)b);
}

// Full AZC computation: polygonal approx -> sort -> diff -> zero-crossing count
#define FXP_DEFINE_AZC_COMPUTATION(SUFFIX, sample_t, EPS_CONVERT)                    \
    int16_t fxp_azc_computation_##SUFFIX(const sample_t *sig, int16_t len,           \
                                         float epsilon)                              \
    {                                                                                \
        uint32_t _eps = EPS_CONVERT(epsilon);                                        \
        int16_t _alen = 0;                                                           \
        int16_t *_aidxs = fxp_polygonal_approx_##SUFFIX(sig, len, _eps, &_alen);     \
        qsort(_aidxs, (size_t)_alen, sizeof(int16_t), _fxp_qsort_cmp);               \
        int16_t _azc = 0;                                                            \
        if (_alen > 2)                                                               \
        {                                                                            \
            int32_t _prev = fxp_azc_diff((int32_t)sig[_aidxs[0]],                    \
                                         (int32_t)sig[_aidxs[1]],                    \
                                         (int16_t)(_aidxs[1] - _aidxs[0]));          \
            for (int16_t _i = 1; _i < _alen - 1; _i++)                               \
            {                                                                        \
                int32_t _cur = fxp_azc_diff((int32_t)sig[_aidxs[_i]],                \
                                            (int32_t)sig[_aidxs[_i + 1]],            \
                                            (int16_t)(_aidxs[_i + 1] - _aidxs[_i])); \
                if ((_prev > 0 && _cur < 0) || (_prev < 0 && _cur > 0))              \
                    _azc++;                                                          \
                _prev = _cur;                                                        \
            }                                                                        \
        }                                                                            \
        free(_aidxs);                                                                \
        return _azc;                                                                 \
    }

// ── 7.7  Convenience: all AZC sub-functions for one signal type ───────────────
#define FXP_DEFINE_AZC_ALL(SUFFIX, sample_t, EPS_CONVERT) \
    FXP_DEFINE_AZC_INTERP(SUFFIX, sample_t)               \
    FXP_DEFINE_AZC_MAX_VDIST(SUFFIX, sample_t)            \
    FXP_DEFINE_AZC_POLYGONAL_APPROX(SUFFIX, sample_t)     \
    FXP_DEFINE_AZC_COMPUTATION(SUFFIX, sample_t, EPS_CONVERT)

#endif // FXP_MODE

// =============================================================================
// Section 8 — Forward declarations for generated functions
//
// The generator macros (Section 7) stamp out functions with external linkage.
// These declarations allow every translation unit that includes fxp.h to call
// the generated kernels without needing to see the generator instantiation.
// The linker resolves each symbol to the definition in the owning .c file:
//   get_rms_{raw,l2a,l2g}, get_max_l2g          ->  time_domain_feat.c
//   get_line_length_{raw,l2g}, get_kurtosis_raw,
//   L2_norm_{accel,gyro}                         ->  helpers.c
//   azc_computation_{raw,l2a,l2g}               ->  azc.c
// =============================================================================
#ifdef FXP_MODE

// ── RMS ──────────────────────────────────────────────────────────────────────
uq11_16_t fxp_get_rms_raw(const q11_5_t *sig, int16_t len);
uq13_3_t fxp_get_rms_l2a(const uq10_6_t *sig, int16_t len);
uq7_9_t fxp_get_rms_l2g(const uq5_11_t *sig, int16_t len);

// ── Line length ───────────────────────────────────────────────────────────────
uq3_23_t fxp_get_line_length_raw(const q11_5_t *sig, int16_t len);
uq2_9_t fxp_get_line_length_l2g(const uq5_11_t *sig, int16_t len);

// ── Kurtosis ──────────────────────────────────────────────────────────────────
q6_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len);

// ── get_max ───────────────────────────────────────────────────────────────────
uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len);

// ── L2 norm ───────────────────────────────────────────────────────────────────
uq10_6_t fxp_L2_norm_accel(float ax, float ay, float az);
uq5_11_t fxp_L2_norm_gyro(float gx, float gy, float gz);

// ── AZC ───────────────────────────────────────────────────────────────────────
int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, float epsilon);

#endif // FXP_MODE
