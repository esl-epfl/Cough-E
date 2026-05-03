#pragma once

#include <limits.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- */
/*  Q-format fractional widths                                                */
/* -------------------------------------------------------------------------- */

#define FXP_FRAC_IMU_RAW 5
#define FXP_FRAC_AUDIO_INPUT 14

#define FXP_FRAC_IMU_RMS_RAW 3
#define FXP_FRAC_IMU_RMS_L2A 3
#define FXP_FRAC_IMU_RMS_L2G 9
#define FXP_FRAC_IMU_LINE_LENGTH_RAW 9
#define FXP_FRAC_IMU_LINE_LENGTH_L2G 9
#define FXP_FRAC_IMU_KURTOSIS_RAW 22
#define FXP_FRAC_IMU_CREST_L2G 14

#define FXP_FRAC_AUDIO_FFT_RE_IM 20
#define FXP_FRAC_AUDIO_FFT_FREQUENCIES 20
#define FXP_FRAC_AUDIO_FFT_CENTROID 21
#define FXP_FRAC_AUDIO_FFT_SPREAD 5
#define FXP_FRAC_AUDIO_FFT_KURTOSIS 15

#define FXP_FRAC_AUDIO_PSD_PROXY 11
#define FXP_FRAC_AUDIO_PSD_INTEGRAL 8
#define FXP_FRAC_AUDIO_PSD_FLATNESS 16
#define FXP_FRAC_AUDIO_PSD_BANDPOWER 16

/* -------------------------------------------------------------------------- */
/*  Q-format type aliases                                                     */
/* -------------------------------------------------------------------------- */

/* 16-bit aliases */
typedef int16_t  q11_5_t;
typedef int16_t  q5_11_t;
typedef uint16_t uq7_9_t;
typedef uint16_t uq5_11_t;
typedef uint16_t uq2_14_t;
typedef uint16_t uq10_6_t;
typedef uint16_t uq11_5_t;
typedef uint16_t uq0_16_t;

/* 32-bit aliases */
typedef int32_t  q12_20_t;
typedef int32_t  q13_19_t;
typedef int32_t  q10_22_t;
typedef uint32_t uq10_22_t;
typedef uint32_t uq12_20_t;
typedef uint32_t uq15_17_t;
typedef uint32_t uq11_21_t;
typedef uint32_t uq17_15_t;
typedef uint32_t uq21_11_t;

/* 64-bit aliases */
typedef uint64_t uq20_44_t;
/* -------------------------------------------------------------------------- */
/*  Saturating helpers                                                        */
/* -------------------------------------------------------------------------- */

static inline int16_t fxp_sat_s16_from_s32(int32_t x)
{
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

static inline uint16_t fxp_sat_u16_from_u32(uint32_t x)
{
    if (x > UINT16_MAX) return UINT16_MAX;
    return (uint16_t)x;
}

static inline int32_t fxp_sat_s32_from_s64(int64_t x)
{
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static inline uint32_t fxp_sat_u32_from_u64(uint64_t x)
{
    if (x > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)x;
}

/* -------------------------------------------------------------------------- */
/*  Math helpers                                                              */
/* -------------------------------------------------------------------------- */

static inline int32_t  fxp_mul_s32(int32_t a, int32_t b)    { return a * b; }
static inline uint32_t fxp_mul_u32(uint32_t a, uint32_t b)  { return a * b; }
static inline uint64_t fxp_mul_u64(uint64_t a, uint64_t b)  { return a * b; }

static inline int32_t fxp_div_s32(int32_t num, int32_t denom, int extra)
{
    if (denom == 0) return (num >= 0) ? INT32_MAX : INT32_MIN;
    int64_t scaled = ((int64_t)num) << extra;
    return fxp_sat_s32_from_s64(scaled / denom);
}

static inline int32_t fxp_round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static inline int32_t fxp_round_div_s32(int32_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (num + (den / 2)) / den;
    return -(((-num) + (den / 2)) / den);
}

static inline int64_t fxp_round_div_i64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (num + (den / 2)) / den;
    return -(((-num) + (den / 2)) / den);
}

static inline uint32_t fxp_round_div_u32(uint32_t num, uint32_t den)
{
    if (den == 0U) return 0U;
    return (num + (den >> 1)) / den;
}

static inline int32_t fxp_floor_div_s64(int64_t num, int32_t den)
{
    int64_t q = num / (int64_t)den;
    int64_t r = num - q * (int64_t)den;
    if (r != 0 && num < 0) q -= 1;
    return (int32_t)q;
}

static inline uint64_t fxp_round_shift_u64(uint64_t v, uint32_t shift)
{
    if (shift == 0U) return v;
    if (shift >= 64U) return 0ULL;
    return (v + (1ULL << (shift - 1U))) >> shift;
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

/* -------------------------------------------------------------------------- */
/*  Pipeline carrier helpers                                                  */
/* -------------------------------------------------------------------------- */

/* Unified fixed-point carrier for cross-module feature exchange and I/O. */
typedef int32_t fxp_q16_t;

#define FXP_PIPE_FRAC 16
#define FXP_Q16_ONE ((fxp_q16_t)(1 << FXP_PIPE_FRAC))

static inline fxp_q16_t fxp_q16_from_int(int32_t x)
{
    int64_t v = ((int64_t)x) << FXP_PIPE_FRAC;
    return fxp_sat_s32_from_s64(v);
}

static inline fxp_q16_t fxp_q16_from_u32(uint32_t x, uint8_t src_frac)
{
    int32_t shift = (int32_t)FXP_PIPE_FRAC - (int32_t)src_frac;
    int64_t v = (int64_t)x;
    if (shift > 0) {
        v <<= shift;
    } else if (shift < 0) {
        int32_t r = -shift;
        if (r >= 63) {
            v = 0;
        } else {
            v = (v + ((int64_t)1 << (r - 1))) >> r;
        }
    }
    return fxp_sat_s32_from_s64(v);
}

static inline fxp_q16_t fxp_q16_from_s32(int32_t x, uint8_t src_frac)
{
    int32_t shift = (int32_t)FXP_PIPE_FRAC - (int32_t)src_frac;
    int64_t v = (int64_t)x;
    if (shift > 0) {
        v <<= shift;
    } else if (shift < 0) {
        int32_t r = -shift;
        if (r >= 63) {
            v = 0;
        } else if (v >= 0) {
            v = (v + ((int64_t)1 << (r - 1))) >> r;
        } else {
            v = -(((-v) + ((int64_t)1 << (r - 1))) >> r);
        }
    }
    return fxp_sat_s32_from_s64(v);
}

/* -------------------------------------------------------------------------- */
/*  Float/fixed conversion helpers                                            */
/* -------------------------------------------------------------------------- */

static inline int32_t fxp_from_float_signed(float x, uint8_t frac_bits)
{
    float scale = (float)(1ULL << frac_bits);
    float scaled = x * scale;
    scaled += (scaled >= 0.0f) ? 0.5f : -0.5f;

    if (scaled > (float)INT32_MAX) return INT32_MAX;
    if (scaled < (float)INT32_MIN) return INT32_MIN;
    return (int32_t)scaled;
}

static inline float fxp_to_float(int64_t x, uint8_t frac_bits)
{
    float scale = (float)(1ULL << frac_bits);
    return (float)x / scale;
}

/* Compatibility macros used across the existing codebase. */
#define FXP_FROM_FLOAT(x, f)   (fxp_from_float_signed((float)(x), (uint8_t)(f)))
#define FXP_TO_FLOAT(x, f)     (fxp_to_float((int64_t)(x), (uint8_t)(f)))

static inline q11_5_t fxp_imu_raw_from_float(float x)
{
    return fxp_sat_s16_from_s32(FXP_FROM_FLOAT(x, FXP_FRAC_IMU_RAW));
}

static inline int16_t fxp_audio_from_float(float x)
{
    return fxp_sat_s16_from_s32(FXP_FROM_FLOAT(x, FXP_FRAC_AUDIO_INPUT));
}

#define FXP_IMU_RAW_FROM_FLOAT(x) (fxp_imu_raw_from_float((float)(x)))
#define FXP_AUDIO_FROM_FLOAT(x)   (fxp_audio_from_float((float)(x)))

/* -------------------------------------------------------------------------- */
/*  Backend scalar carriers                                                   */
/* -------------------------------------------------------------------------- */

#ifdef FXP_MODE

typedef int16_t cough_audio_sample_t; /* Q14 */
typedef q11_5_t cough_imu_sample_t;   /* Q11.5 raw IMU */
typedef fxp_q16_t cough_feat_t;       /* Q16 cross-module feature carrier */

static inline cough_audio_sample_t cough_source_audio_sample(float x)
{
    return FXP_AUDIO_FROM_FLOAT(x);
}

static inline cough_imu_sample_t cough_source_imu_sample(float x)
{
    return FXP_IMU_RAW_FROM_FLOAT(x);
}

static inline cough_feat_t cough_source_feat(float x)
{
    return FXP_FROM_FLOAT(x, FXP_PIPE_FRAC);
}

#else

typedef float cough_audio_sample_t;
typedef float cough_imu_sample_t;
typedef float cough_feat_t;

static inline cough_audio_sample_t cough_source_audio_sample(float x)
{
    return x;
}

static inline cough_imu_sample_t cough_source_imu_sample(float x)
{
    return x;
}

static inline cough_feat_t cough_source_feat(float x)
{
    return x;
}

#endif
