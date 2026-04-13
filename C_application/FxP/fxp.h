#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <fxp_types.h>
#include <fxp_math.h>

// =============================================================================
// fxp.h — Fixed-point feature kernels
//
// This file contains everything specific to the IMU feature pipeline:
//   - Per-kernel arithmetic helpers (Section 1)
//   - Generator macros that stamp out named kernel functions (Section 2)
//   - Forward declarations for all generated functions (Section 3)
//
// Types and generic math live in fxp_types.h and fxp_math.h respectively.
//
// Build flags:
//   -DFXP_MODE   activates fixed-point kernels (Sections 2 and 3)
//   Without this flag the original float code compiles unchanged.
// =============================================================================

// =============================================================================
// Section 1 — Per-kernel arithmetic helpers  (static inline)
//
// Each group encodes the Q-format arithmetic for one kernel, as specified in
// OVERLEAF.md.  Keeping arithmetic in named helpers makes the Q-format chain
// auditable without reading macro bodies.
// =============================================================================

// ── Kurtosis ─────────────────────────────────────────────────────────────────
//
// 3.0 in Q34.30, used for the Fisher (excess) correction: kurtosis - 3.
#define FXP_KURT_FISHER_Q34_30 ((int64_t)3 << 30)

// Mean computed in Q11.10 (5× finer than Q11.5) to minimise systematic bias
// from truncation.  mean_q10 = (sum_q5 * 32) / N  [int32, Q11.10]
// Centred values rounded to Q11.5:
//   centred_q5 = ((sig_q5 << 5) - mean_q10 + 16) >> 5
// +16 (= 0.5 LSB at Q11.10) gives nearest-integer rounding.
static inline int32_t fxp_kurt_mean_q10(int32_t accum, int16_t N)
{
    return (accum * 32) / (int32_t)N;
}
static inline int16_t fxp_kurt_centred(q11_5_t sig, int32_t mean_q10)
{
    return (int16_t)(((int32_t)sig * 32 - mean_q10 + 16) >> 5);
}

// Centred^2: Q4.5 * Q4.5 -> Q8.10; SL(12) -> UQ10.22.
static inline uint32_t fxp_kurt_sq_dev(int16_t c)
{
    return (uint32_t)fxp_mul_s32(c, c);
}
static inline uint64_t fxp_kurt_sq_to_var_accum(uint32_t sq)
{
    return (uint64_t)(sq) << 12;
}

// Variance: sum / N -> UQ4.22 (fits in uq10_22_t);  std: sqrt(UQ4.22) -> UQ2.11 (fits in uq5_11_t)
static inline uq10_22_t fxp_kurt_variance(uint64_t sum, int16_t N)
{
    return (uq10_22_t)(sum / (uint64_t)N);
}
static inline uq5_11_t fxp_kurt_std(uq10_22_t variance)
{
    return (uq5_11_t)fxp_sqrt32(variance);
}

// std^4: UQ2.11 -> (sq) UQ4.22 -> (sq again) UQ8.44 (fits in uq20_44_t)
static inline uint32_t fxp_kurt_std2(uq5_11_t std)
{
    return fxp_mul_u32(std, std);
}
static inline uq20_44_t fxp_kurt_std4(uint32_t std2)
{
    return fxp_mul_u64(std2, std2);
}
static inline uq20_44_t fxp_kurt_denom(uq20_44_t std4, int16_t N)
{
    return (uq20_44_t)((uint64_t)N * std4);
}

// Fourth moment: centred^2 = UQ8.10; centred^4 = UQ16.20.
static inline uint64_t fxp_kurt_c2(int16_t c)
{
    return (uint64_t)fxp_mul_s32(c, c);
}
static inline uint64_t fxp_kurt_c4(uint64_t c2)
{
    return fxp_mul_u64(c2, c2);
}

// Result in Q34.30:
//   result = (sum_x4 << 24) / (denom >> 30) - 3
//   Guard: if denom >> 30 == 0, std was 0 — return 0.
static inline q34_30_t fxp_kurt_result(uint64_t sum_x4, uq20_44_t denom)
{
    uint64_t denom_shifted = (uint64_t)denom >> 30;
    if (denom_shifted == 0)
        return 0;
    return (q34_30_t)((sum_x4 << 24) / denom_shifted) - FXP_KURT_FISHER_Q34_30;
}

// ── Line length ───────────────────────────────────────────────────────────────
//
// Both signal types use the same two helpers, parameterised by shift amounts:
//   SR : right-shift applied to |diff| before accumulation
//   SL : left-shift applied to the accumulator before dividing by N
//
// RAW  (Q11.5):  SR=0, SL=18  -> result in UQ9.23  (uq9_23_t)
// L2_G (UQ5.11): SR=2, SL=0   -> result in UQ7.9   (uq7_9_t)
static inline uint32_t fxp_linelen_diff_to_accum(int32_t diff, uint8_t sr)
{
    return (uint32_t)fxp_abs_s32(diff) >> sr;
}
static inline uint32_t fxp_linelen_result(uint32_t accum, int16_t N, uint8_t sl)
{
    return (uint32_t)(((uint64_t)accum << sl) / (uint32_t)N);
}

// ── RMS ──────────────────────────────────────────────────────────────────────
//
// All signal types share the same macro (Section 2.1); only the shift amounts
// and accumulator/sqrt width differ.  See macro parameters for per-type values.

// ── Crest factor ─────────────────────────────────────────────────────────────
// peak UQ5.11; SL(12) -> UQ9.23 (32-bit); divide by rms UQ7.9 -> UQ2.14
static inline uq2_14_t fxp_cf_l2g_result(uq5_11_t peak, uq7_9_t rms)
{
    return (uq2_14_t)(((uint32_t)peak << 12) / (uint32_t)rms);
}

// ── ZCR (Zero Crossing Rate) ─────────────────────────────────────────────────
// Counts sign changes in the signed signal.  Result is a ratio in [0,1]
// returned as float.  For unsigned signals ZCR is always 0 by definition.
static inline float fxp_compute_zcr_raw(const q11_5_t *sig, int16_t len)
{
    int sum = 0;
    for (int16_t i = 0; i < len - 1; i++) {
        if ((sig[i] ^ sig[i + 1]) < 0)
            sum++;
    }
    return (float)sum / (float)(len - 1);
}

// ── AZC epsilon converters ────────────────────────────────────────────────────
// Convert a float epsilon threshold to FxP signal units for each signal type.
static inline uint32_t fxp_azc_eps_raw(float eps)
{
    return (uint32_t)FXP_FROM_FLOAT(eps, 5);
}
static inline uint32_t fxp_azc_eps_l2a(float eps)
{
    return (uint32_t)FXP_FROM_FLOAT_U(eps, 6);
}
static inline uint32_t fxp_azc_eps_l2g(float eps)
{
    return (uint32_t)FXP_FROM_FLOAT_U(eps, 11);
}

// =============================================================================
// Section 2 — Generator macros  (must be macros: stamp out named functions)
//
// Each macro expands to a complete C function when instantiated in a .c file
// under #ifdef FXP_MODE.  Q-format arithmetic is entirely in Section 1 helpers
// or expressed as shift-constant arguments — macro bodies are generic loops.
//
// Instantiation map (which .c file owns which macro):
//   fxp_get_rms_{raw,l2a,l2g}, fxp_get_max_l2g   ->  time_domain_feat.c
//   fxp_get_line_length_{raw,l2g},
//   fxp_get_kurtosis_raw,
//   fxp_l2_norm_{accel,gyro}                      ->  helpers.c
//   fxp_azc_computation_{raw,l2a,l2g}             ->  azc.c
// =============================================================================

#ifdef FXP_MODE

// ── 2.1  get_rms ─────────────────────────────────────────────────────────────
// Parameters:
//   SUFFIX     : raw / l2a / l2g
//   sample_t   : input element type
//   result_t   : output type
//   accum_t    : accumulator type (uint64_t for RAW; uint32_t for L2A/L2G)
//   SQ_SR      : right-shift applied to each squared sample before accumulation
//   MEAN_SHIFT : net shift applied to mean-of-squares before sqrt;
//                positive = left-shift (RAW: <<22), negative = right-shift (L2A/L2G: >>1)
//   SQRT_FN    : fxp_sqrt64 (RAW) or fxp_sqrt32 (L2A/L2G)
//
// The MEAN_SHIFT trick: a single signed constant covers both directions.
//   (MEAN_SHIFT) >= 0  ->  mean << MEAN_SHIFT
//   (MEAN_SHIFT) <  0  ->  mean >> (-MEAN_SHIFT)
// Implemented via a ternary that the compiler folds to a single shift at -O1+.
#define FXP_RMS_SHIFT(val, sh) ((sh) >= 0 ? (val) << (sh) : (val) >> (-(sh)))

#define FXP_DEFINE_GET_RMS(SUFFIX, sample_t, result_t, accum_t, SQ_SR, MEAN_SHIFT, SQRT_FN) \
    result_t fxp_get_rms_##SUFFIX(const sample_t *sig, int16_t len)                         \
    {                                                                                       \
        accum_t _sum = 0;                                                                   \
        for (int16_t _i = 0; _i < len; _i++) {                                             \
            uint32_t _sq = (uint32_t)fxp_mul_s32((int32_t)sig[_i], (int32_t)sig[_i]);     \
            _sum += (accum_t)(_sq >> SQ_SR);                                               \
        }                                                                                   \
        return (result_t)SQRT_FN(FXP_RMS_SHIFT(_sum / (accum_t)len, MEAN_SHIFT));          \
    }

// ── 2.2  get_line_length ─────────────────────────────────────────────────────
// Parameters:
//   SUFFIX   : raw / l2g
//   result_t : output type
//   sample_t : input element type
//   SR       : right-shift on |diff| before accumulation
//   SL       : left-shift on accumulator before dividing by N
#define FXP_DEFINE_GET_LINE_LENGTH(SUFFIX, result_t, sample_t, SR, SL)        \
    result_t fxp_get_line_length_##SUFFIX(const sample_t *sig, int16_t len)   \
    {                                                                         \
        uint32_t _accum = 0;                                                  \
        for (int16_t _i = 0; _i < len - 1; _i++) {                           \
            int32_t _diff = (int32_t)sig[_i + 1] - (int32_t)sig[_i];         \
            _accum += fxp_linelen_diff_to_accum(_diff, SR);                   \
        }                                                                     \
        return (result_t)fxp_linelen_result(_accum, len - 1, SL);             \
    }

// ── 2.3  get_kurtosis (RAW only) ─────────────────────────────────────────────
// Spelled out rather than parameterised — the Q-format chain is unique.
// Two passes: first for variance/std, then for the fourth moment.
#define FXP_DEFINE_GET_KURTOSIS_RAW()                                            \
    q34_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len)               \
    {                                                                            \
        /* Pass 1a: mean in Q11.10 */                                            \
        int32_t _sum_mean = 0;                                                   \
        for (int16_t _i = 0; _i < len; _i++)                                     \
            _sum_mean += (int32_t)sig[_i];                                       \
        int32_t _mean_q10 = fxp_kurt_mean_q10(_sum_mean, len);                   \
        /* Pass 1b: variance -> std */                                            \
        uint64_t _sum_var = 0;                                                   \
        for (int16_t _i = 0; _i < len; _i++) {                                   \
            int16_t _c = fxp_kurt_centred(sig[_i], _mean_q10);                   \
            _sum_var += fxp_kurt_sq_to_var_accum(fxp_kurt_sq_dev(_c));           \
        }                                                                        \
        uq10_22_t _var   = fxp_kurt_variance(_sum_var, len);                     \
        uq5_11_t  _std   = fxp_kurt_std(_var);                                   \
        /* Denominator = N * std^4 */                                            \
        uint32_t   _std2  = fxp_kurt_std2(_std);                                 \
        uq20_44_t  _std4  = fxp_kurt_std4(_std2);                                \
        uq20_44_t  _denom = fxp_kurt_denom(_std4, len);                          \
        if (_denom == 0) return 0;  /* std==0: kurtosis undefined */             \
        /* Pass 2: fourth moment accumulation */                                  \
        uint64_t _sum_x4 = 0;                                                    \
        for (int16_t _i = 0; _i < len; _i++) {                                   \
            int16_t _c = fxp_kurt_centred(sig[_i], _mean_q10);                   \
            _sum_x4 += fxp_kurt_c4(fxp_kurt_c2(_c));                             \
        }                                                                        \
        return fxp_kurt_result(_sum_x4, _denom);                                 \
    }

// ── 2.4  get_max (L2_G only) ─────────────────────────────────────────────────
#define FXP_DEFINE_GET_MAX_L2G()                               \
    uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len) \
    {                                                          \
        uq5_11_t _max = sig[0];                                \
        for (int16_t _i = 1; _i < len; _i++) {                \
            if (sig[_i] > _max) _max = sig[_i];               \
        }                                                      \
        return _max;                                           \
    }

// ── 2.5  L2 norm (3 float axes -> one FxP output sample) ─────────────────────
//
// Accel: sqrt(ax^2 + ay^2 + az^2).
//   Sum of squares in Q22.10; SL(2) -> Q22.12; sqrt -> UQ11.6; cast to UQ10.6.
#define FXP_DEFINE_L2_NORM_ACCEL()                                                                                            \
    uq10_6_t fxp_l2_norm_accel(float ax, float ay, float az)                                                                 \
    {                                                                                                                        \
        q11_5_t _ax = FXP_IMU_RAW_FROM_FLOAT(ax);                                                                            \
        q11_5_t _ay = FXP_IMU_RAW_FROM_FLOAT(ay);                                                                            \
        q11_5_t _az = FXP_IMU_RAW_FROM_FLOAT(az);                                                                            \
        uint32_t _sum = (uint32_t)fxp_mul_s32(_ax, _ax) + (uint32_t)fxp_mul_s32(_ay, _ay) + (uint32_t)fxp_mul_s32(_az, _az); \
        return (uq10_6_t)fxp_sqrt32(_sum << 2);                                                                              \
    }

// Gyro: sqrt(gx^2 + gy^2 + gz^2).
//   Sum of squares in Q22.10; SL(12) -> Q22.22; sqrt -> UQ11.11; cast to UQ5.11.
#define FXP_DEFINE_L2_NORM_GYRO()                                                                                            \
    uq5_11_t fxp_l2_norm_gyro(float gx, float gy, float gz)                                                                  \
    {                                                                                                                        \
        q11_5_t _gx = FXP_IMU_RAW_FROM_FLOAT(gx);                                                                            \
        q11_5_t _gy = FXP_IMU_RAW_FROM_FLOAT(gy);                                                                            \
        q11_5_t _gz = FXP_IMU_RAW_FROM_FLOAT(gz);                                                                            \
        uint32_t _sum = (uint32_t)fxp_mul_s32(_gx, _gx) + (uint32_t)fxp_mul_s32(_gy, _gy) + (uint32_t)fxp_mul_s32(_gz, _gz); \
        return (uq5_11_t)fxp_sqrt64((uint64_t)_sum << 12);                                                                   \
    }

// ── 2.6  AZC ─────────────────────────────────────────────────────────────────
//
// The three internal functions (interp, max_vdist, polygonal_approx) operate
// on int32_t so a single implementation serves all signal types.
// FXP_DEFINE_AZC_COMPUTATION stamps out one named entry-point per signal type;
// it widens the typed input array to int32_t before calling the shared impl.

static inline int32_t _fxp_azc_diff(int32_t a, int32_t b, int16_t gap)
{
    return (gap == 0) ? 0 : (b - a) / (int32_t)gap;
}

static void _fxp_azc_interp(int16_t len, int16_t xf, int32_t yf,
                             int16_t xl, int32_t yl, int32_t *res)
{
    res[0] = yf;
    res[len - 1] = yl;
    int32_t _dy = yl - yf;
    int16_t _dx = xl - xf;
    for (int16_t _i = 1; _i < len - 1; _i++)
        res[_i] = yf + (_dy * _i) / _dx;
}

static uint32_t _fxp_azc_max_vdist(const int32_t *sig, int16_t first,
                                    int16_t last, int16_t *idx)
{
    if (first == last) { *idx = first; return 0; }
    int16_t _len = last - first + 1;
    int32_t *_intrp = (int32_t *)malloc((size_t)_len * sizeof(int32_t));
    _fxp_azc_interp(_len, first, sig[first], last, sig[last], _intrp);
    uint32_t _max_d = 0;
    *idx = first;
    for (int16_t _i = 0; _i < _len; _i++) {
        uint32_t _d = (uint32_t)fxp_abs_s32(sig[first + _i] - _intrp[_i]);
        if (_d > _max_d) { _max_d = _d; *idx = first + _i; }
    }
    free(_intrp);
    return _max_d;
}

static int16_t *_fxp_azc_polygonal_approx(const int32_t *sig, int16_t len,
                                           uint32_t eps_fxp, int16_t *res_len)
{
    int16_t *_res = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    int16_t _found = 0;
    typedef struct { int16_t first; int16_t last; } _seg_t;
    _seg_t *_stk = (_seg_t *)malloc((size_t)len * sizeof(_seg_t));
    _stk[0].first = 0;
    _stk[0].last  = len - 1;
    int16_t _next = 0;
    while (_next >= 0) {
        int16_t _f = _stk[_next].first;
        int16_t _l = _stk[_next].last;
        _next--;
        int16_t _mid;
        uint32_t _md = _fxp_azc_max_vdist(sig, _f, _l, &_mid);
        if (_md > eps_fxp) {
            _stk[_next + 1].first = _f;
            _stk[_next + 1].last  = _mid;
            _stk[_next + 2].first = _mid;
            _stk[_next + 2].last  = _l;
            _next += 2;
        } else {
            int16_t _af = 1, _al = 1;
            for (int16_t _j = 0; _j < _found; _j++) {
                if (_f == _res[_j]) _af = 0;
                if (_l == _res[_j]) _al = 0;
            }
            if (_af) _res[_found++] = _f;
            if (_al) _res[_found++] = _l;
        }
    }
    free(_stk);
    *res_len = _found;
    return _res;
}

static inline int _fxp_qsort_cmp(const void *a, const void *b)
{
    return (int)(*(const int16_t *)a) - (int)(*(const int16_t *)b);
}

static int16_t _fxp_azc_impl(const int32_t *sig, int16_t len, uint32_t eps_fxp)
{
    int16_t _alen = 0;
    int16_t *_aidxs = _fxp_azc_polygonal_approx(sig, len, eps_fxp, &_alen);
    qsort(_aidxs, (size_t)_alen, sizeof(int16_t), _fxp_qsort_cmp);
    int16_t _azc = 0;
    if (_alen > 2) {
        int32_t _prev = _fxp_azc_diff((int32_t)sig[_aidxs[0]],
                                       (int32_t)sig[_aidxs[1]],
                                       (int16_t)(_aidxs[1] - _aidxs[0]));
        for (int16_t _i = 1; _i < _alen - 1; _i++) {
            int32_t _cur = _fxp_azc_diff((int32_t)sig[_aidxs[_i]],
                                          (int32_t)sig[_aidxs[_i + 1]],
                                          (int16_t)(_aidxs[_i + 1] - _aidxs[_i]));
            if ((_prev > 0 && _cur < 0) || (_prev < 0 && _cur > 0)) _azc++;
            _prev = _cur;
        }
    }
    free(_aidxs);
    return _azc;
}

// Thin per-type entry point: widens typed array to int32_t, calls shared impl.
// EPS_FN is one of fxp_azc_eps_raw / _l2a / _l2g (inline functions, Section 1).
#define FXP_DEFINE_AZC_COMPUTATION(SUFFIX, sample_t, EPS_FN)                    \
    int16_t fxp_azc_computation_##SUFFIX(const sample_t *sig, int16_t len,       \
                                         float epsilon)                          \
    {                                                                            \
        uint32_t _eps = EPS_FN(epsilon);                                         \
        int32_t *_wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));       \
        for (int16_t _i = 0; _i < len; _i++) _wide[_i] = (int32_t)sig[_i];      \
        int16_t _azc = _fxp_azc_impl(_wide, len, _eps);                          \
        free(_wide);                                                             \
        return _azc;                                                             \
    }

#endif // FXP_MODE

// =============================================================================
// Section 3 — Forward declarations for generated functions
//
// These declarations let every translation unit call the generated kernels
// without seeing the instantiation site.  The linker resolves each symbol
// to its definition in the owning .c file.
// =============================================================================
#ifdef FXP_MODE

uq16_16_t fxp_get_rms_raw(const q11_5_t *sig, int16_t len);
uq13_3_t  fxp_get_rms_l2a(const uq10_6_t *sig, int16_t len);
uq7_9_t   fxp_get_rms_l2g(const uq5_11_t *sig, int16_t len);

uq9_23_t fxp_get_line_length_raw(const q11_5_t *sig, int16_t len);
uq7_9_t  fxp_get_line_length_l2g(const uq5_11_t *sig, int16_t len);

q34_30_t fxp_get_kurtosis_raw(const q11_5_t *sig, int16_t len);

uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len);

uq10_6_t fxp_l2_norm_accel(float ax, float ay, float az);
uq5_11_t fxp_l2_norm_gyro(float gx, float gy, float gz);

int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, float epsilon);
int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, float epsilon);

#endif // FXP_MODE
