#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <audio_features.h>
#include <kiss_fftr.h>
#include <mel_basis.h>
#include <mfcc_hann_wind.h>
#include <mfcc_module.h>

#include <audio/audio_mel_block.h>
#include <audio/audio_periodogram_lut.h>
#include <core/fxp_convert.h>
#include <core/fxp_math.h>
#include <core/fxp_sat.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

#define FXP_MEL_WIN_FRAC 15
#define FXP_MEL_BASIS_FRAC 15
#define FXP_MEL_STATS_FRAC 11
#define FXP_MEL_PROB_FRAC 24

#define FXP_MEL_DB_PER_LN_Q20 ((int32_t)4553913) /* round((10/ln(10))*2^20) */
#define FXP_LN2_Q24 ((int32_t)11629080)          /* round(ln(2) * 2^24) */
#define FXP_LN2_Q11 ((int32_t)((FXP_LN2_Q24 + (1 << 12)) >> 13))
#define FXP_MEL_LN_Q11_SCALE_P ((int32_t)(FXP_MEL_PROB_FRAC * FXP_LN2_Q11))
#define FXP_MEL_TOP_DB_Q11 ((int32_t)(TOP_DB * (1 << FXP_MEL_STATS_FRAC)))
#define FXP_MEL_DB_PER_SHIFT_Q11 ((int32_t)6165) /* round((10*log10(2))*2^11) */
#define FXP_MEL_POWER_MSB_TARGET 46U
#define FXP_MEL_ENT_ALIGN_FRAC 8U

#if (FIXED_POINT == 32)
#define FXP_MEL_INPUT_FRAC 31
#define FXP_MEL_SCALAR_MAX_F 2147483647.0f
#define FXP_MEL_SCALAR_MAX_I INT32_MAX
typedef int32_t fxp_mel_sig_t;
#else
#define FXP_MEL_INPUT_FRAC 15
#define FXP_MEL_SCALAR_MAX_F 32767.0f
#define FXP_MEL_SCALAR_MAX_I INT16_MAX
typedef int16_t fxp_mel_sig_t;
#endif

static uint16_t _hann_q15[N_FFT];
static uint16_t _mel_basis_q15[MEL_ROWS][MAX_NZ_ELEMS];
static int _mel_tables_initialized = 0;
static float _kiss_gain_f = (float)N_FFT;
static int _kiss_gain_initialized = 0;

static int _mel_any_required(const int8_t *features_selector)
{
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT; i < ZERO_CROSSING_RATE; i++) {
        if (features_selector[i]) return 1;
    }
    return 0;
}

static inline int32_t _round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static inline int64_t _round_div_s64_to_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (num + (den / 2)) / den;
    return -(((-num) + (den / 2)) / den);
}

static inline uint64_t _round_shift_u64(uint64_t v, uint32_t shift)
{
    if (shift == 0U) return v;
    if (shift >= 64U) return 0ULL;
    return (v + (1ULL << (shift - 1U))) >> shift;
}

/* Computes round((num / den) * 2^frac_bits) without 128-bit arithmetic. */
static inline uint32_t _uq_div_u64_q(uint64_t num, uint64_t den, uint8_t frac_bits)
{
    if (den == 0ULL || num == 0ULL) return 0U;

    uint64_t q = num / den;
    uint64_t r = num % den;

    for (uint8_t i = 0; i < frac_bits; i++) {
        if (q > (UINT64_MAX >> 1U)) {
            q = UINT64_MAX;
        } else {
            q <<= 1U;
        }

        /* 2*r >= den check without overflow: r >= den - r */
        if (r >= (den - r)) {
            r = r - (den - r);
            q |= 1ULL;
        } else {
            r += r;
        }
    }

    /* Round to nearest by testing one more fractional bit. */
    if (r >= (den - r) && q < UINT64_MAX) {
        q += 1ULL;
    }

    if (q > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)q;
}

static inline uint32_t _pick_power_shift(uint64_t max_power)
{
    if (max_power == 0ULL) return 0U;
    uint32_t msb = 63U - (uint32_t)__builtin_clzll(max_power);
    if (msb <= FXP_MEL_POWER_MSB_TARGET) return 0U;
    return msb - FXP_MEL_POWER_MSB_TARGET;
}

static inline uint8_t _ceil_log2_u16(uint16_t v)
{
    if (v <= 1U) return 0U;
    uint16_t x = (uint16_t)(v - 1U);
    uint8_t bits = 0U;
    while (x) {
        x >>= 1U;
        bits++;
    }
    return bits;
}

static inline uint64_t _entropy_align_term(uint64_t value, uint8_t frame_shift, uint8_t max_frame_shift)
{
    if (value == 0ULL) return 0ULL;
    uint8_t dshift = (uint8_t)(max_frame_shift - frame_shift);
    if (dshift <= FXP_MEL_ENT_ALIGN_FRAC) {
        uint32_t lshift = (uint32_t)(FXP_MEL_ENT_ALIGN_FRAC - dshift);
        if (lshift >= 64U) return 0ULL;
        if (value > (UINT64_MAX >> lshift)) return UINT64_MAX;
        return value << lshift;
    }
    return _round_shift_u64(value, (uint32_t)(dshift - FXP_MEL_ENT_ALIGN_FRAC));
}

static uint16_t _to_uq15(float v)
{
    uint32_t q = FXP_FROM_FLOAT_U(v, FXP_MEL_WIN_FRAC);
    return fxp_sat_u16_from_u32(q);
}

static inline fxp_mel_sig_t _to_sig_q(float x)
{
    int32_t q = fxp_from_float_signed(x, FXP_MEL_INPUT_FRAC);
#if (FIXED_POINT == 32)
    return q;
#else
    return fxp_sat_s16_from_s32(q);
#endif
}

static void _ensure_kiss_gain(void)
{
    if (_kiss_gain_initialized) return;

    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);
    kiss_fft_scalar *x = (kiss_fft_scalar *)calloc((size_t)N_FFT, sizeof(kiss_fft_scalar));
    kiss_fft_cpx *y = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));

    if (!cfg || !x || !y) {
        free(cfg);
        free(x);
        free(y);
        _kiss_gain_f = (float)N_FFT;
        _kiss_gain_initialized = 1;
        return;
    }

    x[0] = (kiss_fft_scalar)FXP_MEL_SCALAR_MAX_I;
    kiss_fftr(cfg, x, y);

    float ref = ((float)y[1].r) / FXP_MEL_SCALAR_MAX_F;
    if (fabsf(ref) < 1e-12f) {
        ref = ((float)y[0].r) / FXP_MEL_SCALAR_MAX_F;
    }
    if (fabsf(ref) < 1e-12f) ref = 1.0f;
    _kiss_gain_f = 1.0f / fabsf(ref);
    _kiss_gain_initialized = 1;

    free(cfg);
    free(x);
    free(y);
}

static void _ensure_mel_tables(void)
{
    if (_mel_tables_initialized) return;

    for (int16_t i = 0; i < N_FFT; i++) {
        _hann_q15[i] = _to_uq15(hann_mfcc_wind[i]);
    }

    memset(_mel_basis_q15, 0, sizeof(_mel_basis_q15));
    for (int16_t i = 0; i < MEL_ROWS; i++) {
        int16_t start = mel_nz_indexes[i][0];
        int16_t end = mel_nz_indexes[i][1];
        for (int16_t k = start; k <= end; k++) {
            _mel_basis_q15[i][k - start] = _to_uq15(mel_basis[i][k - start]);
        }
    }

    _mel_tables_initialized = 1;
}

/* Natural logarithm on unsigned integer input, result in Q11. */
static int32_t _fxp_ln_u64_q11(uint64_t x)
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

static int32_t _db_from_power_q11(uint64_t p_scaled, int32_t db_offset_q11)
{
    int32_t ln_q11 = _fxp_ln_u64_q11((p_scaled == 0ULL) ? 1ULL : p_scaled);
    int32_t db_q11 = (int32_t)((((int64_t)ln_q11 * (int64_t)FXP_MEL_DB_PER_LN_Q20) + (1LL << 19)) >> 20);
    return db_q11 + db_offset_q11;
}

static int32_t _entropy_row_q11(const uint64_t *row_power,
                                const uint8_t *frame_shift,
                                int16_t n_frames)
{
    if (!row_power || !frame_shift || n_frames <= 0) return 0;

    uint64_t row_max = 0ULL;
    uint8_t row_max_shift = 0U;
    for (int16_t t = 0; t < n_frames; t++) {
        if (row_power[t] > row_max) row_max = row_power[t];
        if (row_power[t] > 0ULL && frame_shift[t] > row_max_shift) {
            row_max_shift = frame_shift[t];
        }
    }
    if (row_max == 0ULL) return 0;

    uint32_t row_msb = 63U - (uint32_t)__builtin_clzll(row_max);
    uint8_t sum_bits = _ceil_log2_u16((uint16_t)n_frames);
    int32_t pre_shift_i = (int32_t)row_msb + (int32_t)FXP_MEL_ENT_ALIGN_FRAC + (int32_t)sum_bits - 62;
    uint32_t pre_shift = (pre_shift_i > 0) ? (uint32_t)pre_shift_i : 0U;

    uint64_t row_sum = 0ULL;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t v = _round_shift_u64(row_power[t], pre_shift);
        uint64_t term = _entropy_align_term(v, frame_shift[t], row_max_shift);
        if (term == 0ULL) continue;
        if (UINT64_MAX - row_sum < term) {
            row_sum = UINT64_MAX;
        } else {
            row_sum += term;
        }
    }
    if (row_sum == 0ULL) return 0;

    int64_t entropy_q11 = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t v = _round_shift_u64(row_power[t], pre_shift);
        uint64_t term = _entropy_align_term(v, frame_shift[t], row_max_shift);
        if (term == 0ULL) continue;

        uint32_t p_qp = _uq_div_u64_q(term, row_sum, FXP_MEL_PROB_FRAC);
        if (p_qp == 0U) continue;

        int32_t ln_p_q11 = _fxp_ln_u64_q11((uint64_t)p_qp) - FXP_MEL_LN_Q11_SCALE_P;
        if (ln_p_q11 > 0) ln_p_q11 = 0;

        int64_t contrib_q11 = -((((int64_t)p_qp * (int64_t)ln_p_q11) + (1LL << (FXP_MEL_PROB_FRAC - 1))) >> FXP_MEL_PROB_FRAC);
        entropy_q11 += contrib_q11;
    }

    return fxp_sat_s32_from_s64(entropy_q11);
}

void fxp_audio_mel_features_from_signal(const int8_t *features_selector,
                                        const float *sig,
                                        int16_t len,
                                        float *feats)
{
    if (!features_selector || !sig || !feats || len <= 0) return;
    if (!_mel_any_required(features_selector)) return;
    if (len <= PAD_LEN) return;

    uint8_t idxs_needed[N_MFCC];
    int16_t n_mels_needed = 0;
    for (uint8_t i = 0; i < N_MFCC; i++) {
        if (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i]) {
            idxs_needed[n_mels_needed] = i;
            n_mels_needed++;
        }
    }
    if (n_mels_needed <= 0) return;

    int16_t padded_len = (int16_t)(len + (2 * PAD_LEN));
    int16_t n_frames = (int16_t)(((padded_len - N_FFT) / HOP_LEN) + 1);
    if (n_frames <= 0) return;

    fxp_mel_sig_t *sig_q = (fxp_mel_sig_t *)malloc((size_t)len * sizeof(fxp_mel_sig_t));
    fxp_mel_sig_t *padded_q = (fxp_mel_sig_t *)malloc((size_t)padded_len * sizeof(fxp_mel_sig_t));
    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    uint64_t *frame_power = (uint64_t *)malloc((size_t)FFT_RES_LEN * sizeof(uint64_t));
    uint64_t *mel_power = (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));
    int32_t *mel_db_q11 = (int32_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int32_t));
    uint8_t *frame_shift = (uint8_t *)malloc((size_t)n_frames * sizeof(uint8_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!sig_q || !padded_q || !timedata || !cx_out || !frame_power || !mel_power || !mel_db_q11 || !frame_shift || !cfg) {
        free(sig_q);
        free(padded_q);
        free(timedata);
        free(cx_out);
        free(frame_power);
        free(mel_power);
        free(mel_db_q11);
        free(frame_shift);
        free(cfg);
        return;
    }

    _ensure_mel_tables();
    _ensure_kiss_gain();

    float max_abs = 0.0f;
    for (int16_t i = 0; i < len; i++) {
        float a = fabsf(sig[i]);
        if (a > max_abs) max_abs = a;
    }
    float signal_scale = (max_abs > 1.0f) ? max_abs : 1.0f;
    float inv_signal_scale = (signal_scale > 0.0f) ? (1.0f / signal_scale) : 1.0f;

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = _to_sig_q(sig[i] * inv_signal_scale);
    }

    for (int16_t i = 0; i < PAD_LEN; i++) {
        padded_q[i] = sig_q[PAD_LEN - i];
        padded_q[PAD_LEN + len + i] = sig_q[len - 2 - i];
    }
    for (int16_t i = 0; i < len; i++) {
        padded_q[PAD_LEN + i] = sig_q[i];
    }

    float db_offset_f =
        20.0f * log10f(_kiss_gain_f / FXP_MEL_SCALAR_MAX_F) +
        20.0f * log10f(signal_scale);
    int32_t db_offset_base_q11 = FXP_FROM_FLOAT(db_offset_f, FXP_MEL_STATS_FRAC);

    int32_t max_db_q11 = INT32_MIN;
    for (int16_t f = 0; f < n_frames; f++) {
        int32_t frame_start = (int32_t)f * HOP_LEN;

        for (int16_t n = 0; n < N_FFT; n++) {
            int32_t centered_q = (int32_t)padded_q[frame_start + n];
            int64_t prod = (int64_t)centered_q * (int64_t)_hann_q15[n];
            int32_t win_q;
            if (prod >= 0) {
                win_q = (int32_t)((prod + (1LL << (FXP_MEL_WIN_FRAC - 1))) >> FXP_MEL_WIN_FRAC);
            } else {
                win_q = -(int32_t)(((-prod) + (1LL << (FXP_MEL_WIN_FRAC - 1))) >> FXP_MEL_WIN_FRAC);
            }

#if (FIXED_POINT == 32)
            timedata[n] = (kiss_fft_scalar)win_q;
#else
            timedata[n] = (kiss_fft_scalar)fxp_sat_s16_from_s32(win_q);
#endif
        }

        kiss_fftr(cfg, timedata, cx_out);

        uint64_t max_power = 0ULL;
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            int64_t re = (int64_t)cx_out[k].r;
            int64_t im = (int64_t)cx_out[k].i;
            uint64_t p = (uint64_t)(re * re) + (uint64_t)(im * im);
            frame_power[k] = p;
            if (p > max_power) max_power = p;
        }
        uint8_t cur_shift = (uint8_t)_pick_power_shift(max_power);
        frame_shift[f] = cur_shift;
        int32_t frame_db_offset_q11 = db_offset_base_q11 + (int32_t)cur_shift * FXP_MEL_DB_PER_SHIFT_Q11;

        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            frame_power[k] = _round_shift_u64(frame_power[k], cur_shift);
        }

        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)idxs_needed[m];
            int16_t start = mel_nz_indexes[mel_idx][0];
            int16_t end = mel_nz_indexes[mel_idx][1];

            uint64_t sum = 0ULL;
            for (int16_t k = start; k <= end; k++) {
                uint16_t w_q15 = _mel_basis_q15[mel_idx][k - start];
                uint64_t term = ((frame_power[k] * (uint64_t)w_q15) + (1ULL << (FXP_MEL_BASIS_FRAC - 1))) >> FXP_MEL_BASIS_FRAC;
                if (UINT64_MAX - sum < term) {
                    sum = UINT64_MAX;
                } else {
                    sum += term;
                }
            }

            mel_power[(size_t)m * (size_t)n_frames + (size_t)f] = sum;
            mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f] = _db_from_power_q11(sum, frame_db_offset_q11);
            if (mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f] > max_db_q11) {
                max_db_q11 = mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f];
            }
        }
    }

    int32_t clip_floor_q11 = max_db_q11 - FXP_MEL_TOP_DB_Q11;

    for (int16_t m = 0; m < n_mels_needed; m++) {
        int64_t sum_db_q11 = 0;
        int32_t row_max_q11 = INT32_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int32_t v = mel_db_q11[idx];
            if (v < clip_floor_q11) v = clip_floor_q11;
            mel_db_q11[idx] = v;
            sum_db_q11 += (int64_t)v;
            if (v > row_max_q11) row_max_q11 = v;
        }

        int32_t mean_q11 = _round_div_s64(sum_db_q11, n_frames);

        int64_t sum_sq_q22 = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            int32_t d = mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f] - mean_q11;
            sum_sq_q22 += (int64_t)d * (int64_t)d;
        }
        int64_t var_q22 = _round_div_s64_to_s64(sum_sq_q22, n_frames);
        if (var_q22 < 0) var_q22 = 0;
        int32_t std_q11 = fxp_sat_s32_from_s64((int64_t)fxp_sqrt64((uint64_t)var_q22));

        int32_t ent_q11 = _entropy_row_q11(&mel_power[(size_t)m * (size_t)n_frames],
                                           frame_shift,
                                           n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] = FXP_TO_FLOAT(mean_q11, FXP_MEL_STATS_FRAC);
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] = FXP_TO_FLOAT(std_q11, FXP_MEL_STATS_FRAC);
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] = FXP_TO_FLOAT(row_max_q11, FXP_MEL_STATS_FRAC);
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] = FXP_TO_FLOAT(ent_q11, FXP_MEL_STATS_FRAC);
    }

    free(sig_q);
    free(padded_q);
    free(timedata);
    free(cx_out);
    free(frame_power);
    free(mel_power);
    free(mel_db_q11);
    free(frame_shift);
    free(cfg);
}

#endif
