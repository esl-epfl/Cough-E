#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <audio/audio_pipeline_fxp.h>

#include <kiss_fftr.h>
#include <mfcc_module.h>
#include <welch_psd.h>

#include <audio/audio_periodogram_lut.h>
#include <audio/hann_window_q15.h>
#include <audio/mel_basis_q15.h>
#include <audio/mfcc_hann_wind_q15.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

/* -------------------------------------------------------------------------- */
/*  FFT feature kernels + block                                               */
/* -------------------------------------------------------------------------- */

#define FXP_FFT_ROLLOFF_95_Q16 ((uint32_t)62259U) /* round(0.95 * 2^16) */

static inline q13_19_t _audio_fft_dev_q19(uq12_20_t freq_q20, uq10_21_t centroid_q21)
{
    uint32_t freq_q19 = freq_q20 >> 1;
    uint32_t cent_q19 = centroid_q21 >> 2;
    return (q13_19_t)((int32_t)freq_q19 - (int32_t)cent_q19);
}

static uq12_20_t _audio_fft_rolloff_q20(const fxp_audio_fft_view_t *view)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0) return 0;

    uq15_17_t rolloff_energy_q17 =
        (uq15_17_t)((((uint64_t)view->sum_mags_q17 * (uint64_t)FXP_FFT_ROLLOFF_95_Q16) + (1ULL << 15)) >> 16);

    uq15_17_t running_sum_q17 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        running_sum_q17 += (view->mags_q20[i] >> 3);
        if (running_sum_q17 >= rolloff_energy_q17) {
            return view->freqs_q20[i];
        }
    }
    return view->freqs_q20[view->len - 1];
}

static uq10_21_t _audio_fft_centroid_q21(const fxp_audio_fft_view_t *view)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0) return 0;

    uint64_t accum_q23 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        uint64_t prod_q24_40 = (uint64_t)view->freqs_q20[i] * (uint64_t)view->mags_q20[i];
        uint64_t term_q23 = (prod_q24_40 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        accum_q23 += term_q23;
    }

    return (uq10_21_t)fxp_sat_u32_from_u64((accum_q23 + 2ULL) >> 2);
}

static uq11_5_t _audio_fft_spread_q5(const fxp_audio_fft_view_t *view, uq10_21_t centroid_q21)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0) return 0;

    uint64_t mean_q22_10 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        q13_19_t dev_q19 = _audio_fft_dev_q19(view->freqs_q20[i], centroid_q21);
        uint64_t dev2_q25_7 = ((uint64_t)((int64_t)dev_q19 * (int64_t)dev_q19)) >> 31;
        uint64_t weighted_q37_27 = dev2_q25_7 * (uint64_t)view->mags_q20[i];
        uint64_t term_q10 = (weighted_q37_27 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        mean_q22_10 += term_q10;
    }

    return (uq11_5_t)fxp_sat_u16_from_u32((uint32_t)fxp_sqrt64(mean_q22_10));
}

static uq7_15_t _audio_fft_kurtosis_q15(const fxp_audio_fft_view_t *view,
                                        uq10_21_t centroid_q21,
                                        uq11_5_t spread_q5)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0 || spread_q5 == 0) return 0;

    uint64_t kurt_q15 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        q13_19_t dev_q19 = _audio_fft_dev_q19(view->freqs_q20[i], centroid_q21);

        int64_t num = (dev_q19 >= 0) ? ((int64_t)dev_q19 >> 3) : -(((int64_t)(-dev_q19)) >> 3);
        int64_t norm_q11 = (num >= 0)
                               ? ((num + ((int64_t)spread_q5 >> 1)) / (int64_t)spread_q5)
                               : -(((-num) + ((int64_t)spread_q5 >> 1)) / (int64_t)spread_q5);

        uint64_t abs_norm_q11 = (norm_q11 < 0) ? (uint64_t)(-norm_q11) : (uint64_t)norm_q11;
        if (abs_norm_q11 > 65535ULL) abs_norm_q11 = 65535ULL;

        uint64_t norm2_q22 = abs_norm_q11 * abs_norm_q11;
        uint64_t norm4_q12 = (norm2_q22 * norm2_q22 + (1ULL << 31)) >> 32;

        uint64_t weighted_q32 = norm4_q12 * (uint64_t)view->mags_q20[i];
        uint64_t term_q15 = (weighted_q32 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        kurt_q15 += term_q15;
    }

    return (uq7_15_t)fxp_sat_u32_from_u64(kurt_q15);
}

static void _fxp_audio_fft_write_features_q16(const int8_t *features_selector,
                                              const fxp_audio_fft_view_t *view,
                                              fxp_q16_t *feats_q16)
{
    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];

    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    if (need_rolloff) {
        uq12_20_t rolloff_q20 = _audio_fft_rolloff_q20(view);
        feats_q16[SPECTRAL_ROLLOFF] = fxp_q16_from_u32(rolloff_q20, FXP_FRAC_AUDIO_FFT_FREQUENCIES);
    }

    uq10_21_t centroid_q21 = 0;
    if (need_centroid) {
        centroid_q21 = _audio_fft_centroid_q21(view);
        if (features_selector[SPECTRAL_CENTROID]) {
            feats_q16[SPECTRAL_CENTROID] = fxp_q16_from_u32(centroid_q21, FXP_FRAC_AUDIO_FFT_CENTROID);
        }
    }

    uq11_5_t spread_q5 = 0;
    if (need_spread) {
        spread_q5 = _audio_fft_spread_q5(view, centroid_q21);
        if (features_selector[SPECTRAL_SPREAD]) {
            feats_q16[SPECTRAL_SPREAD] = fxp_q16_from_u32((uint32_t)spread_q5, FXP_FRAC_AUDIO_FFT_SPREAD);
        }
    }

    if (need_kurt) {
        uq7_15_t kurt_q15 = _audio_fft_kurtosis_q15(view, centroid_q21, spread_q5);
        feats_q16[SPECTRAL_KURTOSIS] = fxp_q16_from_u32(kurt_q15, FXP_FRAC_AUDIO_FFT_KURTOSIS);
    }
}

void fxp_audio_fft_features_from_q14(const int8_t *features_selector,
                                     const int16_t *sig_q14,
                                     int16_t len,
                                     int16_t fs,
                                     fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig_q14 || !feats_q16 || len <= 0 || fs <= 0) return;

    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];
    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    int16_t fft_len = (int16_t)((len / 2) + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    if (!cfg) return;

    kiss_fft_scalar *sig_q = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));
    uq12_20_t *mags_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uint32_t *mag_raw = (uint32_t *)malloc((size_t)fft_len * sizeof(uint32_t));

    if (!sig_q || !cx_out || !mags_q20 || !freqs_q20 || !mag_raw) {
        free(sig_q);
        free(cx_out);
        free(mags_q20);
        free(freqs_q20);
        free(mag_raw);
        free(cfg);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = (kiss_fft_scalar)sig_q14[i];
    }

    kiss_fftr(cfg, sig_q, cx_out);

    uint32_t max_mag = 0U;
    for (int16_t i = 0; i < fft_len; i++) {
        int64_t re = (int64_t)cx_out[i].r;
        int64_t im = (int64_t)cx_out[i].i;
        uint64_t re_sq = (uint64_t)(re * re);
        uint64_t im_sq = (uint64_t)(im * im);
        uint64_t mag = fxp_sqrt64(re_sq + im_sq);
        uint32_t mag_u32 = fxp_sat_u32_from_u64(mag);
        mag_raw[i] = mag_u32;
        if (mag_u32 > max_mag) max_mag = mag_u32;
    }
    if (max_mag == 0U) max_mag = 1U;

    uint64_t sum_q17 = 0ULL;
    for (int16_t i = 0; i < fft_len; i++) {
        uint64_t scaled_q20 =
            (((uint64_t)mag_raw[i] << FXP_FRAC_AUDIO_FFT_MAGNITUDES) + ((uint64_t)max_mag >> 1U)) / (uint64_t)max_mag;
        mags_q20[i] = fxp_sat_u32_from_u64(scaled_q20);
        sum_q17 += (uint64_t)(mags_q20[i] >> 3);

        uint64_t freq_q20 =
            (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)len;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    fxp_audio_fft_view_t view = {
        .mags_q20 = mags_q20,
        .freqs_q20 = freqs_q20,
        .len = fft_len,
        .sum_mags_q17 = fxp_sat_u32_from_u64(sum_q17),
    };

    _fxp_audio_fft_write_features_q16(features_selector, &view, feats_q16);

    free(sig_q);
    free(cx_out);
    free(mags_q20);
    free(freqs_q20);
    free(mag_raw);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Periodogram kernels + block                                               */
/* -------------------------------------------------------------------------- */

#define FXP_PSD_LN2_Q24 ((int32_t)11629080)
#define FXP_PSD_PROXY_TO_INT_SHIFT (FXP_FRAC_AUDIO_PSD_PROXY - FXP_FRAC_AUDIO_PSD_INTEGRAL)
#define FXP_HANN_FRAC_BITS 15

#if (FIXED_POINT == 32)
#define FXP_PSD_SIG_FRAC_BITS 30
typedef int32_t fxp_psd_sig_t;
#else
#define FXP_PSD_SIG_FRAC_BITS FXP_FRAC_AUDIO_INPUT
typedef int16_t fxp_psd_sig_t;
#endif

static inline int32_t _psd_round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static inline uint32_t _psd_round_div_u64(uint64_t num, uint32_t den)
{
    if (den == 0U) return 0U;
    return (uint32_t)((num + ((uint64_t)den >> 1)) / (uint64_t)den);
}

static inline int32_t _psd_floor_div_s64(int64_t num, int32_t den)
{
    int64_t q = num / (int64_t)den;
    int64_t r = num - q * (int64_t)den;
    if (r != 0 && num < 0) q -= 1;
    return (int32_t)q;
}

static inline uint64_t _psd_round_shift_u64(uint64_t v, uint32_t shift)
{
    if (shift == 0U) return v;
    if (shift >= 64U) return 0ULL;
    return (v + (1ULL << (shift - 1U))) >> shift;
}

static inline fxp_psd_sig_t _psd_to_sig_q(int16_t x_q14)
{
#if (FIXED_POINT == 32)
    return (fxp_psd_sig_t)(((int32_t)x_q14) << (FXP_PSD_SIG_FRAC_BITS - FXP_FRAC_AUDIO_INPUT));
#else
    return (fxp_psd_sig_t)x_q14;
#endif
}

/* Natural logarithm on UQ18.14 input, result in Q5.11. */
static q5_11_t _psd_ln_proxy_q11(uq18_14_t x_q14)
{
    if (x_q14 == 0U) x_q14 = 1U;

    uint32_t msb = 31U - (uint32_t)__builtin_clz(x_q14);
    uint32_t base = (uint32_t)1U << msb;
    uint32_t frac_q24 = (uint32_t)((((uint64_t)(x_q14 - base)) << 24) / (uint64_t)base);

    uint32_t idx = frac_q24 >> 16;
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = frac_q24 & 0xFFFFU;

    int32_t y0 = fxp_ln_lut_q24[idx];
    int32_t y1 = fxp_ln_lut_q24[idx + 1];
    int32_t y = y0 + (int32_t)((((int64_t)(y1 - y0) * (int64_t)alpha) + (1LL << 15)) >> 16);

    int32_t exp2 = (int32_t)msb - FXP_FRAC_AUDIO_PSD_PROXY;
    int64_t ln_x_q24 = (int64_t)exp2 * (int64_t)FXP_PSD_LN2_Q24 + (int64_t)y;
    int64_t ln_x_q11 = (ln_x_q24 >= 0) ? ((ln_x_q24 + (1LL << 12)) >> 13) : -(((-ln_x_q24) + (1LL << 12)) >> 13);

    return fxp_sat_s16_from_s32((int32_t)ln_x_q11);
}

static uq0_16_t _psd_exp_q16_from_q11(q5_11_t x_q11)
{
    int64_t x_q24 = (int64_t)x_q11 * (int64_t)(1U << 13);
    int32_t k = _psd_floor_div_s64(x_q24, FXP_PSD_LN2_Q24);
    int64_t rem_q24 = x_q24 - (int64_t)k * (int64_t)FXP_PSD_LN2_Q24;
    if (rem_q24 < 0) rem_q24 = 0;

    uint32_t z_q24 = (uint32_t)(((uint64_t)rem_q24 << 24) / (uint32_t)FXP_PSD_LN2_Q24);
    uint32_t idx = z_q24 >> 16;
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = z_q24 & 0xFFFFU;

    uint32_t y0 = fxp_exp_lut_q24[idx];
    uint32_t y1 = fxp_exp_lut_q24[idx + 1];
    uint32_t er_q24 = y0 + (uint32_t)((((int64_t)((int32_t)y1 - (int32_t)y0) * (int64_t)alpha) + (1LL << 15)) >> 16);
    uint32_t er_q16 = (er_q24 + (1U << 7)) >> 8;

    uint64_t out_q16;
    if (k >= 0) {
        if (k >= 16) return UINT16_MAX;
        out_q16 = ((uint64_t)er_q16) << (uint32_t)k;
    } else {
        uint32_t shift = (uint32_t)(-k);
        if (shift >= 32) out_q16 = 0;
        else out_q16 = (((uint64_t)er_q16) + ((uint64_t)1U << (shift - 1U))) >> shift;
    }

    return fxp_sat_u16_from_u32(fxp_sat_u32_from_u64(out_q16));
}

static uint64_t _psd_simpson_step_q8(const uq18_14_t *x_q14, int16_t start, int16_t end)
{
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uint64_t sum_q8 = 0;

    for (int i = 0; i < n_intervals; i++) {
        uint64_t x0_q8 = (uint64_t)(x_q14[idx] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint64_t x1_q8 = (uint64_t)(x_q14[idx + 1] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint64_t x2_q8 = (uint64_t)(x_q14[idx + 2] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        sum_q8 += x0_q8 + (x1_q8 << 2) + x2_q8;
        idx += 2;
    }

    return (sum_q8 + 1ULL) / 3ULL;
}

static uint64_t _psd_simpson_q8(const uq18_14_t *x_q14, int16_t len)
{
    if (!x_q14 || len <= 1) return 0ULL;

    if ((len & 1) == 0) {
        uint64_t val_q8 = (((uint64_t)(x_q14[len - 1] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                            (uint64_t)(x_q14[len - 2] >> FXP_PSD_PROXY_TO_INT_SHIFT)) + 1ULL) >> 1;
        uint64_t result_q8 = _psd_simpson_step_q8(x_q14, 0, len - 1);

        val_q8 += ((((uint64_t)(x_q14[0] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                     (uint64_t)(x_q14[1] >> FXP_PSD_PROXY_TO_INT_SHIFT)) + 1ULL) >> 1);
        result_q8 += _psd_simpson_step_q8(x_q14, 1, len);

        val_q8 = (val_q8 + 1ULL) >> 1;
        result_q8 = (result_q8 + 1ULL) >> 1;
        return result_q8 + val_q8;
    }

    return _psd_simpson_step_q8(x_q14, 0, len);
}

static uq12_20_t _audio_psd_dominant_freq_q20(const fxp_audio_psd_view_t *view)
{
    if (!view || !view->proxy_q14 || !view->freqs_q20 || view->len <= 0) return 0;

    int16_t max_idx = 0;
    uq18_14_t max_val = view->proxy_q14[0];
    for (int16_t i = 1; i < view->len; i++) {
        if (view->proxy_q14[i] > max_val) {
            max_val = view->proxy_q14[i];
            max_idx = i;
        }
    }
    return view->freqs_q20[max_idx];
}

static uq0_16_t _audio_psd_flatness_q16(const fxp_audio_psd_view_t *view)
{
    if (!view || view->len <= 0) return 0;

    if (view->log_proxy_q11) {
        int64_t sum_logs_q11 = 0;
        for (int16_t i = 0; i < view->len; i++) {
            sum_logs_q11 += (int64_t)view->log_proxy_q11[i];
        }
        int32_t diff_q11 = _psd_round_div_s64(sum_logs_q11, view->len);
        if (diff_q11 > 0) diff_q11 = 0;
        return _psd_exp_q16_from_q11(fxp_sat_s16_from_s32(diff_q11));
    }

    if (!view->proxy_q14) return 0;

    int64_t sum_logs_q11 = 0;
    uint64_t sum_proxy_q14 = 0;

    for (int16_t i = 0; i < view->len; i++) {
        uq18_14_t x_q14 = view->proxy_q14[i];
        if (x_q14 == 0U) x_q14 = 1U;
        sum_logs_q11 += (int64_t)_psd_ln_proxy_q11(x_q14);
        sum_proxy_q14 += (uint64_t)x_q14;
    }

    if (sum_proxy_q14 == 0ULL) return 0;

    int32_t mean_log_q11 = _psd_round_div_s64(sum_logs_q11, view->len);
    uq18_14_t mean_proxy_q14 = fxp_sat_u32_from_u64((uint64_t)_psd_round_div_u64(sum_proxy_q14, (uint32_t)view->len));
    if (mean_proxy_q14 == 0U) mean_proxy_q14 = 1U;

    q5_11_t log_mean_q11 = _psd_ln_proxy_q11(mean_proxy_q14);
    int32_t diff_q11 = mean_log_q11 - (int32_t)log_mean_q11;
    if (diff_q11 > 0) diff_q11 = 0;
    return _psd_exp_q16_from_q11(fxp_sat_s16_from_s32(diff_q11));
}

static void _audio_psd_bandpowers_q16(const fxp_audio_psd_view_t *view,
                                      const int8_t *psd_selector,
                                      uq0_16_t *band_powers_q16)
{
    if (!band_powers_q16) return;
    for (int8_t i = 0; i < N_PSD; i++) band_powers_q16[i] = 0;

    if (!view || !view->proxy_q14 || !view->freqs_q20 || !psd_selector || view->len <= 2) return;

    uint64_t total_power_q8 = _psd_simpson_q8(view->proxy_q14, view->len);
    if (total_power_q8 == 0ULL) return;

    for (int8_t i = 0; i < N_PSD; i++) {
        if (!psd_selector[i]) continue;

        uq12_20_t band_start_q20 = (uq12_20_t)((uint32_t)psd_bands[i].start << FXP_FRAC_AUDIO_FFT_FREQUENCIES);
        uq12_20_t band_end_q20 = (uq12_20_t)((uint32_t)psd_bands[i].end << FXP_FRAC_AUDIO_FFT_FREQUENCIES);

        int16_t start_idx = 0;
        int16_t n_bins = 0;
        int found = 0;

        for (int16_t j = 0; j < view->len; j++) {
            uq12_20_t f_q20 = view->freqs_q20[j];
            if (!found && f_q20 >= band_start_q20) {
                start_idx = j;
                found = 1;
            }
            if (found && f_q20 <= band_end_q20) {
                n_bins++;
            } else if (found) {
                break;
            }
        }

        if (!found || n_bins <= 1) {
            band_powers_q16[i] = 0;
            continue;
        }

        uint64_t band_power_q8 = _psd_simpson_q8(&view->proxy_q14[start_idx], n_bins);
        uint64_t ratio_q16 = ((band_power_q8 << 16) + (total_power_q8 >> 1)) / total_power_q8;
        band_powers_q16[i] = fxp_sat_u16_from_u32(fxp_sat_u32_from_u64(ratio_q16));
    }
}

/* Natural logarithm on unsigned integer input, result in Q11. */
static int32_t _psd_ln_u64_q11(uint64_t x)
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

    int64_t ln_x_q24 = (int64_t)msb * (int64_t)FXP_PSD_LN2_Q24 + (int64_t)y;
    return (int32_t)((ln_x_q24 + (1LL << 12)) >> 13);
}

static void _fxp_audio_psd_write_features_q16(const int8_t *features_selector,
                                              const fxp_audio_psd_view_t *view,
                                              fxp_q16_t *feats_q16)
{
    int need_flatness = features_selector[SPECTRAL_FLATNESS];
    int need_dom_freq = features_selector[DOMINANT_FREQUENCY];
    int need_bandpowers = 0;
    for (int8_t i = 0; i < N_PSD; i++) {
        if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
            need_bandpowers = 1;
            break;
        }
    }

    if (!need_flatness && !need_dom_freq && !need_bandpowers) return;

    if (need_dom_freq) {
        uq12_20_t dom_q20 = _audio_psd_dominant_freq_q20(view);
        feats_q16[DOMINANT_FREQUENCY] = fxp_q16_from_u32(dom_q20, FXP_FRAC_AUDIO_FFT_FREQUENCIES);
    }

    if (need_flatness) {
        uq0_16_t flatness_q16 = _audio_psd_flatness_q16(view);
        feats_q16[SPECTRAL_FLATNESS] = fxp_q16_from_u32(flatness_q16, FXP_FRAC_AUDIO_PSD_FLATNESS);
    }

    if (need_bandpowers) {
        uq0_16_t band_powers_q16[N_PSD] = {0};
        _audio_psd_bandpowers_q16(view, &features_selector[POWER_SPECTRAL_DENSITY], band_powers_q16);
        for (int8_t i = 0; i < N_PSD; i++) {
            if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
                feats_q16[POWER_SPECTRAL_DENSITY + i] =
                    fxp_q16_from_u32(band_powers_q16[i], FXP_FRAC_AUDIO_PSD_BANDPOWER);
            }
        }
    }
}

void fxp_audio_periodogram_features_from_q14(const int8_t *features_selector,
                                             const int16_t *sig_q14,
                                             int16_t sig_len,
                                             int16_t fs,
                                             fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig_q14 || !feats_q16 || sig_len <= 0 || fs <= 0) return;
    if (sig_len < NPERSEG) return;

    int need_flatness = features_selector[SPECTRAL_FLATNESS];
    int need_dom_freq = features_selector[DOMINANT_FREQUENCY];
    int need_bandpowers = 0;
    for (int8_t i = 0; i < N_PSD; i++) {
        if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
            need_bandpowers = 1;
            break;
        }
    }
    if (!need_flatness && !need_dom_freq && !need_bandpowers) return;

    const int16_t psd_len = (int16_t)((NPERSEG / 2) + 1);
    const int16_t hop = (int16_t)(NPERSEG - NOVERLAP);
    if (hop <= 0) return;

    int16_t steps = (int16_t)((sig_len - NOVERLAP) / hop);
    if (steps <= 0) steps = 1;

    fxp_psd_sig_t *sig_q = (fxp_psd_sig_t *)malloc((size_t)sig_len * sizeof(fxp_psd_sig_t));
    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)NPERSEG * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)psd_len * sizeof(kiss_fft_cpx));
    uint64_t *acc_power = (uint64_t *)malloc((size_t)psd_len * sizeof(uint64_t));
    uq18_14_t *proxy_q14 = (uq18_14_t *)malloc((size_t)psd_len * sizeof(uq18_14_t));
    int32_t *log_proxy_q11 = (int32_t *)malloc((size_t)psd_len * sizeof(int32_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(NPERSEG, 0, 0, 0);

    if (!sig_q || !timedata || !cx_out || !acc_power || !proxy_q14 || !log_proxy_q11 || !freqs_q20 || !cfg) {
        free(sig_q);
        free(timedata);
        free(cx_out);
        free(acc_power);
        free(proxy_q14);
        free(log_proxy_q11);
        free(freqs_q20);
        free(cfg);
        return;
    }

    for (int16_t i = 0; i < sig_len; i++) {
        sig_q[i] = _psd_to_sig_q(sig_q14[i]);
    }
    memset(acc_power, 0, (size_t)psd_len * sizeof(uint64_t));

    int16_t start = 0;
    for (int16_t step = 0; step < steps; step++) {
        if ((int32_t)start + NPERSEG > sig_len) break;

        int64_t sum_qsig = 0;
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_qsig += (int64_t)sig_q[start + i];
        }
        int32_t mean_qsig = _psd_round_div_s64(sum_qsig, NPERSEG);

        for (int16_t i = 0; i < NPERSEG; i++) {
            int32_t centered_qsig = (int32_t)sig_q[start + i] - mean_qsig;
            int64_t prod_qsig_w = (int64_t)centered_qsig * (int64_t)fxp_hann_window_q15[i];
            int32_t win_qsig;
            if (prod_qsig_w >= 0) {
                win_qsig = (int32_t)((prod_qsig_w + (1LL << (FXP_HANN_FRAC_BITS - 1))) >> FXP_HANN_FRAC_BITS);
            } else {
                win_qsig = -(int32_t)(((-prod_qsig_w) + (1LL << (FXP_HANN_FRAC_BITS - 1))) >> FXP_HANN_FRAC_BITS);
            }

#if (FIXED_POINT == 32)
            timedata[i] = (kiss_fft_scalar)win_qsig;
#else
            timedata[i] = (kiss_fft_scalar)fxp_sat_s16_from_s32(win_qsig);
#endif
        }

        kiss_fftr(cfg, timedata, cx_out);

        for (int16_t i = 0; i < psd_len; i++) {
            int64_t re = (int64_t)cx_out[i].r;
            int64_t im = (int64_t)cx_out[i].i;
            uint64_t re_sq = (uint64_t)(re * re);
            uint64_t im_sq = (uint64_t)(im * im);
            uint64_t power = re_sq + im_sq;
            if (i != 0 && i != (psd_len - 1)) {
                power = (power > (UINT64_MAX >> 1U)) ? UINT64_MAX : (power << 1U);
            }
            if (UINT64_MAX - acc_power[i] < power) {
                acc_power[i] = UINT64_MAX;
            } else {
                acc_power[i] += power;
            }
        }

        start = (int16_t)(start + hop);
    }

    uint64_t max_power = 0ULL;
    uint64_t total_power = 0ULL;
    for (int16_t i = 0; i < psd_len; i++) {
        if (acc_power[i] > max_power) max_power = acc_power[i];
        if (UINT64_MAX - total_power < acc_power[i]) {
            total_power = UINT64_MAX;
        } else {
            total_power += acc_power[i];
        }
    }
    if (max_power == 0ULL) max_power = 1ULL;
    if (total_power == 0ULL) total_power = 1ULL;

    uint64_t mean_power = (total_power + ((uint64_t)psd_len >> 1U)) / (uint64_t)psd_len;
    if (mean_power == 0ULL) mean_power = 1ULL;
    int32_t ln_mean_power_q11 = _psd_ln_u64_q11(mean_power);

    uint64_t scale_num = (uint64_t)psd_len << FXP_FRAC_AUDIO_PSD_PROXY;
    uint32_t acc_msb = (max_power > 0ULL) ? (63U - (uint32_t)__builtin_clzll(max_power)) : 0U;
    uint32_t scale_msb = (scale_num > 0ULL) ? (63U - (uint32_t)__builtin_clzll(scale_num)) : 0U;
    uint32_t norm_shift = 0U;
    if (acc_msb + scale_msb + 2U > 63U) {
        norm_shift = (acc_msb + scale_msb + 2U) - 63U;
    }
    uint64_t total_scaled = _psd_round_shift_u64(total_power, norm_shift);
    if (total_scaled == 0ULL) total_scaled = 1ULL;

    for (int16_t i = 0; i < psd_len; i++) {
        uint64_t p_scaled = _psd_round_shift_u64(acc_power[i], norm_shift);
        uint64_t num = p_scaled * scale_num;
        uint64_t proxy64 = (num + (total_scaled >> 1U)) / total_scaled;
        if (proxy64 == 0ULL) proxy64 = 1ULL;
        proxy_q14[i] = fxp_sat_u32_from_u64(proxy64);

        int32_t ln_acc_q11 = _psd_ln_u64_q11((acc_power[i] == 0ULL) ? 1ULL : acc_power[i]);
        log_proxy_q11[i] = ln_acc_q11 - ln_mean_power_q11;

        uint64_t freq_q20 = (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)NPERSEG;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    fxp_audio_psd_view_t view = {
        .proxy_q14 = proxy_q14,
        .log_proxy_q11 = log_proxy_q11,
        .freqs_q20 = freqs_q20,
        .len = psd_len,
    };

    _fxp_audio_psd_write_features_q16(features_selector, &view, feats_q16);

    free(sig_q);
    free(timedata);
    free(cx_out);
    free(acc_power);
    free(proxy_q14);
    free(log_proxy_q11);
    free(freqs_q20);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Mel feature block                                                         */
/* -------------------------------------------------------------------------- */

#define FXP_MEL_WIN_FRAC 15
#define FXP_MEL_BASIS_FRAC 15
#define FXP_MEL_STATS_FRAC 11
#define FXP_MEL_PROB_FRAC 24

#define FXP_MEL_DB_PER_LN_Q20 ((int32_t)4553913)
#define FXP_MEL_20DB_PER_LN_Q20 ((int32_t)9107826)
#define FXP_MEL_LN2_Q24 ((int32_t)11629080)
#define FXP_MEL_LN2_Q11 ((int32_t)((FXP_MEL_LN2_Q24 + (1 << 12)) >> 13))
#define FXP_MEL_LN_Q11_SCALE_P ((int32_t)(FXP_MEL_PROB_FRAC * FXP_MEL_LN2_Q11))
#define FXP_MEL_TOP_DB_Q11 ((int32_t)163840)
#define FXP_MEL_DB_PER_SHIFT_Q11 ((int32_t)12330)
#define FXP_MEL_POWER_MSB_TARGET 46U
#define FXP_MEL_ENT_ALIGN_FRAC 8U

#if (FIXED_POINT == 32)
#define FXP_MEL_INPUT_FRAC 31
#define FXP_MEL_SCALAR_MAX_I INT32_MAX
typedef int32_t fxp_mel_sig_t;
#else
#define FXP_MEL_INPUT_FRAC 15
#define FXP_MEL_SCALAR_MAX_I INT16_MAX
typedef int16_t fxp_mel_sig_t;
#endif

static uint32_t _kiss_ref_abs = (uint32_t)N_FFT;
static int _kiss_ref_initialized = 0;

static int _mel_any_required(const int8_t *features_selector)
{
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT; i < ZERO_CROSSING_RATE; i++) {
        if (features_selector[i]) return 1;
    }
    return 0;
}

static inline int32_t _mel_round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static inline int64_t _mel_round_div_s64_to_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (num + (den / 2)) / den;
    return -(((-num) + (den / 2)) / den);
}

static inline uint64_t _mel_round_shift_u64(uint64_t v, uint32_t shift)
{
    if (shift == 0U) return v;
    if (shift >= 64U) return 0ULL;
    return (v + (1ULL << (shift - 1U))) >> shift;
}

static inline uint32_t _mel_uq_div_u64_q(uint64_t num, uint64_t den, uint8_t frac_bits)
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

        if (r >= (den - r)) {
            r = r - (den - r);
            q |= 1ULL;
        } else {
            r += r;
        }
    }

    if (r >= (den - r) && q < UINT64_MAX) {
        q += 1ULL;
    }

    if (q > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)q;
}

static inline uint32_t _mel_pick_power_shift(uint64_t max_power)
{
    if (max_power == 0ULL) return 0U;
    uint32_t msb = 63U - (uint32_t)__builtin_clzll(max_power);
    if (msb <= FXP_MEL_POWER_MSB_TARGET) return 0U;
    return msb - FXP_MEL_POWER_MSB_TARGET;
}

static inline uint8_t _mel_ceil_log2_u16(uint16_t v)
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

static inline uint64_t _mel_entropy_align_term(uint64_t value, uint8_t frame_shift, uint8_t max_frame_shift)
{
    if (value == 0ULL) return 0ULL;
    uint8_t dshift = (uint8_t)(max_frame_shift - frame_shift);
    if (dshift <= FXP_MEL_ENT_ALIGN_FRAC) {
        uint32_t lshift = (uint32_t)(FXP_MEL_ENT_ALIGN_FRAC - dshift);
        if (lshift >= 64U) return 0ULL;
        if (value > (UINT64_MAX >> lshift)) return UINT64_MAX;
        return value << lshift;
    }
    return _mel_round_shift_u64(value, (uint32_t)(dshift - FXP_MEL_ENT_ALIGN_FRAC));
}

static inline fxp_mel_sig_t _mel_from_input_q14(int16_t x_q14)
{
    int32_t q = ((int32_t)x_q14) << (FXP_MEL_INPUT_FRAC - FXP_FRAC_AUDIO_INPUT);
#if (FIXED_POINT == 32)
    return (fxp_mel_sig_t)q;
#else
    return (fxp_mel_sig_t)fxp_sat_s16_from_s32(q);
#endif
}

static void _mel_ensure_kiss_ref(void)
{
    if (_kiss_ref_initialized) return;

    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);
    kiss_fft_scalar *x = (kiss_fft_scalar *)calloc((size_t)N_FFT, sizeof(kiss_fft_scalar));
    kiss_fft_cpx *y = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));

    if (!cfg || !x || !y) {
        free(cfg);
        free(x);
        free(y);
        _kiss_ref_abs = (uint32_t)N_FFT;
        _kiss_ref_initialized = 1;
        return;
    }

    x[0] = (kiss_fft_scalar)FXP_MEL_SCALAR_MAX_I;
    kiss_fftr(cfg, x, y);

    int32_t ref = y[1].r;
    if (ref == 0) ref = y[0].r;
    if (ref < 0) ref = -ref;
    if (ref == 0) ref = 1;
    _kiss_ref_abs = (uint32_t)ref;
    _kiss_ref_initialized = 1;

    free(cfg);
    free(x);
    free(y);
}

static int32_t _mel_ln_u64_q11(uint64_t x)
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

    int64_t ln_x_q24 = (int64_t)msb * (int64_t)FXP_MEL_LN2_Q24 + (int64_t)y;
    return (int32_t)((ln_x_q24 + (1LL << 12)) >> 13);
}

static int32_t _mel_db_from_power_q11(uint64_t p_scaled, int32_t db_offset_q11)
{
    int32_t ln_q11 = _mel_ln_u64_q11((p_scaled == 0ULL) ? 1ULL : p_scaled);
    int32_t db_q11 = (int32_t)((((int64_t)ln_q11 * (int64_t)FXP_MEL_DB_PER_LN_Q20) + (1LL << 19)) >> 20);
    return db_q11 + db_offset_q11;
}

static int32_t _mel_entropy_row_q11(const uint64_t *row_power,
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
    uint8_t sum_bits = _mel_ceil_log2_u16((uint16_t)n_frames);
    int32_t pre_shift_i = (int32_t)row_msb + (int32_t)FXP_MEL_ENT_ALIGN_FRAC + (int32_t)sum_bits - 62;
    uint32_t pre_shift = (pre_shift_i > 0) ? (uint32_t)pre_shift_i : 0U;

    uint64_t row_sum = 0ULL;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t v = _mel_round_shift_u64(row_power[t], pre_shift);
        uint64_t term = _mel_entropy_align_term(v, frame_shift[t], row_max_shift);
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
        uint64_t v = _mel_round_shift_u64(row_power[t], pre_shift);
        uint64_t term = _mel_entropy_align_term(v, frame_shift[t], row_max_shift);
        if (term == 0ULL) continue;

        uint32_t p_qp = _mel_uq_div_u64_q(term, row_sum, FXP_MEL_PROB_FRAC);
        if (p_qp == 0U) continue;

        int32_t ln_p_q11 = _mel_ln_u64_q11((uint64_t)p_qp) - FXP_MEL_LN_Q11_SCALE_P;
        if (ln_p_q11 > 0) ln_p_q11 = 0;

        int64_t contrib_q11 = -((((int64_t)p_qp * (int64_t)ln_p_q11) + (1LL << (FXP_MEL_PROB_FRAC - 1))) >> FXP_MEL_PROB_FRAC);
        entropy_q11 += contrib_q11;
    }

    return fxp_sat_s32_from_s64(entropy_q11);
}

void fxp_audio_mel_features_from_q14(const int8_t *features_selector,
                                     const int16_t *sig_q14,
                                     int16_t len,
                                     fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig_q14 || !feats_q16 || len <= 0) return;
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

    _mel_ensure_kiss_ref();

    int32_t ln_ref_q11 = _mel_ln_u64_q11((uint64_t)_kiss_ref_abs);
    int32_t db_offset_base_q11 = -((int32_t)((((int64_t)ln_ref_q11 * (int64_t)FXP_MEL_20DB_PER_LN_Q20) + (1LL << 19)) >> 20));

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = _mel_from_input_q14(sig_q14[i]);
    }

    for (int16_t i = 0; i < PAD_LEN; i++) {
        padded_q[i] = sig_q[PAD_LEN - i];
        padded_q[PAD_LEN + len + i] = sig_q[len - 2 - i];
    }
    for (int16_t i = 0; i < len; i++) {
        padded_q[PAD_LEN + i] = sig_q[i];
    }

    int32_t max_db_q11 = INT32_MIN;
    for (int16_t f = 0; f < n_frames; f++) {
        int32_t frame_start = (int32_t)f * HOP_LEN;

        for (int16_t n = 0; n < N_FFT; n++) {
            int32_t centered_q = (int32_t)padded_q[frame_start + n];
            int64_t prod = (int64_t)centered_q * (int64_t)fxp_mfcc_hann_q15[n];
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
        uint8_t cur_shift = (uint8_t)_mel_pick_power_shift(max_power);
        frame_shift[f] = cur_shift;
        int32_t frame_db_offset_q11 = db_offset_base_q11 + (int32_t)cur_shift * FXP_MEL_DB_PER_SHIFT_Q11;

        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            frame_power[k] = _mel_round_shift_u64(frame_power[k], cur_shift);
        }

        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uint64_t sum = 0ULL;
            for (int16_t k = start; k <= end; k++) {
                uint16_t w_q15 = fxp_mel_basis_q15[mel_idx][k - start];
                uint64_t term = ((frame_power[k] * (uint64_t)w_q15) + (1ULL << (FXP_MEL_BASIS_FRAC - 1))) >> FXP_MEL_BASIS_FRAC;
                if (UINT64_MAX - sum < term) {
                    sum = UINT64_MAX;
                } else {
                    sum += term;
                }
            }

            mel_power[(size_t)m * (size_t)n_frames + (size_t)f] = sum;
            mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f] = _mel_db_from_power_q11(sum, frame_db_offset_q11);
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

        int32_t mean_q11 = _mel_round_div_s64(sum_db_q11, n_frames);

        int64_t sum_sq_q22 = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            int32_t d = mel_db_q11[(size_t)m * (size_t)n_frames + (size_t)f] - mean_q11;
            sum_sq_q22 += (int64_t)d * (int64_t)d;
        }
        int64_t var_q22 = _mel_round_div_s64_to_s64(sum_sq_q22, n_frames);
        if (var_q22 < 0) var_q22 = 0;
        int32_t std_q11 = fxp_sat_s32_from_s64((int64_t)fxp_sqrt64((uint64_t)var_q22));

        int32_t ent_q11 = _mel_entropy_row_q11(&mel_power[(size_t)m * (size_t)n_frames],
                                               frame_shift,
                                               n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats_q16[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] = fxp_q16_from_s32(mean_q11, FXP_MEL_STATS_FRAC);
        feats_q16[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] = fxp_q16_from_s32(std_q11, FXP_MEL_STATS_FRAC);
        feats_q16[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] = fxp_q16_from_s32(row_max_q11, FXP_MEL_STATS_FRAC);
        feats_q16[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] = fxp_q16_from_s32(ent_q11, FXP_MEL_STATS_FRAC);
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
