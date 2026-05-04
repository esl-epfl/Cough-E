#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <audio/audio_pipeline_fxp.h>

#include <kiss_fftr.h>
#include <mfcc_module.h>
#include <welch_psd.h>

#include <audio/audio_tables_q15.h>
#include <core/fxp_log_exp.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

/* -------------------------------------------------------------------------- */
/*  FFT feature kernels + block                                               */
/* -------------------------------------------------------------------------- */

typedef struct {
    const uq12_20_t *mags_q20;
    const uq12_20_t *freqs_q20;
    int16_t len;
    uq15_17_t sum_mags_q17;
} audio_fft_view_t;

typedef struct {
    const uq21_11_t *proxy_q11;
    const uq12_20_t *freqs_q20;
    int16_t len;
} audio_psd_view_t;

#define FXP_FFT_ROLLOFF_95_Q16 ((uint32_t)62259U)

#if (FIXED_POINT == 32)
#define FXP_FFT_INPUT_FRAC 30
#else
#define FXP_FFT_INPUT_FRAC FXP_FRAC_AUDIO_INPUT
#endif


static inline q13_19_t _dev(uq12_20_t freq_q20, uq11_21_t centroid_q21)
{
    uint32_t freq_q19 = freq_q20 >> 1;
    uint32_t cent_q19 = centroid_q21 >> 2;
    return (q13_19_t)((int32_t)freq_q19 - (int32_t)cent_q19);
}

static uq12_20_t _rolloff(const audio_fft_view_t *view)
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

static inline int32_t _round_shift_s64_to_s32(int64_t v, uint32_t shift)
{
    if (shift == 0U) return fxp_sat_s32_from_s64(v);
    if (shift >= 63U) return (v < 0) ? -1 : 0;
    if (v >= 0) return fxp_sat_s32_from_s64((v + (1LL << (shift - 1U))) >> shift);
    return -fxp_sat_s32_from_s64(((-v) + (1LL << (shift - 1U))) >> shift);
}

static inline uint64_t _round_shift_u64_local(uint64_t v, uint32_t shift)
{
    return fxp_round_shift_u64(v, shift);
}

static uq11_21_t _centroid(const audio_fft_view_t *view)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0) return 0;

    uint64_t accum_q38 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        uint64_t prod_q24_40 = (uint64_t)view->freqs_q20[i] * (uint64_t)view->mags_q20[i];
        accum_q38 += _round_shift_u64_local(prod_q24_40, 2U);
    }

    uint64_t result_q21 = (accum_q38 + ((uint64_t)view->sum_mags_q17 >> 1U)) /
                          (uint64_t)view->sum_mags_q17;
    return (uq11_21_t)result_q21;
}

static uq11_5_t _spread(const audio_fft_view_t *view, uq11_21_t centroid_q21)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0) return 0;

    uint64_t accum_q37_27 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        q13_19_t dev_q19 = _dev(view->freqs_q20[i], centroid_q21);
        uint32_t dev2_q25_7 = (uint32_t)(((uint64_t)((int64_t)dev_q19 * (int64_t)dev_q19)) >> 31);
        accum_q37_27 += (uint64_t)dev2_q25_7 * (uint64_t)view->mags_q20[i];
    }

    uint64_t mean_q22_10 = (accum_q37_27 + ((uint64_t)view->sum_mags_q17 >> 1U)) /
                           (uint64_t)view->sum_mags_q17;
    return (uq11_5_t)fxp_sqrt32(fxp_sat_u32_from_u64(mean_q22_10));
}

static uq17_15_t _kurtosis(const audio_fft_view_t *view,
                           uq11_21_t centroid_q21,
                           uq11_5_t spread_q5)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0) return 0;
    if (view->sum_mags_q17 == 0 || spread_q5 == 0) return 0;

    uint32_t inv_spread_q20 = ((1U << 25) + ((uint32_t)spread_q5 >> 1)) /
                              (uint32_t)spread_q5;
    uint64_t accum_q32 = 0;
    for (int16_t i = 0; i < view->len; i++) {
        q13_19_t dev_q19 = _dev(view->freqs_q20[i], centroid_q21);
        int64_t norm_q39 = (int64_t)dev_q19 * (int64_t)inv_spread_q20;
        int16_t norm_q11 = fxp_sat_s16_from_s32(_round_shift_s64_to_s32(norm_q39, 28U));
        uint32_t abs_norm_q11 = (norm_q11 < 0) ? (uint32_t)(-norm_q11)
                                               : (uint32_t)norm_q11;
        uint32_t norm2_q22 = (uint32_t)((uint64_t)abs_norm_q11 * (uint64_t)abs_norm_q11);
        uint32_t norm4_q12 = (uint32_t)(((uint64_t)norm2_q22 * norm2_q22 + (1ULL << 31)) >> 32);
        accum_q32 += (uint64_t)norm4_q12 * (uint64_t)view->mags_q20[i];
    }

    uint32_t kurt_q15 = (uint32_t)((accum_q32 + ((uint64_t)view->sum_mags_q17 >> 1U)) /
                                   (uint64_t)view->sum_mags_q17);
    return (uq17_15_t)kurt_q15;
}

static void _write_fft_features(const int8_t *features_selector,
                                const audio_fft_view_t *view,
                                fxp_feat_t *feats)
{
    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];
    if (need_rolloff) {
        uq12_20_t rolloff_q20 = _rolloff(view);
        feats[SPECTRAL_ROLLOFF] = (fxp_feat_t)rolloff_q20;
    }

    uq11_21_t centroid_q21 = 0;
    if (need_centroid) {
        centroid_q21 = _centroid(view);
        if (features_selector[SPECTRAL_CENTROID]) {
            feats[SPECTRAL_CENTROID] = (fxp_feat_t)centroid_q21;
        }
    }

    uq11_5_t spread_q5 = 0;
    if (need_spread) {
        spread_q5 = _spread(view, centroid_q21);
        if (features_selector[SPECTRAL_SPREAD]) {
            feats[SPECTRAL_SPREAD] = (fxp_feat_t)spread_q5;
        }
    }

    if (need_kurt) {
        uq17_15_t kurt_q15 = _kurtosis(view, centroid_q21, spread_q5);
        feats[SPECTRAL_KURTOSIS] = (fxp_feat_t)kurt_q15;
    }
}

static inline kiss_fft_scalar _fft_from_input_q14(int16_t x_q14)
{
#if (FIXED_POINT == 32)
    int64_t q = (int64_t)x_q14 << (FXP_FFT_INPUT_FRAC - FXP_FRAC_AUDIO_INPUT);
    return (kiss_fft_scalar)fxp_sat_s32_from_s64(q);
#else
    return (kiss_fft_scalar)x_q14;
#endif
}

static inline q12_20_t _fft_reim_q20(kiss_fft_scalar x)
{
#if (FXP_FFT_INPUT_FRAC > FXP_FRAC_AUDIO_FFT_RE_IM)
    return (q12_20_t)_round_shift_s64_to_s32((int64_t)x,
                                             FXP_FFT_INPUT_FRAC - FXP_FRAC_AUDIO_FFT_RE_IM);
#else
    return fxp_sat_s32_from_s64((int64_t)x << (FXP_FRAC_AUDIO_FFT_RE_IM - FXP_FFT_INPUT_FRAC));
#endif
}

#if defined(FXP_STAGE_PROBES)
int audio_fft_stage_probe(const int16_t *sig_q14,
                          int16_t len,
                          int16_t fs,
                          uq12_20_t *mags_q20,
                          uq12_20_t *freqs_q20,
                          uq15_17_t *sum_mags_q17)
{
    if (!sig_q14 || !mags_q20 || !freqs_q20 || !sum_mags_q17 || len <= 0 || fs <= 0) return 0;

    int16_t fft_len = (int16_t)((len / 2) + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    if (!cfg) return 0;

    kiss_fft_scalar *sig_q = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));

    if (!sig_q || !cx_out) {
        free(sig_q);
        free(cx_out);
        free(cfg);
        return 0;
    }

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = _fft_from_input_q14(sig_q14[i]);
    }

    kiss_fftr(cfg, sig_q, cx_out);

    uint32_t sum_q17 = 0U;
    for (int16_t i = 0; i < fft_len; i++) {
        q12_20_t re_q20 = _fft_reim_q20(cx_out[i].r);
        q12_20_t im_q20 = _fft_reim_q20(cx_out[i].i);
        uint64_t re_sq = (uint64_t)((int64_t)re_q20 * (int64_t)re_q20);
        uint64_t im_sq = (uint64_t)((int64_t)im_q20 * (int64_t)im_q20);
        mags_q20[i] = (uq12_20_t)fxp_sqrt64(re_sq + im_sq);
        sum_q17 += (mags_q20[i] >> 3);

        uint64_t freq_q20 =
            (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)len;
        freqs_q20[i] = (uq12_20_t)freq_q20;
    }

    *sum_mags_q17 = (uq15_17_t)sum_q17;

    free(sig_q);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

void audio_fft_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t len,
                        int16_t fs,
                        fxp_feat_t *feats)
{
    if (!features_selector || !sig_q14 || !feats || len <= 0 || fs <= 0) return;

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

    if (!sig_q || !cx_out || !mags_q20 || !freqs_q20) {
        free(sig_q);
        free(cx_out);
        free(mags_q20);
        free(freqs_q20);
        free(cfg);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = _fft_from_input_q14(sig_q14[i]);
    }

    kiss_fftr(cfg, sig_q, cx_out);

    uint32_t sum_q17 = 0U;
    for (int16_t i = 0; i < fft_len; i++) {
        q12_20_t re_q20 = _fft_reim_q20(cx_out[i].r);
        q12_20_t im_q20 = _fft_reim_q20(cx_out[i].i);
        uint64_t re_sq = (uint64_t)((int64_t)re_q20 * (int64_t)re_q20);
        uint64_t im_sq = (uint64_t)((int64_t)im_q20 * (int64_t)im_q20);
        mags_q20[i] = (uq12_20_t)fxp_sqrt64(re_sq + im_sq);
        sum_q17 += (mags_q20[i] >> 3);

        uint64_t freq_q20 =
            (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)len;
        freqs_q20[i] = (uq12_20_t)freq_q20;
    }

    audio_fft_view_t view = {
        .mags_q20 = mags_q20,
        .freqs_q20 = freqs_q20,
        .len = fft_len,
        .sum_mags_q17 = sum_q17,
    };

    _write_fft_features(features_selector, &view, feats);

    free(sig_q);
    free(cx_out);
    free(mags_q20);
    free(freqs_q20);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Periodogram kernels + block                                               */
/* -------------------------------------------------------------------------- */

#define FXP_PSD_PROXY_TO_INT_SHIFT (FXP_FRAC_AUDIO_PSD_PROXY - FXP_FRAC_AUDIO_PSD_INTEGRAL)
#define FXP_HANN_FRAC_BITS 15

typedef int16_t fxp_psd_sig_t;

static inline fxp_psd_sig_t _psd_to_sig_q(int16_t x_q14)
{
    return (fxp_psd_sig_t)x_q14;
}

/* Natural logarithm on PSD proxy input, result in Q5.11. */
static q5_11_t _psd_ln_proxy_q11(uq21_11_t x_q11)
{
    if (x_q11 == 0U) x_q11 = 1U;

    uint32_t msb = 31U - (uint32_t)__builtin_clz(x_q11);
    uint32_t base = (uint32_t)1U << msb;
    uint32_t frac_q24 = (uint32_t)((((uint64_t)(x_q11 - base)) << 24) / (uint64_t)base);

    uint32_t idx = frac_q24 >> 16;
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = frac_q24 & 0xFFFFU;

    int32_t y0 = fxp_ln_lut_q24[idx];
    int32_t y1 = fxp_ln_lut_q24[idx + 1];
    int32_t y = y0 + (int32_t)((((int64_t)(y1 - y0) * (int64_t)alpha) + (1LL << 15)) >> 16);

    int32_t exp2 = (int32_t)msb - FXP_FRAC_AUDIO_PSD_PROXY;
    int64_t ln_x_q24 = (int64_t)exp2 * (int64_t)FXP_LN2_Q24 + (int64_t)y;
    int64_t ln_x_q11 = (ln_x_q24 >= 0) ? ((ln_x_q24 + (1LL << 12)) >> 13) : -(((-ln_x_q24) + (1LL << 12)) >> 13);

    return fxp_sat_s16_from_s32((int32_t)ln_x_q11);
}

static uint32_t _psd_simpson_step_q8(const uq21_11_t *x_q11, int16_t start, int16_t end)
{
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uint32_t sum_q8 = 0;

    for (int i = 0; i < n_intervals; i++) {
        uint32_t x0_q8 = (uint32_t)(x_q11[idx] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint32_t x1_q8 = (uint32_t)(x_q11[idx + 1] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint32_t x2_q8 = (uint32_t)(x_q11[idx + 2] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        sum_q8 += x0_q8 + (x1_q8 << 2) + x2_q8;
        idx += 2;
    }

    return (sum_q8 + 1U) / 3U;
}

static uint32_t _psd_simpson_q8(const uq21_11_t *x_q11, int16_t len)
{
    if (!x_q11 || len <= 1) return 0U;

    if ((len & 1) == 0) {
        uint32_t val_q8 = (((uint32_t)(x_q11[len - 1] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                            (uint32_t)(x_q11[len - 2] >> FXP_PSD_PROXY_TO_INT_SHIFT)) + 1U) >> 1;
        uint32_t result_q8 = _psd_simpson_step_q8(x_q11, 0, len - 1);

        val_q8 += ((((uint32_t)(x_q11[0] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                     (uint32_t)(x_q11[1] >> FXP_PSD_PROXY_TO_INT_SHIFT)) + 1U) >> 1);
        result_q8 += _psd_simpson_step_q8(x_q11, 1, len);

        val_q8 = (val_q8 + 1U) >> 1;
        result_q8 = (result_q8 + 1U) >> 1;
        return result_q8 + val_q8;
    }

    return _psd_simpson_step_q8(x_q11, 0, len);
}

static uq12_20_t _dominant_freq(const audio_psd_view_t *view)
{
    if (!view || !view->proxy_q11 || !view->freqs_q20 || view->len <= 0) return 0;

    int16_t max_idx = 0;
    uq21_11_t max_val = view->proxy_q11[0];
    for (int16_t i = 1; i < view->len; i++) {
        if (view->proxy_q11[i] > max_val) {
            max_val = view->proxy_q11[i];
            max_idx = i;
        }
    }
    return view->freqs_q20[max_idx];
}

static uq0_16_t _flatness(const audio_psd_view_t *view)
{
    if (!view || view->len <= 0) return 0;

    if (!view->proxy_q11) return 0;


    int32_t sum_logs_q11 = 0;
    uint32_t sum_proxy_q11 = 0;

    for (int16_t i = 0; i < view->len; i++) {
        uq21_11_t x_q11 = view->proxy_q11[i];
        if (x_q11 == 0U) x_q11 = 1U;
        sum_logs_q11 += (int32_t)_psd_ln_proxy_q11(x_q11);
        sum_proxy_q11 += (uint32_t)x_q11;
    }

    if (sum_proxy_q11 == 0U) return 0;

    int32_t mean_log_q11 = fxp_round_div_s32(sum_logs_q11, view->len);
    uq21_11_t mean_proxy_q11 = fxp_round_div_u32(sum_proxy_q11, (uint32_t)view->len);
    if (mean_proxy_q11 == 0U) mean_proxy_q11 = 1U;

    q5_11_t log_mean_q11 = _psd_ln_proxy_q11(mean_proxy_q11);
    int32_t diff_q11 = mean_log_q11 - (int32_t)log_mean_q11;
    if (diff_q11 > 0) diff_q11 = 0;
    return fxp_exp_uq0_16_from_q11((q5_11_t)diff_q11);
}

static void _bandpowers(const audio_psd_view_t *view,
                        const int8_t *psd_selector,
                        uq0_16_t *band_powers_q16)
{
    if (!band_powers_q16) return;
    for (int8_t i = 0; i < N_PSD; i++) band_powers_q16[i] = 0;

    if (!view || !view->proxy_q11 || !view->freqs_q20 || !psd_selector || view->len <= 2) return;

    uint32_t total_power_q8 = _psd_simpson_q8(view->proxy_q11, view->len);
    if (total_power_q8 == 0U) return;

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

        uint32_t band_power_q8 = _psd_simpson_q8(&view->proxy_q11[start_idx], n_bins);
        uint32_t ratio_q16 = (uint32_t)((((uint64_t)band_power_q8 << 16) + (total_power_q8 >> 1)) /
                                        total_power_q8);
        band_powers_q16[i] = (uq0_16_t)ratio_q16;
    }
}

static void _write_psd_features(const int8_t *features_selector,
                                const audio_psd_view_t *view,
                                fxp_feat_t *feats)
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
        uq12_20_t dom_q20 = _dominant_freq(view);
        feats[DOMINANT_FREQUENCY] = (fxp_feat_t)dom_q20;
    }

    if (need_flatness) {
        uq0_16_t flatness_q16 = _flatness(view);
        feats[SPECTRAL_FLATNESS] = (fxp_feat_t)flatness_q16;
    }

    if (need_bandpowers) {
        uq0_16_t band_powers_q16[N_PSD] = {0};
        _bandpowers(view, &features_selector[POWER_SPECTRAL_DENSITY], band_powers_q16);
        for (int8_t i = 0; i < N_PSD; i++) {
            if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
                feats[POWER_SPECTRAL_DENSITY + i] = (fxp_feat_t)band_powers_q16[i];
            }
        }
    }
}

void audio_psd_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t sig_len,
                        int16_t fs,
                        fxp_feat_t *feats)
{
    if (!features_selector || !sig_q14 || !feats || sig_len <= 0 || fs <= 0) return;
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
    uint32_t *acc_power = (uint32_t *)malloc((size_t)psd_len * sizeof(uint32_t));
    uq21_11_t *proxy_q11 = (uq21_11_t *)malloc((size_t)psd_len * sizeof(uq21_11_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(NPERSEG, 0, 0, 0);

    if (!sig_q || !timedata || !cx_out || !acc_power || !proxy_q11 || !freqs_q20 || !cfg) {
        free(sig_q);
        free(timedata);
        free(cx_out);
        free(acc_power);
        free(proxy_q11);
        free(freqs_q20);
        free(cfg);
        return;
    }

    for (int16_t i = 0; i < sig_len; i++) {
        sig_q[i] = _psd_to_sig_q(sig_q14[i]);
    }
    memset(acc_power, 0, (size_t)psd_len * sizeof(uint32_t));

    int16_t start = 0;
    for (int16_t step = 0; step < steps; step++) {
        if ((int32_t)start + NPERSEG > sig_len) break;
        int32_t sum_qsig = 0;
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_qsig += (int32_t)sig_q[start + i];
        }
        int32_t mean_qsig = fxp_round_div_s32(sum_qsig, NPERSEG);

        for (int16_t i = 0; i < NPERSEG; i++) {
            int32_t centered_qsig = (int32_t)sig_q[start + i] - mean_qsig;
            int64_t prod_qsig_w = (int64_t)centered_qsig * (int64_t)fxp_hann_window_q15[i];
            int32_t win_qsig = _round_shift_s64_to_s32(prod_qsig_w, FXP_HANN_FRAC_BITS);

#if (FIXED_POINT == 32)
            timedata[i] = (kiss_fft_scalar)win_qsig;
#else
            timedata[i] = (kiss_fft_scalar)fxp_sat_s16_from_s32(win_qsig);
#endif
        }

        kiss_fftr(cfg, timedata, cx_out);

        for (int16_t i = 0; i < psd_len; i++) {
            int32_t re_q8 = _round_shift_s64_to_s32((int64_t)cx_out[i].r,
                                                    FXP_FRAC_AUDIO_INPUT - 8U);
            int32_t im_q8 = _round_shift_s64_to_s32((int64_t)cx_out[i].i,
                                                    FXP_FRAC_AUDIO_INPUT - 8U);

            re_q8 = fxp_sat_s16_from_s32(re_q8);
            im_q8 = fxp_sat_s16_from_s32(im_q8);

            uint64_t raw_mag_sq_q16 = (uint64_t)((int64_t)re_q8 * (int64_t)re_q8) +
                                      (uint64_t)((int64_t)im_q8 * (int64_t)im_q8);
            if (i != 0 && i != (psd_len - 1)) {
                raw_mag_sq_q16 <<= 1U;
            }
            uint32_t power_q11 = (uint32_t)_round_shift_u64_local(raw_mag_sq_q16, 5U);
            acc_power[i] += power_q11;
        }

        start = (int16_t)(start + hop);
    }

    for (int16_t i = 0; i < psd_len; i++) {
        proxy_q11[i] = (uq21_11_t)acc_power[i];
        uint64_t freq_q20 = (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)NPERSEG;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    audio_psd_view_t view = {
        .proxy_q11 = proxy_q11,
        .freqs_q20 = freqs_q20,
        .len = psd_len,
    };

    _write_psd_features(features_selector, &view, feats);

    free(sig_q);
    free(timedata);
    free(cx_out);
    free(acc_power);
    free(proxy_q11);
    free(freqs_q20);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Mel feature block                                                         */
/* -------------------------------------------------------------------------- */

#define FXP_MEL_WIN_FRAC 15
#define FXP_MEL_BASIS_FRAC 15
#define FXP_MEL_POWER_FRAC 28
#define FXP_MEL_ACC_FRAC 36
#define FXP_MEL_ACC_SHIFT ((FXP_MEL_POWER_FRAC + FXP_MEL_BASIS_FRAC) - FXP_MEL_ACC_FRAC)
#define FXP_MEL_ACC_EXTRA_FRAC (FXP_MEL_ACC_FRAC - FXP_MEL_POWER_FRAC)
#define FXP_MEL_STATS_FRAC 9
#define FXP_MEL_ENTROPY_FRAC 14
#define FXP_MEL_PROB_FRAC 16
#define FXP_MEL_ENTROPY_PROD_SHIFT ((FXP_MEL_PROB_FRAC + FXP_MEL_STATS_FRAC) - FXP_MEL_ENTROPY_FRAC)

#define FXP_MEL_DB_PER_LN_Q20 ((int32_t)4553913)
#define FXP_MEL_20DB_PER_LN_Q20 ((int32_t)9107826)
#define FXP_MEL_LN2_Q24 ((int32_t)11629080)
#define FXP_MEL_LN2_Q9 ((int32_t)((FXP_MEL_LN2_Q24 + (1 << 14)) >> 15))
#define FXP_MEL_LN_Q9_SCALE_P ((int32_t)(FXP_MEL_PROB_FRAC * FXP_MEL_LN2_Q9))
#define FXP_MEL_TOP_DB_Q9 ((int32_t)40960)
#define FXP_MEL_DB_PER_POWER_BIT_Q9 ((int32_t)1542)

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
            q |= 1U;
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

static inline fxp_mel_sig_t _mel_from_input_q14(int16_t x_q14)
{
    int64_t q = ((int64_t)x_q14) << (FXP_MEL_INPUT_FRAC - FXP_FRAC_AUDIO_INPUT);
#if (FIXED_POINT == 32)
    return (fxp_mel_sig_t)fxp_sat_s32_from_s64(q);
#else
    return (fxp_mel_sig_t)fxp_sat_s16_from_s32((int32_t)q);
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

static int16_t _mel_db_from_power_q9(uint64_t p_scaled, int32_t db_offset_q9)
{
    int16_t ln_q9 = fxp_ln_u64_q9((p_scaled == 0ULL) ? 1ULL : p_scaled);
    int32_t db_q9 = (int32_t)((((int64_t)ln_q9 * (int64_t)FXP_MEL_DB_PER_LN_Q20) + (1LL << 19)) >> 20);
    return fxp_sat_s16_from_s32(db_q9 + db_offset_q9);
}

static uint16_t _mel_entropy_row_q14(const uint64_t *row_power,
                                    int16_t n_frames)
{
    if (!row_power || n_frames <= 0) return 0;

    uint64_t row_max = 0ULL;
    for (int16_t t = 0; t < n_frames; t++) {
        if (row_power[t] > row_max) row_max = row_power[t];
    }
    if (row_max == 0ULL) return 0;

    uint32_t row_msb = 63U - (uint32_t)__builtin_clzll(row_max);
    uint8_t sum_bits = _mel_ceil_log2_u16((uint16_t)n_frames);
    int32_t pre_shift_i = (int32_t)row_msb + (int32_t)sum_bits - 62;
    uint32_t pre_shift = (pre_shift_i > 0) ? (uint32_t)pre_shift_i : 0U;

    uint64_t row_sum = 0ULL;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t term = fxp_round_shift_u64(row_power[t], pre_shift);
        if (term == 0ULL) continue;
        if (UINT64_MAX - row_sum < term) {
            row_sum = UINT64_MAX;
        } else {
            row_sum += term;
        }
    }
    if (row_sum == 0ULL) return 0;

    int32_t entropy_q14 = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t term = fxp_round_shift_u64(row_power[t], pre_shift);
        if (term == 0ULL) continue;

        uint16_t p_q16 = fxp_sat_u16_from_u32(_mel_uq_div_u64_q(term, row_sum, FXP_MEL_PROB_FRAC));
        if (p_q16 == 0U) continue;

        int32_t ln_p_q9 = (int32_t)fxp_ln_u64_q9((uint64_t)p_q16) - FXP_MEL_LN_Q9_SCALE_P;
        if (ln_p_q9 > 0) ln_p_q9 = 0;

        int32_t prod_q25 = (int32_t)p_q16 * (int32_t)(-ln_p_q9);
        entropy_q14 += (prod_q25 + (1LL << (FXP_MEL_ENTROPY_PROD_SHIFT - 1))) >> FXP_MEL_ENTROPY_PROD_SHIFT;
    }

    if (entropy_q14 <= 0) return 0U;
    if (entropy_q14 > UINT16_MAX) return UINT16_MAX;
    return (uint16_t)entropy_q14;
}

void audio_mel_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t len,
                        fxp_feat_t *feats)
{
    if (!features_selector || !sig_q14 || !feats || len <= 0) return;
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
    int16_t *mel_db_q9 = (int16_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int16_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!sig_q || !padded_q || !timedata || !cx_out || !frame_power || !mel_power || !mel_db_q9 || !cfg) {
        free(sig_q);
        free(padded_q);
        free(timedata);
        free(cx_out);
        free(frame_power);
        free(mel_power);
        free(mel_db_q9);
        free(cfg);
        return;
    }

    _mel_ensure_kiss_ref();

    int32_t ln_ref_q9 = fxp_ln_u64_q9((uint64_t)_kiss_ref_abs);
    int32_t db_offset_base_q9 =
        -((int32_t)((((int64_t)ln_ref_q9 * (int64_t)FXP_MEL_20DB_PER_LN_Q20) + (1LL << 19)) >> 20))
        - ((int32_t)FXP_MEL_ACC_EXTRA_FRAC * FXP_MEL_DB_PER_POWER_BIT_Q9);

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

    int16_t max_db_q9 = INT16_MIN;
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

        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            int64_t re = (int64_t)cx_out[k].r;
            int64_t im = (int64_t)cx_out[k].i;
            uint64_t p = (uint64_t)(re * re) + (uint64_t)(im * im);
            frame_power[k] = p;
        }

        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uint64_t sum = 0ULL;
            for (int16_t k = start; k <= end; k++) {
                uint16_t w_q15 = fxp_mel_basis_q15[mel_idx][k - start];
                uint64_t term = fxp_round_shift_u64(frame_power[k] * (uint64_t)w_q15,
                                                    FXP_MEL_ACC_SHIFT);
                if (UINT64_MAX - sum < term) {
                    sum = UINT64_MAX;
                } else {
                    sum += term;
                }
            }

            mel_power[(size_t)m * (size_t)n_frames + (size_t)f] = sum;
            mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] = _mel_db_from_power_q9(sum, db_offset_base_q9);
            if (mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] > max_db_q9) {
                max_db_q9 = mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f];
            }
        }
    }

    int16_t clip_floor_q9 = fxp_sat_s16_from_s32((int32_t)max_db_q9 - FXP_MEL_TOP_DB_Q9);

    for (int16_t m = 0; m < n_mels_needed; m++) {
        int32_t sum_db_q9 = 0;
        int16_t row_max_q9 = INT16_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int16_t v = mel_db_q9[idx];
            if (v < clip_floor_q9) v = clip_floor_q9;
            mel_db_q9[idx] = v;
            sum_db_q9 += (int32_t)v;
            if (v > row_max_q9) row_max_q9 = v;
        }

        int16_t mean_q9 = fxp_sat_s16_from_s32(fxp_round_div_s32(sum_db_q9, n_frames));

        int32_t sum_sq_q18 = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            int32_t d = (int32_t)mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] - (int32_t)mean_q9;
            sum_sq_q18 += d * d;
        }
        int32_t var_q18 = fxp_round_div_s32(sum_sq_q18, n_frames);
        if (var_q18 < 0) var_q18 = 0;
        int16_t std_q9 = fxp_sat_s16_from_s32((int32_t)fxp_sqrt32((uint32_t)var_q18));

        uint16_t ent_q14 = _mel_entropy_row_q14(&mel_power[(size_t)m * (size_t)n_frames],
                                                n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] =
            (fxp_feat_t)mean_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] =
            (fxp_feat_t)std_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] =
            (fxp_feat_t)row_max_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] =
            (fxp_feat_t)ent_q14;
    }

    free(sig_q);
    free(padded_q);
    free(timedata);
    free(cx_out);
    free(frame_power);
    free(mel_power);
    free(mel_db_q9);
    free(cfg);
}

#if defined(FXP_STAGE_PROBES)
void audio_mel_stage_probe_free(audio_mel_stage_probe_t *probe)
{
    if (!probe) return;
    free(probe->frame_power);
    free(probe->mel_power);
    free(probe->mel_db_q9);
    probe->frame_power = NULL;
    probe->mel_power = NULL;
    probe->mel_db_q9 = NULL;
}

int audio_mel_stage_probe(const int8_t *features_selector,
                          const int16_t *sig_q14,
                          int16_t len,
                          audio_mel_stage_probe_t *probe)
{
    if (!features_selector || !sig_q14 || !probe || len <= 0) return 0;

    memset(probe, 0, sizeof(*probe));
    if (!_mel_any_required(features_selector)) return 0;
    if (len <= PAD_LEN) return 0;

    int16_t n_mels_needed = 0;
    for (uint8_t i = 0; i < N_MFCC; i++) {
        if (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] ||
            features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i]) {
            probe->idxs_needed[n_mels_needed] = i;
            n_mels_needed++;
        }
    }
    if (n_mels_needed <= 0) return 0;

    int16_t padded_len = (int16_t)(len + (2 * PAD_LEN));
    int16_t n_frames = (int16_t)(((padded_len - N_FFT) / HOP_LEN) + 1);
    if (n_frames <= 0) return 0;

    probe->n_frames = n_frames;
    probe->n_mels = n_mels_needed;
    probe->frame_power = (uint64_t *)malloc((size_t)n_frames * (size_t)FFT_RES_LEN * sizeof(uint64_t));
    probe->mel_power = (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));
    probe->mel_db_q9 = (int16_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int16_t));

    fxp_mel_sig_t *sig_q = (fxp_mel_sig_t *)malloc((size_t)len * sizeof(fxp_mel_sig_t));
    fxp_mel_sig_t *padded_q = (fxp_mel_sig_t *)malloc((size_t)padded_len * sizeof(fxp_mel_sig_t));
    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!probe->frame_power || !probe->mel_power || !probe->mel_db_q9 ||
        !sig_q || !padded_q || !timedata || !cx_out || !cfg) {
        audio_mel_stage_probe_free(probe);
        free(sig_q);
        free(padded_q);
        free(timedata);
        free(cx_out);
        free(cfg);
        memset(probe, 0, sizeof(*probe));
        return 0;
    }

    _mel_ensure_kiss_ref();

    int32_t ln_ref_q9 = fxp_ln_u64_q9((uint64_t)_kiss_ref_abs);
    int32_t kiss_ref_offset_q9 =
        -((int32_t)((((int64_t)ln_ref_q9 * (int64_t)FXP_MEL_20DB_PER_LN_Q20) + (1LL << 19)) >> 20));
    int32_t db_offset_base_q9 =
        kiss_ref_offset_q9 - ((int32_t)FXP_MEL_ACC_EXTRA_FRAC * FXP_MEL_DB_PER_POWER_BIT_Q9);
    probe->stft_db_offset_q9 = kiss_ref_offset_q9;
    probe->mel_db_offset_q9 = db_offset_base_q9;

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

    int16_t max_db_q9 = INT16_MIN;
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

        uint64_t *frame = &probe->frame_power[(size_t)f * (size_t)FFT_RES_LEN];
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            int64_t re = (int64_t)cx_out[k].r;
            int64_t im = (int64_t)cx_out[k].i;
            frame[k] = (uint64_t)(re * re) + (uint64_t)(im * im);
        }

        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)probe->idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uint64_t sum = 0ULL;
            for (int16_t k = start; k <= end; k++) {
                uint16_t w_q15 = fxp_mel_basis_q15[mel_idx][k - start];
                uint64_t term = fxp_round_shift_u64(frame[k] * (uint64_t)w_q15,
                                                    FXP_MEL_ACC_SHIFT);
                if (UINT64_MAX - sum < term) {
                    sum = UINT64_MAX;
                } else {
                    sum += term;
                }
            }

            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            probe->mel_power[idx] = sum;
            probe->mel_db_q9[idx] = _mel_db_from_power_q9(sum, db_offset_base_q9);
            if (probe->mel_db_q9[idx] > max_db_q9) {
                max_db_q9 = probe->mel_db_q9[idx];
            }
        }
    }

    int16_t clip_floor_q9 = fxp_sat_s16_from_s32((int32_t)max_db_q9 - FXP_MEL_TOP_DB_Q9);

    for (int16_t m = 0; m < n_mels_needed; m++) {
        int32_t sum_db_q9 = 0;
        int16_t row_max_q9 = INT16_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int16_t v = probe->mel_db_q9[idx];
            if (v < clip_floor_q9) v = clip_floor_q9;
            probe->mel_db_q9[idx] = v;
            sum_db_q9 += (int32_t)v;
            if (v > row_max_q9) row_max_q9 = v;
        }

        int16_t mean_q9 = fxp_sat_s16_from_s32(fxp_round_div_s32(sum_db_q9, n_frames));

        int32_t sum_sq_q18 = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            int32_t d = (int32_t)probe->mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] - (int32_t)mean_q9;
            sum_sq_q18 += d * d;
        }
        int32_t var_q18 = fxp_round_div_s32(sum_sq_q18, n_frames);
        if (var_q18 < 0) var_q18 = 0;

        uint8_t mel_bin = probe->idxs_needed[m];
        probe->mean_q9[mel_bin] = mean_q9;
        probe->std_q9[mel_bin] = fxp_sat_s16_from_s32((int32_t)fxp_sqrt32((uint32_t)var_q18));
        probe->max_q9[mel_bin] = row_max_q9;
        probe->entropy_q14[mel_bin] =
            _mel_entropy_row_q14(&probe->mel_power[(size_t)m * (size_t)n_frames],
                                 n_frames);
    }

    free(sig_q);
    free(padded_q);
    free(timedata);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

#endif
