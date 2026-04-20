#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <audio_features.h>
#include <kiss_fftr.h>
#include <welch_psd.h>

#include <audio/audio_periodogram_block.h>
#include <audio/audio_periodogram_bridge.h>
#include <audio/audio_periodogram_kernels.h>
#include <audio/audio_periodogram_lut.h>
#include <core/fxp_convert.h>
#include <core/fxp_sat.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

#define FXP_HANN_FRAC_BITS 15
#define FXP_LN2_Q24 ((int32_t)11629080) /* round(ln(2) * 2^24) */

#if (FIXED_POINT == 32)
/* Periodogram-only high-precision input path for flatness robustness.
 * Keep FFT block untouched (still int16/Q14).
 */
#define FXP_PSD_SIG_FRAC_BITS 30
typedef int32_t fxp_psd_sig_t;
#else
#define FXP_PSD_SIG_FRAC_BITS FXP_FRAC_AUDIO_INPUT
typedef int16_t fxp_psd_sig_t;
#endif

static uint16_t _hann_q15[NPERSEG];
static int _hann_q15_initialized = 0;

static inline int32_t _round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static uint16_t _to_hann_q15(float v)
{
    uint32_t q = FXP_FROM_FLOAT_U(v, FXP_HANN_FRAC_BITS);
    return fxp_sat_u16_from_u32(q);
}

static inline fxp_psd_sig_t _to_psd_sig_q(float x)
{
    int32_t q = fxp_from_float_signed(x, FXP_PSD_SIG_FRAC_BITS);
#if (FIXED_POINT == 32)
    return q;
#else
    return fxp_sat_s16_from_s32(q);
#endif
}

static inline uint64_t _round_shift_u64(uint64_t v, uint32_t shift)
{
    if (shift == 0U) return v;
    if (shift >= 64U) return 0ULL;
    return (v + (1ULL << (shift - 1U))) >> shift;
}

/* Natural logarithm on unsigned integer input, result in Q11.
 * LUT interpolation on mantissa in [1,2), exponent via ln(2).
 */
static int32_t _fxp_ln_u64_q11(uint64_t x)
{
    if (x == 0ULL) x = 1ULL;

    uint32_t msb = 63U - (uint32_t)__builtin_clzll(x);
    uint64_t base = 1ULL << msb;
    uint64_t diff = x - base; /* diff < base = 2^msb, so diff fits in msb bits */
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

static void _ensure_hann_q15(void)
{
    if (_hann_q15_initialized) return;
    for (int16_t i = 0; i < NPERSEG; i++) {
        _hann_q15[i] = _to_hann_q15(hann_window[i]);
    }
    _hann_q15_initialized = 1;
}

static void _fxp_audio_psd_write_features(const int8_t *features_selector,
                                          const fxp_audio_psd_view_t *view,
                                          float *feats)
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
        uq12_20_t dom_q20 = fxp_audio_psd_dominant_freq_q20(view);
        feats[DOMINANT_FREQUENCY] = fxp_audio_psd_bridge_domfreq_to_float(dom_q20);
    }

    if (need_flatness) {
        uq0_16_t flatness_q16 = fxp_audio_psd_flatness_q16(view);
        feats[SPECTRAL_FLATNESS] = fxp_audio_psd_bridge_flatness_to_float(flatness_q16);
    }

    if (need_bandpowers) {
        uq0_16_t band_powers_q16[N_PSD] = {0};
        fxp_audio_psd_bandpowers_q16(view, &features_selector[POWER_SPECTRAL_DENSITY], band_powers_q16);
        for (int8_t i = 0; i < N_PSD; i++) {
            if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
                feats[POWER_SPECTRAL_DENSITY + i] =
                    fxp_audio_psd_bridge_bandpower_to_float(band_powers_q16[i]);
            }
        }
    }
}

void fxp_audio_periodogram_features_from_signal(const int8_t *features_selector,
                                                const float *sig,
                                                int16_t sig_len,
                                                int16_t fs,
                                                float *feats)
{
    if (!features_selector || !sig || !feats || sig_len <= 0 || fs <= 0) return;
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

    float max_abs = 0.0f;
    for (int16_t i = 0; i < sig_len; i++) {
        float a = fabsf(sig[i]);
        if (a > max_abs) max_abs = a;
    }
    float norm_gain = (max_abs > 0.0f) ? (1.0f / max_abs) : 1.0f;

    for (int16_t i = 0; i < sig_len; i++) {
        sig_q[i] = _to_psd_sig_q(sig[i] * norm_gain);
    }
    memset(acc_power, 0, (size_t)psd_len * sizeof(uint64_t));
    _ensure_hann_q15();

    int16_t start = 0;
    for (int16_t step = 0; step < steps; step++) {
        if ((int32_t)start + NPERSEG > sig_len) break;

        int64_t sum_qsig = 0;
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_qsig += (int64_t)sig_q[start + i];
        }
        int32_t mean_qsig = _round_div_s64(sum_qsig, NPERSEG);

        for (int16_t i = 0; i < NPERSEG; i++) {
            int32_t centered_qsig = (int32_t)sig_q[start + i] - mean_qsig;
            int64_t prod_qsig_w = (int64_t)centered_qsig * (int64_t)_hann_q15[i];
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
    int32_t ln_mean_power_q11 = _fxp_ln_u64_q11(mean_power);

    /* proxy[i] = acc_power[i] * psd_len * 2^PROXY_FRAC / total_power.
     * Pick norm_shift so (acc_power[i] >> norm_shift) * scale_num fits in u64,
     * then divide by (total_power >> norm_shift) to preserve the ratio.
     */
    uint64_t scale_num = (uint64_t)psd_len << FXP_FRAC_AUDIO_PSD_PROXY;
    uint32_t acc_msb = (max_power > 0ULL) ? (63U - (uint32_t)__builtin_clzll(max_power)) : 0U;
    uint32_t scale_msb = (scale_num > 0ULL) ? (63U - (uint32_t)__builtin_clzll(scale_num)) : 0U;
    uint32_t norm_shift = 0U;
    if (acc_msb + scale_msb + 2U > 63U) {
        norm_shift = (acc_msb + scale_msb + 2U) - 63U;
    }
    uint64_t total_scaled = _round_shift_u64(total_power, norm_shift);
    if (total_scaled == 0ULL) total_scaled = 1ULL;

    for (int16_t i = 0; i < psd_len; i++) {
        uint64_t p_scaled = _round_shift_u64(acc_power[i], norm_shift);
        uint64_t num = p_scaled * scale_num;
        uint64_t proxy64 = (num + (total_scaled >> 1U)) / total_scaled;
        uint32_t proxy = (uint32_t)((proxy64 > (uint64_t)UINT32_MAX) ? UINT32_MAX : proxy64);
        if (proxy == 0U && acc_power[i] != 0ULL) proxy = 1U;
        proxy_q14[i] = fxp_sat_u32_from_u64((uint64_t)proxy);

        uint64_t p_for_log = (acc_power[i] == 0ULL) ? 1ULL : acc_power[i];
        log_proxy_q11[i] = _fxp_ln_u64_q11(p_for_log) - ln_mean_power_q11;

        uint64_t freq_q20 =
            (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)NPERSEG;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    fxp_audio_psd_view_t view = {
        .proxy_q14 = proxy_q14,
        .log_proxy_q11 = log_proxy_q11,
        .freqs_q20 = freqs_q20,
        .len = psd_len,
    };
    _fxp_audio_psd_write_features(features_selector, &view, feats);

    free(sig_q);
    free(timedata);
    free(cx_out);
    free(acc_power);
    free(proxy_q14);
    free(log_proxy_q11);
    free(freqs_q20);
    free(cfg);
}

void fxp_audio_periodogram_features_hybrid(const int8_t *features_selector,
                                           const float *psd,
                                           int16_t psd_len,
                                           int16_t fs,
                                           int16_t sig_len,
                                           float *feats)
{
    if (!features_selector || !psd || !feats || psd_len <= 0 || fs <= 0 || sig_len <= 0) return;

    uq18_14_t *proxy_q14 = (uq18_14_t *)malloc((size_t)psd_len * sizeof(uq18_14_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    if (!proxy_q14 || !freqs_q20) {
        free(proxy_q14);
        free(freqs_q20);
        return;
    }

    fxp_audio_psd_bridge_from_float(psd, psd_len, fs, sig_len, proxy_q14, freqs_q20);

    fxp_audio_psd_view_t view = {
        .proxy_q14 = proxy_q14,
        .log_proxy_q11 = NULL,
        .freqs_q20 = freqs_q20,
        .len = psd_len,
    };
    _fxp_audio_psd_write_features(features_selector, &view, feats);

    free(proxy_q14);
    free(freqs_q20);
}

#endif
