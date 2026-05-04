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

#define FXP_FFT_ROLLOFF_95_Q16 ((uint32_t)62259U)

#define FXP_FFT_INPUT_FRAC 30

/* Frequency deviation from the spectral centroid.
 * Frequency is UQ12.20 and centroid is UQ11.21, so both are converted to Q13.19
 * before subtracting.
 */
static inline q13_19_t _dev(uq12_20_t freq_q20, uq11_21_t centroid_q21) {
    q13_19_t freq_q19 = (q13_19_t)(freq_q20 >> 1);
    q13_19_t centroid_q19 = (q13_19_t)(centroid_q21 >> 2);
    return (q13_19_t)(freq_q19 - centroid_q19);
}
/* Spectral rolloff: first frequency bin where the running magnitude reaches
 * 95% of the total magnitude. Magnitudes are accumulated as UQ15.17 to match
 * sum_mags.
 */
static uq12_20_t _rolloff(const uq12_20_t *mags, const uq12_20_t *freqs, int16_t len,
                          uq15_17_t sum_mags) {
    // Add half a Q16 step before shifting so the 0.95 multiply rounds instead of flooring.
    uq15_17_t rolloff_energy =
        (uq15_17_t)((((uint64_t)sum_mags * (uint64_t)FXP_FFT_ROLLOFF_95_Q16) + (1ULL << 15)) >> 16);

    uq15_17_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        sum += (mags[i] >> 3);
        if (sum >= rolloff_energy) {
            return freqs[i];
        }
    }
    return freqs[len - 1];
}
/* Shared rounded shifts for FFT/PSD conversions where truncation causes a
 * visible bias against the floating-point reference.
 */
static inline int32_t _round_shift_s64_to_s32(int64_t v, uint32_t shift) {
    if (shift == 0U) return fxp_sat_s32_from_s64(v);
    if (shift >= 63U) return (v < 0) ? -1 : 0;
    if (v >= 0) return fxp_sat_s32_from_s64((v + (1LL << (shift - 1U))) >> shift);
    return -fxp_sat_s32_from_s64(((-v) + (1LL << (shift - 1U))) >> shift);
}
static inline uint64_t _round_shift_u64_local(uint64_t v, uint32_t shift) {
    return fxp_round_shift_u64(v, shift);
}

/* Spectral centroid: weighted mean frequency, sum(freq * magnitude) / sum(magnitude).
 * The product UQ12.20 * UQ12.20 is shifted to UQ26.38 before accumulation;
 * dividing by UQ15.17 produces UQ11.21.
 */
static uq11_21_t _centroid(const uq12_20_t *mags, const uq12_20_t *freqs, int16_t len,
                           uq15_17_t sum_mags) {

    uq26_38_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uq24_40_t product = (uq24_40_t)((uint64_t)freqs[i] * (uint64_t)mags[i]);
        sum += (uq26_38_t)(product >> 2U);
    }

    uq11_21_t centroid = (sum) / (uint64_t)sum_mags;
    return centroid;
}
/* Spectral spread: sqrt of the magnitude-weighted variance around centroid.
 * dev^2 is kept as UQ25.7, multiplied by magnitude UQ12.20, then divided by
 * sum_mags to produce a UQ22.10 sqrt input and UQ11.5 output.
 */
static uq11_5_t _spread(const uq12_20_t *mags, const uq12_20_t *freqs, int16_t len,
                        uq15_17_t sum_mags, uq11_21_t centroid) {

    uq37_27_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        q13_19_t dev = _dev(freqs[i], centroid);
        uq25_7_t dev_2 = (uq25_7_t)(((int64_t)dev * (int64_t)dev) >> 31);
        sum += (uq37_27_t)((uint64_t)dev_2 * (uint64_t)mags[i]);
    }

    uq22_10_t mean = (uq22_10_t)(sum / (uint64_t)sum_mags);
    return (uq11_5_t)fxp_sqrt32(mean);
}
/* Spectral kurtosis: magnitude-weighted mean of normalized fourth powers.
 * inv_spread lets each deviation be normalized before raising it to the fourth
 * power, keeping the intermediate integer range manageable.
 */
static uq17_15_t _kurtosis(const uq12_20_t *mags, const uq12_20_t *freqs, int16_t len,
                           uq15_17_t sum_mags, uq11_21_t centroid, uq11_5_t spread) {

    uq12_20_t inv_spread = (uq12_20_t)((1U << 25) / (uint32_t)spread);
    uq32_32_t sum = 0;
    for (int16_t i = 0; i < len; i++) {

        q13_19_t dev = _dev(freqs[i], centroid);

        q5_11_t norm = (q5_11_t)(((int64_t)dev * (int64_t)inv_spread) >> 28U);
        uq10_22_t norm_2 = (uq10_22_t)((int32_t)norm * (int32_t)norm);
        uq20_12_t norm_4 = (uq20_12_t)(((uint64_t)norm_2 * (uint64_t)norm_2) >> 32);
        sum += (uq32_32_t)((uint64_t)norm_4 * (uint64_t)mags[i]);
    }

    uq17_15_t kurtosis = (uq17_15_t)((sum) / (uint64_t)sum_mags);
    return kurtosis;
}

#if defined(FXP_STAGE_PROBES)
int audio_fft_stage_probe(const int16_t *sig, int16_t len, int16_t fs, uq12_20_t *mags,
                          uq12_20_t *freqs, uq15_17_t *sum_mags) {

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
        sig_q[i] = (kiss_fft_scalar)((int32_t)sig[i] << 16);
    }

    kiss_fftr(cfg, sig_q, cx_out);

    uint32_t sum = 0U;
    for (int16_t i = 0; i < fft_len; i++) {
        q12_20_t re = (q12_20_t)(cx_out[i].r >> 10);
        q12_20_t im = (q12_20_t)(cx_out[i].i >> 10);
        uint64_t re_2 = (uint64_t)((int64_t)re * (int64_t)re);
        uint64_t im_2 = (uint64_t)((int64_t)im * (int64_t)im);
        mags[i] = (uq12_20_t)fxp_sqrt64(re_2 + im_2);
        sum += (mags[i] >> 3);

        freqs[i] = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20U) / (uint64_t)len);
    }

    *sum_mags = (uq15_17_t)sum;

    free(sig_q);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

void audio_fft_features(const int8_t *features_selector, const int16_t *sig, int16_t len,
                        int16_t fs, fxp_feat_t *feats) {
    if (!features_selector || !sig || !feats || len <= 0 || fs <= 0) return;

    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID] ||
                        features_selector[SPECTRAL_SPREAD] || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD] || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];
    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    int16_t fft_len = (int16_t)((len / 2) + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    if (!cfg) return;

    kiss_fft_scalar *sig_q = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));
    uq12_20_t *mags = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uq12_20_t *freqs = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));

    if (!sig_q || !cx_out || !mags || !freqs) {
        free(sig_q);
        free(cx_out);
        free(mags);
        free(freqs);
        free(cfg);
        return;
    }
    // this type is used solely to be compatible with the kiss_fft library and not modify it
    //  the type kiss_fft_scalar should be interpreted as a Q2.30 format
    //  to use 32 bit twiddle factors, this is necessary
    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = (kiss_fft_scalar)((int32_t)sig[i] << 16);
    }
    // Below here are the RFFT kernel features
    kiss_fftr(cfg, sig_q, cx_out);

    uq15_17_t sum = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        // Convert KissFFT output from Q2.30 to Q12.20.
        q12_20_t re = (q12_20_t)(cx_out[i].r >> 10);
        q12_20_t im = (q12_20_t)(cx_out[i].i >> 10);
        uq24_40_t re_2 = (uq24_40_t)((int64_t)re * (int64_t)re);
        uq24_40_t im_2 = (uq24_40_t)((int64_t)im * (int64_t)im);
        mags[i] = (uq12_20_t)fxp_sqrt64(re_2 + im_2);
        sum += (mags[i] >> 3);

        freqs[i] = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20) / (uint64_t)len);
    }

    uq15_17_t sum_mags = (uq15_17_t)sum;

    // Below are all the FFT feature kernel calls
    if (need_rolloff) {
        uq12_20_t rolloff = _rolloff(mags, freqs, fft_len, sum_mags);
        feats[SPECTRAL_ROLLOFF] = (fxp_feat_t)rolloff;
    }

    uq11_21_t centroid = 0;
    if (need_centroid) {
        centroid = _centroid(mags, freqs, fft_len, sum_mags);
        if (features_selector[SPECTRAL_CENTROID]) {
            feats[SPECTRAL_CENTROID] = (fxp_feat_t)centroid;
        }
    }

    uq11_5_t spread = 0;
    if (need_spread) {
        spread = _spread(mags, freqs, fft_len, sum_mags, centroid);
        if (features_selector[SPECTRAL_SPREAD]) {
            feats[SPECTRAL_SPREAD] = (fxp_feat_t)spread;
        }
    }

    if (need_kurt) {
        uq17_15_t kurtosis = _kurtosis(mags, freqs, fft_len, sum_mags, centroid, spread);
        feats[SPECTRAL_KURTOSIS] = (fxp_feat_t)kurtosis;
    }

    free(sig_q);
    free(cx_out);
    free(mags);
    free(freqs);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Periodogram kernels + block                                               */
/* -------------------------------------------------------------------------- */

#define FXP_PSD_PROXY_TO_INT_SHIFT (FXP_FRAC_AUDIO_PSD_PROXY - FXP_FRAC_AUDIO_PSD_INTEGRAL)
#define FXP_HANN_FRAC_BITS 15

/* Composite Simpson's rule over an even number of intervals.
 * Proxy samples UQ21.11 are shifted down to integer units so the running sum
 * fits in u32 before the final divide-by-3.
 */
static uint32_t _psd_simpson_step(const uq21_11_t *x, int16_t start, int16_t end) {
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uint32_t sum = 0;

    for (int i = 0; i < n_intervals; i++) {
        uint32_t x0 = (uint32_t)(x[idx] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint32_t x1 = (uint32_t)(x[idx + 1] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        uint32_t x2 = (uint32_t)(x[idx + 2] >> FXP_PSD_PROXY_TO_INT_SHIFT);
        sum += x0 + (x1 << 2) + x2;
        idx += 2;
    }

    return (sum + 1U) / 3U;
}

/* Numerical integral of the proxy spectrum.
 * For odd lengths Simpson's rule applies directly; for even lengths the last
 * interval is split off as a trapezoid average so the rule's even-interval
 * requirement is preserved on both halves.
 */
static uint32_t _psd_simpson(const uq21_11_t *x, int16_t len) {
    if (!x || len <= 1) return 0U;

    if ((len & 1) == 0) {
        uint32_t val = (((uint32_t)(x[len - 1] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                         (uint32_t)(x[len - 2] >> FXP_PSD_PROXY_TO_INT_SHIFT)) +
                        1U) >>
                       1;
        uint32_t result = _psd_simpson_step(x, 0, len - 1);

        val += ((((uint32_t)(x[0] >> FXP_PSD_PROXY_TO_INT_SHIFT) +
                  (uint32_t)(x[1] >> FXP_PSD_PROXY_TO_INT_SHIFT)) +
                 1U) >>
                1);
        result += _psd_simpson_step(x, 1, len);

        val = (val + 1U) >> 1;
        result = (result + 1U) >> 1;
        return result + val;
    }

    return _psd_simpson_step(x, 0, len);
}

// Dominant frequency: frequency corresponding to the maximum proxy value.
// takes in the proxy in UQ21.11 and frequencies in UQ12.20, returns UQ12.20.
static uq12_20_t _dominant_freq(const uq21_11_t *proxy, const uq12_20_t *freqs, int16_t len) {
    int16_t max_idx = 0;
    uq21_11_t max_val = proxy[0];
    for (int16_t i = 1; i < len; i++) {
        if (proxy[i] > max_val) {
            max_val = proxy[i];
            max_idx = i;
        }
    }
    return freqs[max_idx];
}

// Spectral flatness: exp(mean(log(power)) - log(mean(power))).
// Proxy input is UQ21.11;
static uq0_16_t _flatness(const uq21_11_t *proxy, int16_t len) {
    q21_11_t sum_logs = 0;
    q21_11_t sum_proxy = 0;

    for (int16_t i = 0; i < len; i++) {
        uq21_11_t v = (proxy[i] == 0U) ? 1U : proxy[i];
        sum_logs += _log_psd(v);
        sum_proxy += v;
    }

    if (sum_proxy == 0U) return 0;

    q21_11_t mean_log = (q21_11_t)(sum_logs / (int32_t)len);
    q21_11_t mean_proxy = (q21_11_t)(sum_proxy / (uint32_t)len);
    if (mean_proxy == 0U) mean_proxy = 1U;

    q21_11_t log_mean_proxy = _log_psd(mean_proxy);
    q5_11_t diff = (q5_11_t)(mean_log - log_mean_proxy);
    if (diff > 0) diff = 0;
    return _exp_psd((q5_11_t)diff);
}

/* Per-band relative power: integral of the proxy over each [start, end] band
 * divided by the total proxy integral. Output ratios are UQ0.16 so each band
 * is expressed as a fraction of total power.
 */
static void _bandpowers(const uq21_11_t *proxy, const uq12_20_t *freqs, int16_t len,
                        const int8_t *psd_selector, uq0_16_t *band_powers) {
    if (!band_powers) return;
    for (int8_t i = 0; i < N_PSD; i++)
        band_powers[i] = 0;

    if (!proxy || !freqs || !psd_selector || len <= 2) return;

    uq24_8_t total_power = (uq24_8_t)_psd_simpson(proxy, len);
    if (total_power == 0U) return;

    for (int8_t i = 0; i < N_PSD; i++) {
        if (!psd_selector[i]) continue;

        uq12_20_t band_start = (uq12_20_t)((uint32_t)psd_bands[i].start << 20U);
        uq12_20_t band_end = (uq12_20_t)((uint32_t)psd_bands[i].end << 20U);

        int16_t start_idx = 0;
        int16_t n_bins = 0;
        int found = 0;

        for (int16_t j = 0; j < len; j++) {
            uq12_20_t f = freqs[j];
            if (!found && f >= band_start) {
                start_idx = j;
                found = 1;
            }
            if (found && f <= band_end) {
                n_bins++;
            } else if (found) {
                break;
            }
        }

        if (!found || n_bins <= 1) {
            band_powers[i] = 0;
            continue;
        }

        uq24_8_t band_power = (uq24_8_t)_psd_simpson(&proxy[start_idx], n_bins);
        // this down here probably needs fixing
        uq10_6_t ratio =
            (uq10_6_t)((((uint64_t)band_power << 16) + (total_power >> 1)) / total_power);
        band_powers[i] = ratio;
    }
}

void audio_psd_features(const int8_t *features_selector, const int16_t *sig, int16_t sig_len,
                        int16_t fs, fxp_feat_t *feats) {
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
    // timedata is in Q2.30 format to be compatible with kiss_fft, which uses 32 bit twiddle factors
    // (this is the windowed signal)
    kiss_fft_scalar *timedata =
        (kiss_fft_scalar *)malloc((size_t)NPERSEG * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)psd_len * sizeof(kiss_fft_cpx));
    uq21_11_t *acc_power = (uq21_11_t *)malloc((size_t)psd_len * sizeof(uq21_11_t));
    uq21_11_t *proxy = (uq21_11_t *)malloc((size_t)psd_len * sizeof(uq21_11_t));
    uq12_20_t *freqs = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(NPERSEG, 0, 0, 0);

    if (!timedata || !cx_out || !acc_power || !proxy || !freqs || !cfg) {
        free(timedata);
        free(cx_out);
        free(acc_power);
        free(proxy);
        free(freqs);
        free(cfg);
        return;
    }

    memset(acc_power, 0, (size_t)psd_len * sizeof(uq21_11_t));

    // Welch periodogram: window each segment, RFFT it, accumulate |X|^2.
    int16_t start = 0;
    for (int16_t step = 0; step < steps; step++) {
        if ((int32_t)start + NPERSEG > sig_len) break;
        int32_t sum_sig = 0;
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_sig += (int32_t)sig[start + i];
        }
        q2_14_t mean_sig = (q2_14_t)(sum_sig / (int32_t)NPERSEG);

        for (int16_t i = 0; i < NPERSEG; i++) {

            q2_14_t dev_sig = (q2_14_t)(sig[start + i] - mean_sig);
            q2_14_t window_sig =
                (q2_14_t)(((int32_t)dev_sig * (int32_t)fxp_hann_window_q15[i]) >> 15);

            // Convert from Q2.14 to Q2.30 for KissFFT input.
            timedata[i] = (kiss_fft_scalar)((int32_t)window_sig << 16);
        }

        kiss_fftr(cfg, timedata, cx_out);

        for (int16_t i = 0; i < psd_len; i++) {
            // Convert KissFFT output from Q2.30 to Q8.8.
            q8_8_t re = (q8_8_t)(cx_out[i].r >> 22);
            q8_8_t im = (q8_8_t)(cx_out[i].i >> 22);
            uq16_16_t re_2 = (uq16_16_t)((int32_t)re * (int32_t)re);
            uq16_16_t im_2 = (uq16_16_t)((int32_t)im * (int32_t)im);
            uq16_16_t mag = (uq16_16_t)(re_2 + im_2);
            uq21_11_t power = (uq21_11_t)(mag >> 5);
            acc_power[i] += power;
        }

        start = (int16_t)(start + hop);
    }

    // Accumulated power becomes the PSD proxy; segment count and Hann gain
    // cancel in every downstream feature, so no normalization is needed here.
    for (int16_t i = 0; i < psd_len; i++) {
        proxy[i] = (uq21_11_t)acc_power[i];
        uq12_20_t freq = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20U) / (uint64_t)NPERSEG);
        freqs[i] = freq;
    }

    // Below are all the PSD feature kernel calls
    if (need_dom_freq) {
        uq12_20_t dom = _dominant_freq(proxy, freqs, psd_len);
        feats[DOMINANT_FREQUENCY] = (fxp_feat_t)dom;
    }

    if (need_flatness) {
        uq0_16_t flatness = _flatness(proxy, psd_len);
        feats[SPECTRAL_FLATNESS] = (fxp_feat_t)flatness;
    }

    if (need_bandpowers) {
        uq0_16_t band_powers[N_PSD] = {0};
        _bandpowers(proxy, freqs, psd_len, &features_selector[POWER_SPECTRAL_DENSITY], band_powers);
        for (int8_t i = 0; i < N_PSD; i++) {
            if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
                feats[POWER_SPECTRAL_DENSITY + i] = (fxp_feat_t)band_powers[i];
            }
        }
    }

    free(timedata);
    free(cx_out);
    free(acc_power);
    free(proxy);
    free(freqs);
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

#define FXP_MEL_INPUT_FRAC 31
#define FXP_MEL_SCALAR_MAX_I INT32_MAX
typedef int32_t fxp_mel_sig_t;

static uint32_t _kiss_ref_abs = (uint32_t)N_FFT;
static int _kiss_ref_initialized = 0;

static int _mel_any_required(const int8_t *features_selector) {
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT; i < ZERO_CROSSING_RATE; i++) {
        if (features_selector[i]) return 1;
    }
    return 0;
}

static inline uint32_t _mel_uq_div_u64_q(uint64_t num, uint64_t den, uint8_t frac_bits) {
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

static inline uint8_t _mel_ceil_log2_u16(uint16_t v) {
    if (v <= 1U) return 0U;
    uint16_t x = (uint16_t)(v - 1U);
    uint8_t bits = 0U;
    while (x) {
        x >>= 1U;
        bits++;
    }
    return bits;
}

static inline fxp_mel_sig_t _mel_from_input_q14(int16_t x_q14) {
    int64_t q = ((int64_t)x_q14) << (FXP_MEL_INPUT_FRAC - FXP_FRAC_AUDIO_INPUT);
    return (fxp_mel_sig_t)fxp_sat_s32_from_s64(q);
}

static void _mel_ensure_kiss_ref(void) {
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

static int16_t _mel_db_from_power_q9(uint64_t p_scaled, int32_t db_offset_q9) {
    int16_t ln_q9 = fxp_ln_u64_q9((p_scaled == 0ULL) ? 1ULL : p_scaled);
    int32_t db_q9 =
        (int32_t)((((int64_t)ln_q9 * (int64_t)FXP_MEL_DB_PER_LN_Q20) + (1LL << 19)) >> 20);
    return fxp_sat_s16_from_s32(db_q9 + db_offset_q9);
}

static uint16_t _mel_entropy_row_q14(const uint64_t *row_power, int16_t n_frames) {
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
        entropy_q14 +=
            (prod_q25 + (1LL << (FXP_MEL_ENTROPY_PROD_SHIFT - 1))) >> FXP_MEL_ENTROPY_PROD_SHIFT;
    }

    if (entropy_q14 <= 0) return 0U;
    if (entropy_q14 > UINT16_MAX) return UINT16_MAX;
    return (uint16_t)entropy_q14;
}

void audio_mel_features(const int8_t *features_selector, const int16_t *sig_q14, int16_t len,
                        fxp_feat_t *feats) {
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
    uint64_t *mel_power =
        (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));
    int16_t *mel_db_q9 =
        (int16_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int16_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!sig_q || !padded_q || !timedata || !cx_out || !frame_power || !mel_power || !mel_db_q9 ||
        !cfg) {
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
        -((int32_t)((((int64_t)ln_ref_q9 * (int64_t)FXP_MEL_20DB_PER_LN_Q20) + (1LL << 19)) >>
                    20)) -
        ((int32_t)FXP_MEL_ACC_EXTRA_FRAC * FXP_MEL_DB_PER_POWER_BIT_Q9);

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

            timedata[n] = (kiss_fft_scalar)win_q;
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
                uint64_t term =
                    fxp_round_shift_u64(frame_power[k] * (uint64_t)w_q15, FXP_MEL_ACC_SHIFT);
                if (UINT64_MAX - sum < term) {
                    sum = UINT64_MAX;
                } else {
                    sum += term;
                }
            }

            mel_power[(size_t)m * (size_t)n_frames + (size_t)f] = sum;
            mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] =
                _mel_db_from_power_q9(sum, db_offset_base_q9);
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
            int32_t d =
                (int32_t)mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] - (int32_t)mean_q9;
            sum_sq_q18 += d * d;
        }
        int32_t var_q18 = fxp_round_div_s32(sum_sq_q18, n_frames);
        if (var_q18 < 0) var_q18 = 0;
        int16_t std_q9 = fxp_sat_s16_from_s32((int32_t)fxp_sqrt32((uint32_t)var_q18));

        uint16_t ent_q14 = _mel_entropy_row_q14(&mel_power[(size_t)m * (size_t)n_frames], n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] = (fxp_feat_t)mean_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] = (fxp_feat_t)std_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] = (fxp_feat_t)row_max_q9;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] = (fxp_feat_t)ent_q14;
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
void audio_mel_stage_probe_free(audio_mel_stage_probe_t *probe) {
    if (!probe) return;
    free(probe->frame_power);
    free(probe->mel_power);
    free(probe->mel_db_q9);
    probe->frame_power = NULL;
    probe->mel_power = NULL;
    probe->mel_db_q9 = NULL;
}

int audio_mel_stage_probe(const int8_t *features_selector, const int16_t *sig_q14, int16_t len,
                          audio_mel_stage_probe_t *probe) {
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
    probe->frame_power =
        (uint64_t *)malloc((size_t)n_frames * (size_t)FFT_RES_LEN * sizeof(uint64_t));
    probe->mel_power =
        (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));
    probe->mel_db_q9 =
        (int16_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int16_t));

    fxp_mel_sig_t *sig_q = (fxp_mel_sig_t *)malloc((size_t)len * sizeof(fxp_mel_sig_t));
    fxp_mel_sig_t *padded_q = (fxp_mel_sig_t *)malloc((size_t)padded_len * sizeof(fxp_mel_sig_t));
    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!probe->frame_power || !probe->mel_power || !probe->mel_db_q9 || !sig_q || !padded_q ||
        !timedata || !cx_out || !cfg) {
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

            timedata[n] = (kiss_fft_scalar)win_q;
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
                uint64_t term = fxp_round_shift_u64(frame[k] * (uint64_t)w_q15, FXP_MEL_ACC_SHIFT);
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
            int32_t d = (int32_t)probe->mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] -
                        (int32_t)mean_q9;
            sum_sq_q18 += d * d;
        }
        int32_t var_q18 = fxp_round_div_s32(sum_sq_q18, n_frames);
        if (var_q18 < 0) var_q18 = 0;

        uint8_t mel_bin = probe->idxs_needed[m];
        probe->mean_q9[mel_bin] = mean_q9;
        probe->std_q9[mel_bin] = fxp_sat_s16_from_s32((int32_t)fxp_sqrt32((uint32_t)var_q18));
        probe->max_q9[mel_bin] = row_max_q9;
        probe->entropy_q14[mel_bin] =
            _mel_entropy_row_q14(&probe->mel_power[(size_t)m * (size_t)n_frames], n_frames);
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
