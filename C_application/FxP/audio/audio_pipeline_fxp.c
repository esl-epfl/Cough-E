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
/* Probe version of the FFT front end. It exposes the same magnitudes,
 * frequencies, and magnitude sum used by audio_fft_features so the harness can
 * compare the fixed-point stages directly.
 */
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

    *sum_mags = (uq15_17_t)sum;

    free(sig_q);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

/* FFT feature block. It computes the shared RFFT once, then only runs the
 * feature kernels requested by features_selector.
 */
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
    // Below here are the RFFT kernel features.
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

    // Below are all the FFT feature kernel calls.
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

/* Composite Simpson's rule over an even number of intervals.
 * Proxy samples UQ21.11 are shifted down to integer units so the running sum
 * fits in u32 before the final divide-by-3.
 */
static uint32_t _psd_simpson_step(const uq21_11_t *x, int16_t start, int16_t end) {
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uint32_t sum = 0;

    for (int i = 0; i < n_intervals; i++) {
        uint32_t x0 = (uint32_t)(x[idx] >> 3U);
        uint32_t x1 = (uint32_t)(x[idx + 1] >> 3U);
        uint32_t x2 = (uint32_t)(x[idx + 2] >> 3U);
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
        uint32_t val = (((uint32_t)(x[len - 1] >> 3U) + (uint32_t)(x[len - 2] >> 3U)) + 1U) >> 1;
        uint32_t result = _psd_simpson_step(x, 0, len - 1);

        val += ((((uint32_t)(x[0] >> 3U) + (uint32_t)(x[1] >> 3U)) + 1U) >> 1);
        result += _psd_simpson_step(x, 1, len);

        val = (val + 1U) >> 1;
        result = (result + 1U) >> 1;
        return result + val;
    }

    return _psd_simpson_step(x, 0, len);
}

/* Dominant frequency: frequency corresponding to the maximum proxy value.
 * Proxy is UQ21.11, frequencies are UQ12.20, and the result is UQ12.20.
 */
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

/* Spectral flatness: exp(mean(log(power)) - log(mean(power))).
 * Proxy input is UQ21.11, logs are Q21.11, and the output ratio is UQ0.16.
 */
static uq0_16_t _flatness(const uq21_11_t *proxy, int16_t len) {
    q21_11_t sum_logs = 0;
    q21_11_t sum_proxy = 0;

    for (int16_t i = 0; i < len; i++) {
        uq21_11_t v = (proxy[i] == 0U) ? 1U : proxy[i];
        sum_logs += _log_psd(v);
        sum_proxy += v;
    }

    if (sum_proxy == 0U) return 0;

    q5_11_t mean_log = (q5_11_t)(sum_logs / (int32_t)len);
    q21_11_t mean_proxy = (q21_11_t)(sum_proxy / (uint32_t)len);
    if (mean_proxy == 0U) mean_proxy = 1U;

    q21_11_t log_mean_proxy = _log_psd(mean_proxy);
    q5_11_t diff = (q5_11_t)((q21_11_t)mean_log - log_mean_proxy);
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
        uq10_6_t ratio =
            (uq10_6_t)((((uint64_t)band_power << 16) + (total_power >> 1)) / total_power);
        band_powers[i] = ratio;
    }
}

/* Welch PSD feature block. The accumulated spectrum is kept as a proxy because
 * flatness, bandpower ratios, and dominant frequency do not need absolute PSD
 * normalization.
 */
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
    // timedata is Q2.30 for KissFFT's 32-bit twiddle path.
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

    // Below are all the PSD feature kernel calls.
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

#define FXP_MEL_BASIS_FRAC 15

// Natural-log to power-dB conversion: 10 / ln(10), stored as Q12.20.
#define DB_FROM_LN ((int32_t)4553913)
// Librosa-style top_db clamp: 80 dB, stored as Q7.9.
#define DB_80 ((int32_t)40960)

// dB offset in Q7.9. frame_power is UQ20.44 holding |X[k]|² / N² (the 1/N²
// comes from KissFFT's per-stage HALF_OF). _mel_db computes 10·log10 of the
// raw integer, which adds 44·10·log10(2) ≈ 132.45 dB; subtracting that and
// adding back 20·log10(N_FFT) ≈ 66.23 dB recovers the true power dB:
//   offset = (20·log10(N_FFT) - 44·10·log10(2)) · 2⁹ ≈ -33908.
#define STFT_DB_OFFSET_Q9 ((int32_t)-33908)
#define MEL_DB_OFFSET STFT_DB_OFFSET_Q9

static int _mel_any_required(const int8_t *features_selector) {
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT; i < ZERO_CROSSING_RATE; i++) {
        if (features_selector[i]) return 1;
    }
    return 0;
}

/* Convert mel power to Q7.9 dB, kept as int32_t for headroom. The pre-clip
 * dB range can reach ~-200 dB for very small powers, which overflows q7_9_t;
 */
static int32_t _mel_db(uint64_t power, int32_t offset) {
    q7_9_t ln_power = _log_mel_power(power);
    int32_t db = (int32_t)((((int64_t)ln_power * DB_FROM_LN)) >> 20);
    return db + offset;
}

/* Shannon entropy over one mel row. Probability is UQ0.16, ln(p) is Q7.9,
 * and p * -ln(p) is shifted back to Q2.14 for the feature output.
 */
static uint16_t _mel_entropy(const uq20_44_t *row_power, int16_t n_frames) {
    if (!row_power || n_frames <= 0) return 0;

    uint64_t sum = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        sum += row_power[t];
    }
    if (sum == 0) return 0;

    int32_t entropy = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        uint64_t power = row_power[t];
        if (power == 0 || power >= sum) continue;

        uint32_t p = 0;
        uint64_t rem = power;
        for (uint8_t bit = 0; bit < 16U; bit++) {
            p <<= 1U;
            if (rem >= (sum - rem)) {
                rem -= (sum - rem);
                p |= 1U;
            } else {
                rem += rem;
            }
        }
        if (rem >= (sum - rem) && p < UINT16_MAX) p++;
        if (p == 0U) continue;

        q7_9_t ln_p =
            (q7_9_t)((int32_t)_log_mel_power((uint64_t)p) -
                     (16 * ((FXP_LN2_Q24 + (1 << 14)) >> 15)));
        if (ln_p > 0) ln_p = 0;

        int32_t prod = (int32_t)p * (int32_t)(-ln_p);
        entropy += (prod + (1 << 10)) >> 11;
    }

    if (entropy <= 0) return 0;
    return (entropy > UINT16_MAX) ? UINT16_MAX : (uint16_t)entropy;
}

/* Mel feature block. It mirrors the floating path: reflect-pad, Hann-window,
 * RFFT each frame, apply sparse mel weights, convert to dB, clamp to top_db,
 * then emit mean/std/max/entropy per requested mel bin.
 */
void audio_mel_features(const int8_t *features_selector, const int16_t *sig, int16_t len,
                        fxp_feat_t *feats) {
    if (!features_selector || !sig || !feats || len <= PAD_LEN) return;
    if (!_mel_any_required(features_selector)) return;

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

    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    uq20_44_t *frame_power = (uq20_44_t *)malloc((size_t)FFT_RES_LEN * sizeof(uq20_44_t));
    uint64_t *frame_power_entropy = (uint64_t *)malloc((size_t)FFT_RES_LEN * sizeof(uint64_t));
    uint64_t *mel_entropy_power =
        (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));
    // Pre-clip dB values can reach ~-200 dB, so the temp buffer is widened to
    // int32 (still Q?.9). After the top_db clip, values fit back in q7_9_t.
    int32_t *mel_db = (int32_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int32_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!timedata || !cx_out || !frame_power || !frame_power_entropy || !mel_entropy_power ||
        !mel_db || !cfg) {
        free(timedata);
        free(cx_out);
        free(frame_power);
        free(frame_power_entropy);
        free(mel_entropy_power);
        free(mel_db);
        free(cfg);
        return;
    }

    int32_t max_db = INT32_MIN;
    for (int16_t f = 0; f < n_frames; f++) {
        int32_t frame_start = (int32_t)f * HOP_LEN;

        // Reflect-pad on the fly and apply the Q1.15 Hann window into Q1.31.
        for (int16_t n = 0; n < N_FFT; n++) {
            int32_t idx = frame_start + n;
            int16_t sample_idx;
            if (idx < PAD_LEN) {
                sample_idx = (int16_t)(PAD_LEN - idx);
            } else if (idx < PAD_LEN + len) {
                sample_idx = (int16_t)(idx - PAD_LEN);
            } else {
                sample_idx = (int16_t)(len - 2 - (idx - (PAD_LEN + len)));
            }

            q2_14_t sample = sig[sample_idx];
            int64_t windowed = (int64_t)sample * (int64_t)fxp_mfcc_hann_q15[n];
            timedata[n] = (kiss_fft_scalar)(windowed << 2);
        }

        kiss_fftr(cfg, timedata, cx_out);

        // Store frame power as UQ20.44 from the Q10.22 FFT bins.
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            q10_22_t re = (q10_22_t)(cx_out[k].r >> 9);
            q10_22_t im = (q10_22_t)(cx_out[k].i >> 9);
            uq20_44_t re_2 = (uq20_44_t)((int64_t)re * (int64_t)re);
            uq20_44_t im_2 = (uq20_44_t)((int64_t)im * (int64_t)im);
            uq20_44_t p = re_2 + im_2;
            frame_power[k] = p;

            // Entropy is sensitive to tiny frame powers, so keep the full
            // squared FFT bins for that path instead of the Q20.44 dB path.
            int64_t re_full = (int64_t)cx_out[k].r;
            int64_t im_full = (int64_t)cx_out[k].i;
            frame_power_entropy[k] =
                (uint64_t)(re_full * re_full) + (uint64_t)(im_full * im_full);
        }

        // Apply each sparse mel row. Basis weights are UQ1.15, so shift by 15.
        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uq20_44_t sum = 0;
            uint64_t entropy_sum = 0;
            for (int16_t k = start; k <= end; k++) {
                uq1_15_t w_q15 = fxp_mel_basis_q15[mel_idx][k - start];
                sum += (frame_power[k] * (uint64_t)w_q15) >> FXP_MEL_BASIS_FRAC;
                entropy_sum += ((frame_power_entropy[k] * (uint64_t)w_q15) + (1ULL << 6)) >> 7;
            }

            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int32_t db = _mel_db(sum, MEL_DB_OFFSET);
            mel_entropy_power[idx] = entropy_sum;
            mel_db[idx] = db;
            if (db > max_db) max_db = db;
        }
    }

    // Clamp the spectrogram to max_db - 80 dB before row statistics. After the
    // clamp, values fit in q7_9_t (post-clip range is at most 80 dB wide).
    int32_t clip = max_db - DB_80;

    for (int16_t m = 0; m < n_mels_needed; m++) {
        q21_11_t row_sum = 0;
        q7_9_t row_max = INT16_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int32_t db_wide = mel_db[idx];
            if (db_wide < clip) db_wide = clip;
            q7_9_t db = (q7_9_t)db_wide;
            mel_db[idx] = db_wide;
            row_sum += (q21_11_t)db << 2; // Q7.9 to Q21.11 for mean calculation
            if (db > row_max) row_max = db;
        }

        q7_9_t mean = (q7_9_t)((row_sum / n_frames) >> 2);

        // Variance is accumulated as Q18.14, then restored to Q14.18 for sqrt -> Q7.9.
        uq18_14_t sum = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            q7_9_t dev =
                (q7_9_t)((int32_t)mel_db[(size_t)m * (size_t)n_frames + (size_t)f] - (int32_t)mean);
            uq14_18_t dev_2 = (uq14_18_t)((int32_t)dev * (int32_t)dev);
            sum += (uq18_14_t)(dev_2 >> 4);
        }

        uq14_18_t var = (uq14_18_t)((sum / n_frames) << 4);
        q7_9_t std = (q7_9_t)fxp_sqrt32(var);

        uint32_t entrop =
            _mel_entropy(&mel_entropy_power[(size_t)m * (size_t)n_frames], n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] = (fxp_feat_t)mean;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] = (fxp_feat_t)std;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] = (fxp_feat_t)row_max;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] = (fxp_feat_t)entrop;
    }

    free(timedata);
    free(cx_out);
    free(frame_power);
    free(frame_power_entropy);
    free(mel_entropy_power);
    free(mel_db);
    free(cfg);
}

#if defined(FXP_STAGE_PROBES)
/* Free the probe-owned buffers. The caller owns the probe struct itself. */
void audio_mel_stage_probe_free(audio_mel_stage_probe_t *probe) {
    if (!probe) return;
    free(probe->frame_power);
    free(probe->mel_power);
    free(probe->mel_db_q9);
    probe->frame_power = NULL;
    probe->mel_power = NULL;
    probe->mel_db_q9 = NULL;
}

/* Probe version of the mel feature block. It follows audio_mel_features but
 * keeps frame power, mel power, and pre/post-clipped dB rows for the harness.
 */
int audio_mel_stage_probe(const int8_t *features_selector, const int16_t *sig, int16_t len,
                          audio_mel_stage_probe_t *probe) {
    if (!features_selector || !sig || !probe || len <= 0) return 0;

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
        (uq20_44_t *)malloc((size_t)n_frames * (size_t)FFT_RES_LEN * sizeof(uq20_44_t));
    probe->mel_power =
        (uq20_44_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uq20_44_t));
    probe->mel_db_q9 =
        (int32_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(int32_t));
    uint64_t *frame_power_entropy = (uint64_t *)malloc((size_t)FFT_RES_LEN * sizeof(uint64_t));
    uint64_t *mel_entropy_power =
        (uint64_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uint64_t));

    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!probe->frame_power || !probe->mel_power || !probe->mel_db_q9 || !frame_power_entropy ||
        !mel_entropy_power || !timedata || !cx_out || !cfg) {
        audio_mel_stage_probe_free(probe);
        free(frame_power_entropy);
        free(mel_entropy_power);
        free(timedata);
        free(cx_out);
        free(cfg);
        memset(probe, 0, sizeof(*probe));
        return 0;
    }

    probe->stft_db_offset_q9 = STFT_DB_OFFSET_Q9;
    probe->mel_db_offset_q9 = MEL_DB_OFFSET;

    int32_t max_db = INT32_MIN;
    for (int16_t f = 0; f < n_frames; f++) {
        int32_t frame_start = (int32_t)f * HOP_LEN;

        // Same reflect-pad and Hann-window front end as audio_mel_features.
        for (int16_t n = 0; n < N_FFT; n++) {
            int32_t idx = frame_start + n;
            int16_t sample_idx;
            if (idx < PAD_LEN) {
                sample_idx = (int16_t)(PAD_LEN - idx);
            } else if (idx < PAD_LEN + len) {
                sample_idx = (int16_t)(idx - PAD_LEN);
            } else {
                sample_idx = (int16_t)(len - 2 - (idx - (PAD_LEN + len)));
            }

            q2_14_t sample = sig[sample_idx];
            int64_t windowed = (int64_t)sample * (int64_t)fxp_mfcc_hann_q15[n];
            timedata[n] = (kiss_fft_scalar)(windowed << 2);
        }

        kiss_fftr(cfg, timedata, cx_out);

        uq20_44_t *frame_power = &probe->frame_power[(size_t)f * (size_t)FFT_RES_LEN];
        // Keep the per-frame STFT power visible to the stage harness.
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            q10_22_t re = (q10_22_t)(cx_out[k].r >> 9);
            q10_22_t im = (q10_22_t)(cx_out[k].i >> 9);
            uq20_44_t re_2 = (uq20_44_t)((int64_t)re * (int64_t)re);
            uq20_44_t im_2 = (uq20_44_t)((int64_t)im * (int64_t)im);
            uq20_44_t p = re_2 + im_2;
            frame_power[k] = p;

            // Preserve the full-power path for entropy while the probe keeps
            // the Q20.44 mel power used by the dB stage checks.
            int64_t re_full = (int64_t)cx_out[k].r;
            int64_t im_full = (int64_t)cx_out[k].i;
            frame_power_entropy[k] =
                (uint64_t)(re_full * re_full) + (uint64_t)(im_full * im_full);
        }

        // Apply the requested mel rows and keep both mel power and dB.
        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)probe->idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uq20_44_t sum = 0;
            uint64_t entropy_sum = 0;
            for (int16_t k = start; k <= end; k++) {
                uq1_15_t w_q15 = fxp_mel_basis_q15[mel_idx][k - start];
                sum += (frame_power[k] * (uint64_t)w_q15) >> FXP_MEL_BASIS_FRAC;
                entropy_sum += ((frame_power_entropy[k] * (uint64_t)w_q15) + (1ULL << 6)) >> 7;
            }

            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int32_t db = _mel_db(sum, MEL_DB_OFFSET);
            probe->mel_power[idx] = sum;
            mel_entropy_power[idx] = entropy_sum;
            probe->mel_db_q9[idx] = db;
            if (db > max_db) max_db = db;
        }
    }

    // Match audio_mel_features exactly for clipping and row statistics.
    int32_t clip = max_db - DB_80;

    for (int16_t m = 0; m < n_mels_needed; m++) {
        q21_11_t row_sum = 0;
        q7_9_t row_max = INT16_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            int32_t db_wide = probe->mel_db_q9[idx];
            if (db_wide < clip) db_wide = clip;
            q7_9_t db = (q7_9_t)db_wide;
            probe->mel_db_q9[idx] = db_wide;
            row_sum += (q21_11_t)db << 2; // Q7.9 to Q21.11 for mean calculation
            if (db > row_max) row_max = db;
        }

        q7_9_t mean = (q7_9_t)((row_sum / n_frames) >> 2);

        uq18_14_t sum = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            q7_9_t dev =
                (q7_9_t)((int32_t)probe->mel_db_q9[(size_t)m * (size_t)n_frames + (size_t)f] -
                         (int32_t)mean);
            uq14_18_t dev_2 = (uq14_18_t)((int32_t)dev * (int32_t)dev);
            sum += (uq18_14_t)(dev_2 >> 4);
        }
        uq14_18_t var = (uq14_18_t)((sum / n_frames) << 4);
        q7_9_t std = (q7_9_t)fxp_sqrt32(var);

        uint8_t mel_bin = probe->idxs_needed[m];
        probe->mean_q9[mel_bin] = mean;
        probe->std_q9[mel_bin] = std;
        probe->max_q9[mel_bin] = row_max;
        probe->entropy_q14[mel_bin] =
            _mel_entropy(&mel_entropy_power[(size_t)m * (size_t)n_frames], n_frames);
    }

    free(frame_power_entropy);
    free(mel_entropy_power);
    free(timedata);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

#endif
