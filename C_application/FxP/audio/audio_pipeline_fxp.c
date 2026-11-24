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
/*  Audio RMS and crest factor                                                */
/* -------------------------------------------------------------------------- */

// RMS for centered audio. Input samples are Q2.14; mean is Q18.14;
// output is UQ18.14. Squared energy is accumulated in UQ4.28 and
// downshifted by powers of two only when needed to avoid overflow.
static uq18_14_t _rms_audio(const q2_14_t *sig, int16_t len, q18_14_t mean) {
    if (len <= 0) return 0;

    uint8_t energy_shift = 0;
    uint16_t window_len = (uint16_t)len;
    uq4_28_t energy_sum = 0;
    int overflow = 0;
    do {
        energy_sum = 0;
        overflow = 0;
        for (int16_t i = 0; i < len; i++) {
            q18_14_t cur = (q18_14_t)sig[i] - mean;
            uq18_14_t abs_cur = (cur < 0) ? (uq18_14_t)(-cur) : (uq18_14_t)cur;
            uq4_28_t energy = (abs_cur * abs_cur) >> energy_shift;
            if (energy_sum > (UINT32_MAX - energy)) {
                overflow = 1;
                energy_shift += 2U;
                break;
            }
            energy_sum += energy;
        }
    } while (overflow && energy_shift < 28U);

    uq4_28_t mean_energy = energy_sum / window_len;
    return fxp_sqrt32(mean_energy) << (energy_shift >> 1U);
}

// Compute RMS and crest factor on DC-centered Q2.14 audio.
// Crest factor is positive peak over centered RMS, returned in pipeline Q format.
void audio_crest_factor(const int8_t *features_selector, const int16_t *sig, int16_t len,
                        fxp_feat_t *feats) {
    if ((!features_selector[ROOT_MEANS_SQUARED] && !features_selector[CREST_FACTOR]) || len <= 0) {
        return;
    }

    q18_14_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        sum += sig[i];
    }

    q18_14_t mean = sum / len;

    uq18_14_t rms = _rms_audio(sig, len, mean);
    if (features_selector[CREST_FACTOR]) {
        q18_14_t peak = (q18_14_t)sig[0] - mean;
        for (int16_t i = 1; i < len; i++) {
            q18_14_t cur = (q18_14_t)sig[i] - mean;
            if (cur > peak) peak = cur;
        }

        if (peak <= 0 || rms == 0U) return;

        feats[CREST_FACTOR] = (fxp_feat_t)(((uq18_14_t)peak << FXP_PIPE_FRAC) / rms);
    }
}
/* -------------------------------------------------------------------------- */
/*  FFT feature kernels + block                                               */
/* -------------------------------------------------------------------------- */

#define FFT_ROLLOFF ((uint32_t)62259U)

/* Frequency deviation from the spectral centroid.
 * Frequency is UQ12.20 and centroid is UQ11.21; both are converted to Q13.19
 * before subtracting.
 */
static inline q13_19_t _dev(uq12_20_t freq, uq11_21_t centroid) {
    q13_19_t freq_scaled = (q13_19_t)(freq >> 1);
    q13_19_t centroid_scaled = (q13_19_t)(centroid >> 2);
    return (q13_19_t)(freq_scaled - centroid_scaled);
}
/* Spectral rolloff: first frequency bin where the running magnitude reaches
 * 95% of the total magnitude. Magnitudes and their sum use UQ2.30.
 */
static uq12_20_t _rolloff(const uq4_28_t *mags, const uq12_20_t *freqs, int16_t len,
                          uq7_25_t sum_mags) {
    // Compute 95% as sum - round(sum / 20) to avoid a 32-bit overflow from
    // multiplying by 0.95 in Q0.16 before the final shift.
    uq7_25_t rolloff_energy = (uq7_25_t)(sum_mags - ((sum_mags + 10U) / 20U));

    uq7_25_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        sum += (mags[i] >> 3);
        if (sum >= rolloff_energy) {
            return freqs[i];
        }
    }
    return freqs[len - 1];
}

/* Spectral centroid: weighted mean frequency, sum(freq * magnitude) / sum(magnitude).
 * The product UQ12.20 * UQ4.28 is UQ16.48, then shifted to UQ18.46
 * before accumulation. Dividing by UQ7.25 produces UQ11.21.
 */
static uq11_21_t _centroid(const uq4_28_t *mags, const uq12_20_t *freqs, int16_t len,
                           uq7_25_t sum_mags) {

    uq18_46_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        uq16_48_t product = (uq16_48_t)((uint64_t)freqs[i] * (uint64_t)mags[i]);
        sum += (uq18_46_t)(product >> 2U);
    }

    uq11_21_t centroid = (uq11_21_t)(sum / sum_mags);
    return centroid;
}
/* Spectral spread: sqrt of the magnitude-weighted variance around centroid.
 * dev is Q13.19, so dev^2 starts as UQ26.38 and is shifted to UQ25.7.
 * Multiplying by magnitude UQ4.28 gives UQ29.35; dividing by sum_mags
 * UQ7.25 produces UQ22.10, whose sqrt is UQ11.5.
 */
static uq11_5_t _spread(const uq4_28_t *mags, const uq12_20_t *freqs, int16_t len,
                        uq7_25_t sum_mags, uq11_21_t centroid) {

    uq29_35_t sum = 0;
    for (int16_t i = 0; i < len; i++) {
        q13_19_t dev = _dev(freqs[i], centroid);
        uq25_7_t dev_2 = (uq25_7_t)(((int64_t)dev * (int64_t)dev) >> 31);
        sum += (uq29_35_t)((uint64_t)dev_2 * (uint64_t)mags[i]);
    }

    uq22_10_t mean = (uq22_10_t)(sum / (uint64_t)sum_mags);
    return (uq11_5_t)fxp_sqrt32(mean);
}
/* Spectral kurtosis: magnitude-weighted mean of normalized fourth powers.
 * inv_spread is UQ12.20 and dev is Q13.19; their product is shifted to Q5.11.
 * Squaring twice gives UQ10.22 then UQ20.12, which is weighted by magnitude
 * UQ4.28 and divided by sum_mags UQ7.25 to produce UQ17.15.
 */
static uq17_15_t _kurtosis(const uq4_28_t *mags, const uq12_20_t *freqs, int16_t len,
                           uq7_25_t sum_mags, uq11_21_t centroid, uq11_5_t spread) {
    // 2^25 / UQ11.5 gives an inverse spread in UQ12.20.
    uq12_20_t inv_spread = (uq12_20_t)((1U << 25) / (uint32_t)spread);
    uq22_40_t sum = 0;
    for (int16_t i = 0; i < len; i++) {

        q13_19_t dev = _dev(freqs[i], centroid);

        q5_11_t norm = (q5_11_t)(((int64_t)dev * (int64_t)inv_spread) >> 28U);
        uq10_22_t norm_2 = (uq10_22_t)((int32_t)norm * (int32_t)norm);
        uq20_12_t norm_4 = (uq20_12_t)(((uint64_t)norm_2 * (uint64_t)norm_2) >> 32);
        sum += (uq22_40_t)((uint64_t)norm_4 * (uint64_t)mags[i]);
    }

    uq17_15_t kurtosis = (uq17_15_t)((sum) / (uint64_t)sum_mags);
    return kurtosis;
}

#if defined(FXP_STAGE_PROBES)
/* Probe version of the FFT front end. It exposes the same magnitudes,
 * frequencies, and magnitude sum used by audio_fft_features so the harness can
 * compare the fixed-point stages directly.
 */
int audio_fft_stage_probe(const int16_t *sig, int16_t len, int16_t fs, uq4_28_t *mags,
                          uq12_20_t *freqs, uq7_25_t *sum_mags) {

    int16_t fft_len = (int16_t)((len / 2) + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    if (!cfg) return 0;

    kiss_fft_scalar *fft_in = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));

    if (!fft_in || !cx_out) {
        free(fft_in);
        free(cx_out);
        free(cfg);
        return 0;
    }

    for (int16_t i = 0; i < len; i++) {
        fft_in[i] = (kiss_fft_scalar)((int32_t)sig[i] << 16);
    }

    kiss_fftr(cfg, fft_in, cx_out);

    uq7_25_t sum = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        // Fixed KissFFT is scaled by 1/len; the observed 6400-point FFT bins
        // fit in Q4.28, preserving 8 more fractional bits for weak bins.
        q4_28_t re = (q4_28_t)(cx_out[i].r >> 2);
        q4_28_t im = (q4_28_t)(cx_out[i].i >> 2);
        uq8_56_t re_2 = (uq8_56_t)((int64_t)re * (int64_t)re);
        uq8_56_t im_2 = (uq8_56_t)((int64_t)im * (int64_t)im);
        mags[i] = (uq4_28_t)fxp_sqrt64(re_2 + im_2);
        sum += (mags[i] >> 3);

        freqs[i] = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20) / (uint64_t)len);
    }

    *sum_mags = (uq7_25_t)sum;

    free(fft_in);
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

    kiss_fft_scalar *fft_in = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));
    uq4_28_t *mags = (uq4_28_t *)malloc((size_t)fft_len * sizeof(uq4_28_t));
    uq12_20_t *freqs = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));

    if (!fft_in || !cx_out || !mags || !freqs) {
        free(fft_in);
        free(cx_out);
        free(mags);
        free(freqs);
        free(cfg);
        return;
    }
    // KissFFT uses kiss_fft_scalar for both samples and twiddles; in fixed-point
    // mode this pipeline treats it as Q2.30, so Q2.14 audio is widened here.
    for (int16_t i = 0; i < len; i++) {
        fft_in[i] = (kiss_fft_scalar)((int32_t)sig[i] << 16);
    }

    kiss_fftr(cfg, fft_in, cx_out);

    // Convert the shared RFFT output into magnitudes and frequency bins used by
    // rolloff, centroid, spread, and kurtosis.
    uq7_25_t sum = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        // Fixed KissFFT is scaled by 1/len; the observed 6400-point FFT bins
        // fit in Q4.28, preserving 8 more fractional bits for weak bins.
        q4_28_t re = (q4_28_t)(cx_out[i].r >> 2);
        q4_28_t im = (q4_28_t)(cx_out[i].i >> 2);
        uq8_56_t re_2 = (uq8_56_t)((int64_t)re * (int64_t)re);
        uq8_56_t im_2 = (uq8_56_t)((int64_t)im * (int64_t)im);
        mags[i] = (uq4_28_t)fxp_sqrt64(re_2 + im_2);
        sum += (mags[i] >> 3);

        freqs[i] = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20) / (uint64_t)len);
    }

    uq7_25_t sum_mags = (uq7_25_t)sum;
    if (sum_mags == 0U) {
        free(fft_in);
        free(cx_out);
        free(mags);
        free(freqs);
        free(cfg);
        return;
    }

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

    if (need_kurt && spread != 0U) {
        uq17_15_t kurtosis = _kurtosis(mags, freqs, fft_len, sum_mags, centroid, spread);
        feats[SPECTRAL_KURTOSIS] = (fxp_feat_t)kurtosis;
    }

    free(fft_in);
    free(cx_out);
    free(mags);
    free(freqs);
    free(cfg);
}

/* -------------------------------------------------------------------------- */
/*  Periodogram kernels + block                                               */
/* -------------------------------------------------------------------------- */

/* Composite Simpson integration for the widened PSD proxy. Input samples are
 * UQ9.55 and are first converted to a UQ12.52 integration proxy. An additional
 * dynamic right shift can be applied to every sample before accumulation; only
 * ratios consume the result, so this extra shift cancels between numerator and
 * denominator.
 */
static uint8_t _psd_simpson_extra_shift(const uq9_55_t *x, int16_t len) {
    uq12_52_t max_val = 0;
    for (int16_t i = 0; i < len; i++) {
        uq12_52_t sample = (uq12_52_t)(x[i] >> 3U);
        if (sample > max_val) max_val = sample;
    }

    uint64_t coeff_bound = ((uint64_t)len * 3U) + 8U;
    if (coeff_bound == 0U) coeff_bound = 1U;
    uq12_52_t max_term = (uq12_52_t)((UINT64_MAX >> 16U) / coeff_bound);

    uint8_t extra_shift = 0;
    while (extra_shift < 63U && (max_val >> extra_shift) > max_term) {
        extra_shift++;
    }
    return extra_shift;
}

static uq12_52_t _psd_sample_for_integral(uq9_55_t x, uint8_t extra_shift) {
    uq12_52_t sample = (uq12_52_t)(x >> 3U);
    return (uq12_52_t)(sample >> extra_shift);
}

static uq12_52_t _psd_simpson_step(const uq9_55_t *x, int16_t start, int16_t end,
                                   uint8_t extra_shift) {
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uq12_52_t sum = 0;

    for (int i = 0; i < n_intervals; i++) {
        uq12_52_t x0 = _psd_sample_for_integral(x[idx], extra_shift);
        uq12_52_t x1 = _psd_sample_for_integral(x[idx + 1], extra_shift);
        uq12_52_t x2 = _psd_sample_for_integral(x[idx + 2], extra_shift);
        sum += x0 + (x1 << 2) + x2;
        idx += 2;
    }

    return (uq12_52_t)((sum + 1U) / 3U);
}

static uq12_52_t _psd_simpson(const uq9_55_t *x, int16_t len, uint8_t extra_shift) {
    if (!x || len <= 1) return 0U;

    if ((len & 1) == 0) {
        uq12_52_t val = (_psd_sample_for_integral(x[len - 1], extra_shift) +
                         _psd_sample_for_integral(x[len - 2], extra_shift) + 1U) >>
                        1U;
        uq12_52_t result = _psd_simpson_step(x, 0, len - 1, extra_shift);

        val += (_psd_sample_for_integral(x[0], extra_shift) +
                _psd_sample_for_integral(x[1], extra_shift) + 1U) >>
               1U;
        result += _psd_simpson_step(x, 1, len, extra_shift);

        val = (val + 1U) >> 1;
        result = (result + 1U) >> 1;
        return (uq12_52_t)(result + val);
    }

    return _psd_simpson_step(x, 0, len, extra_shift);
}

static uq0_16_t _unit_ratio(uq12_52_t numerator, uq12_52_t denominator) {
    if (denominator == 0U || numerator == 0U) return 0;
    if (numerator >= denominator) return UINT16_MAX;

    uq0_16_t ratio = 0;
    uq12_52_t rem = numerator;
    for (uint8_t bit = 0; bit < 16U; bit++) {
        ratio = (uq0_16_t)(ratio << 1U);
        if (rem >= (denominator - rem)) {
            rem -= (denominator - rem);
            ratio |= 1U;
        } else {
            rem += rem;
        }
    }

    if (rem >= (denominator - rem) && ratio < UINT16_MAX) ratio++;
    return ratio;
}

/* Dominant frequency: frequency corresponding to the maximum proxy value.
 * Proxy is UQ9.23, frequencies are UQ12.20, and the result is UQ12.20.
 */
static uq12_20_t _dominant_freq(const uq9_23_t *proxy, const uq12_20_t *freqs, int16_t len) {
    int16_t max_idx = 0;
    uq9_23_t max_val = proxy[0];
    for (int16_t i = 1; i < len; i++) {
        if (proxy[i] > max_val) {
            max_val = proxy[i];
            max_idx = i;
        }
    }
    return freqs[max_idx];
}

static q2_30_t _psd_window_sample(q2_14_t sample, q2_14_t mean, uq1_15_t window) {
    q2_14_t centered = (q2_14_t)(sample - mean);
    q3_29_t product = (q3_29_t)((int32_t)centered * (int32_t)window);
    int64_t windowed = (int64_t)product << 1U;
    if (windowed > INT32_MAX) return INT32_MAX;
    if (windowed < INT32_MIN) return INT32_MIN;
    return (q2_30_t)windowed;
}

static uq9_55_t _psd_power_proxy(q2_30_t re, q2_30_t im, uint8_t double_sided) {
    uq4_60_t re_sq = (uq4_60_t)((int64_t)re * (int64_t)re);
    uq4_60_t im_sq = (uq4_60_t)((int64_t)im * (int64_t)im);
    uq4_60_t magnitude = (re_sq > (UINT64_MAX - im_sq)) ? UINT64_MAX : (uq4_60_t)(re_sq + im_sq);
    uq9_55_t power = (uq9_55_t)((magnitude + (1ULL << 4U)) >> 5U);

    if (double_sided) {
        power = (power > (UINT64_MAX >> 1U)) ? UINT64_MAX : (uq9_55_t)(power << 1U);
    }
    return power;
}

/* Spectral flatness from the widened PSD proxy. Power is UQ9.55 and each log
 * arrives as Q21.11. The accumulator promotes logs to Q30.34 so it keeps a
 * complete 64-bit format without wasting integer headroom. The mean log is
 * converted back to Q21.11 before the final log-domain difference, which is
 * clamped to Q5.11 for exp. The result is UQ0.16.
 */
static uq0_16_t _flatness(const uq9_55_t *proxy, int16_t len) {
    if (!proxy || len <= 0) return 0;

    uq9_55_t max_proxy = 0;
    for (int16_t i = 0; i < len; i++) {
        if (proxy[i] > max_proxy) max_proxy = proxy[i];
    }
    if (max_proxy == 0U) return 0;

    q30_34_t sum_logs = 0;
    uq9_55_t mean_proxy = 0;

    for (int16_t i = 0; i < len; i++) {
        uq9_55_t v = proxy[i];
        if (v == 0U) v = 1U;
        sum_logs += (q30_34_t)_log_psd(v, 55U) << 23U;

        uint64_t n = (uint64_t)i + 1U;
        if (v >= mean_proxy) {
            mean_proxy += (v - mean_proxy) / n;
        } else {
            mean_proxy -= (mean_proxy - v) / n;
        }
    }

    if (mean_proxy == 0U) return 0;

    q30_34_t mean_log_wide = (q30_34_t)(sum_logs / (int64_t)len);
    q21_11_t mean_log = (mean_log_wide >= 0)
                             ? (q21_11_t)((mean_log_wide + (1LL << 22)) >> 23U)
                             : (q21_11_t)(-(((-mean_log_wide) + (1LL << 22)) >> 23U));
    q21_11_t log_mean = _log_psd(mean_proxy, 55U);
    q21_11_t diff = (q21_11_t)(mean_log - log_mean);
    if (diff > 0) diff = 0;
    if (diff < INT16_MIN) diff = INT16_MIN;
    return _exp_psd((q5_11_t)diff);
}

/* Per-band relative power: integral of the proxy over each [start, end] band
 * divided by the total proxy integral. Output ratios are UQ0.16 so each band
 * is expressed as a fraction of total power.
 */
static void _bandpowers(const uq9_55_t *proxy, const uq12_20_t *freqs, int16_t len,
                        const int8_t *psd_selector, uq0_16_t *band_powers) {
    if (!band_powers) return;
    for (int8_t i = 0; i < N_PSD; i++)
        band_powers[i] = 0;

    if (!proxy || !freqs || !psd_selector || len <= 2) return;

    uint8_t integral_extra_shift = _psd_simpson_extra_shift(proxy, len);
    uq12_52_t total_power = _psd_simpson(proxy, len, integral_extra_shift);
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

        uq12_52_t band_power = _psd_simpson(&proxy[start_idx], n_bins, integral_extra_shift);
        band_powers[i] = _unit_ratio(band_power, total_power);
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
    uq9_23_t *acc_power = (uq9_23_t *)malloc((size_t)psd_len * sizeof(uq9_23_t));
    uq9_55_t *acc_power_wide = (need_flatness || need_bandpowers)
                                   ? (uq9_55_t *)malloc((size_t)psd_len * sizeof(uq9_55_t))
                                   : NULL;
    uq12_20_t *freqs = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(NPERSEG, 0, 0, 0);

    if (!timedata || !cx_out || !acc_power ||
        ((need_flatness || need_bandpowers) && !acc_power_wide) || !freqs || !cfg) {
        free(timedata);
        free(cx_out);
        free(acc_power);
        free(acc_power_wide);
        free(freqs);
        free(cfg);
        return;
    }

    memset(acc_power, 0, (size_t)psd_len * sizeof(uq9_23_t));
    if (acc_power_wide) {
        memset(acc_power_wide, 0, (size_t)psd_len * sizeof(uq9_55_t));
    }

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
            timedata[i] = (kiss_fft_scalar)_psd_window_sample(sig[start + i], mean_sig,
                                                              fxp_hann_window_q15[i]);
        }

        kiss_fftr(cfg, timedata, cx_out);

        // Square fixed KissFFT bins in Q2.30 once. The widened UQ9.55 proxy
        // feeds flatness/bandpowers; the compact UQ9.23 proxy is derived from
        // it for dominant frequency.
        for (int16_t i = 0; i < psd_len; i++) {
            uint8_t double_sided = (i != 0 && i != (NPERSEG / 2));

            q2_30_t re = (q2_30_t)cx_out[i].r;
            q2_30_t im = (q2_30_t)cx_out[i].i;
            uq9_55_t wide_power = _psd_power_proxy(re, im, double_sided);
            uq9_23_t power = (uq9_23_t)(wide_power >> 32);
            acc_power[i] = (power > (UINT32_MAX - acc_power[i]))
                               ? UINT32_MAX
                               : (uq9_23_t)(acc_power[i] + power);

            if (acc_power_wide) {
                acc_power_wide[i] = (wide_power > (UINT64_MAX - acc_power_wide[i]))
                                        ? UINT64_MAX
                                        : (uq9_55_t)(acc_power_wide[i] + wide_power);
            }
        }

        start = (int16_t)(start + hop);
    }

    // The normalization factors 1/(fs*sum(window^2)) and 1/n_segments cancel
    // in flatness and bandpower ratios and do not affect dominant-frequency
    // argmax.
    for (int16_t i = 0; i < psd_len; i++) {
        uq12_20_t freq = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20U) / (uint64_t)NPERSEG);
        freqs[i] = freq;
    }

    // Below are all the PSD feature kernel calls.
    if (need_dom_freq) {
        uq12_20_t dom = _dominant_freq(acc_power, freqs, psd_len);
        feats[DOMINANT_FREQUENCY] = (fxp_feat_t)dom;
    }

    if (need_flatness) {
        uq0_16_t flatness = _flatness(acc_power_wide, psd_len);
        feats[SPECTRAL_FLATNESS] = (fxp_feat_t)flatness;
    }

    if (need_bandpowers) {
        uq0_16_t band_powers[N_PSD] = {0};
        _bandpowers(acc_power_wide, freqs, psd_len, &features_selector[POWER_SPECTRAL_DENSITY],
                    band_powers);
        for (int8_t i = 0; i < N_PSD; i++) {
            if (features_selector[POWER_SPECTRAL_DENSITY + i]) {
                feats[POWER_SPECTRAL_DENSITY + i] = (fxp_feat_t)band_powers[i];
            }
        }
    }

    free(timedata);
    free(cx_out);
    free(acc_power);
    free(acc_power_wide);
    free(freqs);
    free(cfg);
}

#if defined(FXP_STAGE_PROBES)
int audio_psd_stage_probe(const int16_t *sig, int16_t sig_len, int16_t fs, uq9_23_t *acc_power,
                          uq12_20_t *freqs, int16_t *steps_out) {
    if (!sig || !acc_power || !freqs || sig_len < NPERSEG || fs <= 0) return 0;

    const int16_t psd_len = (int16_t)((NPERSEG / 2) + 1);
    const int16_t hop = (int16_t)(NPERSEG - NOVERLAP);
    if (hop <= 0) return 0;

    int16_t steps = (int16_t)((sig_len - NOVERLAP) / hop);
    if (steps <= 0) steps = 1;
    if (steps_out) *steps_out = steps;

    kiss_fft_scalar *timedata =
        (kiss_fft_scalar *)malloc((size_t)NPERSEG * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)psd_len * sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(NPERSEG, 0, 0, 0);

    if (!timedata || !cx_out || !cfg) {
        free(timedata);
        free(cx_out);
        free(cfg);
        return 0;
    }

    memset(acc_power, 0, (size_t)psd_len * sizeof(uq9_23_t));

    int16_t start = 0;
    for (int16_t step = 0; step < steps; step++) {
        if ((int32_t)start + NPERSEG > sig_len) break;
        int32_t sum_sig = 0;
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_sig += (int32_t)sig[start + i];
        }
        q2_14_t mean_sig = (q2_14_t)(sum_sig / (int32_t)NPERSEG);

        for (int16_t i = 0; i < NPERSEG; i++) {
            timedata[i] = (kiss_fft_scalar)_psd_window_sample(sig[start + i], mean_sig,
                                                              fxp_hann_window_q15[i]);
        }

        kiss_fftr(cfg, timedata, cx_out);

        for (int16_t i = 0; i < psd_len; i++) {
            q2_30_t re = (q2_30_t)cx_out[i].r;
            q2_30_t im = (q2_30_t)cx_out[i].i;
            uq9_55_t wide_power = _psd_power_proxy(re, im, (i != 0 && i != (NPERSEG / 2)));
            uq9_23_t power = (uq9_23_t)(wide_power >> 32);
            if (power > (UINT32_MAX - acc_power[i])) {
                acc_power[i] = UINT32_MAX;
            } else {
                acc_power[i] += power;
            }
        }

        start = (int16_t)(start + hop);
    }

    for (int16_t i = 0; i < psd_len; i++) {
        freqs[i] = (uq12_20_t)((((uint64_t)i * (uint64_t)fs) << 20U) / (uint64_t)NPERSEG);
    }

    free(timedata);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

/* -------------------------------------------------------------------------- */
/*  Mel feature block                                                         */
/* -------------------------------------------------------------------------- */

#define FXP_MEL_BASIS_FRAC 15

// Natural-log to power-dB conversion: 10 / ln(10), stored as Q12.20.
#define DB_FROM_LN ((q12_20_t)4553913)
// top_db clamp: 80 dB, stored as Q23.9.
#define DB_80 ((q23_9_t)40960)

// dB offset in Q7.9. Baselines:
//   frame_power UQ20.44, offset ≈ -33908.
//   frame_power UQ16.48, offset ≈ -40073.
// Current cx_out >> 2 stores FFT bins as Q4.28, so frame_power/mel_power are
// UQ8.56. Fixed KissFFT contributes a 1/N_FFT scale; _mel_db logs the raw
// integer, so compensate with:
//   offset = (20*log10(N_FFT) - 56*10*log10(2)) * 2^9 ≈ -52403.
#define STFT_DB_OFFSET ((q23_9_t) - 52403)
#define MEL_DB_OFFSET STFT_DB_OFFSET
#define LN2 ((q7_9_t)((FXP_LN2_Q24 + (1 << 14)) >> 15))

static int _mel_any_required(const int8_t *features_selector) {
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT; i < ZERO_CROSSING_RATE; i++) {
        if (features_selector[i]) return 1;
    }
    return 0;
}

/* Convert mel power to dB in Q23.9. _log_mel_power returns ln(raw_power) in
 * Q7.9; multiplying by 10/log(10) in Q12.20 gives Q19.29, then shifting by
 * 20 leaves a dB value with 9 fractional bits. This needs 32 bits before the
 * top_db clamp because low-energy bins plus the fixed offset can exceed q7_9_t.
 */
static q23_9_t _mel_db(uq8_56_t power, q23_9_t offset) {
    q7_9_t ln_power = _log_mel_power(power);
    int64_t scaled_db = (int64_t)ln_power * (int64_t)DB_FROM_LN;
    q23_9_t db = (q23_9_t)((scaled_db + (1LL << 19)) >> 20);
    return (q23_9_t)(db + offset);
}

static uq8_56_t _mel_weighted_power(uq8_56_t power, uq1_15_t weight) {
    if (power == 0U || weight == 0U) return 0U;

    uint64_t shifted_power = power >> FXP_MEL_BASIS_FRAC;
    if ((power & ((1ULL << FXP_MEL_BASIS_FRAC) - 1ULL)) >=
        (1ULL << (FXP_MEL_BASIS_FRAC - 1U))) {
        shifted_power++;
    }

    if (shifted_power > (UINT64_MAX / (uint64_t)weight)) return UINT64_MAX;
    return (uq8_56_t)(shifted_power * (uint64_t)weight);
}

/* Shannon entropy over one mel row. Probability is UQ0.16, ln(p) is Q7.9,
 * and p * -ln(p) is shifted back to Q2.14 for the feature output.
 */
static uq2_14_t _mel_entropy(const uq8_56_t *row_power, int16_t n_frames) {
    if (!row_power || n_frames <= 0) return 0;

    uq8_56_t row_sum = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        uq8_56_t power = row_power[t];
        row_sum = (power > (UINT64_MAX - row_sum)) ? UINT64_MAX : (uq8_56_t)(row_sum + power);
    }
    if (row_sum == 0U) return 0;

    uq2_14_t entropy = 0;
    for (int16_t t = 0; t < n_frames; t++) {
        uq8_56_t power = row_power[t];
        if (power == 0U || power >= row_sum) continue;

        // Long-division of power/sum into a UQ0.16 probability without a
        // 64-bit divide. Each iteration doubles the remainder and tests
        // whether 2*rem >= sum (written as rem >= sum-rem to avoid overflow);
        // a final compare provides round-to-nearest on the discarded bit.
        uq0_16_t p = 0;
        uq8_56_t rem = power;
        for (uint8_t bit = 0; bit < 16U; bit++) {
            p = (uq0_16_t)(p << 1U);
            if (rem >= (row_sum - rem)) {
                rem -= (row_sum - rem);
                p |= 1U;
            } else {
                rem += rem;
            }
        }
        if (rem >= (row_sum - rem) && p < UINT16_MAX) {
            p++;
        }
        if (p == 0U) continue;

        // _log_mel_power returns ln(p_raw) in Q7.9 where p_raw is a UQ0.16
        // integer (so p_raw = p * 2^16). Subtract 16*ln(2) to recover the
        // ln(p) of the true UQ0.16 probability. Numerical noise can push
        // ln_p slightly positive when p ≈ 1; clamp to 0 so -ln_p stays >= 0.
        q7_9_t ln_p = (q7_9_t)((int32_t)_log_mel_power((uint64_t)p) - (16 * (int32_t)LN2));
        if (ln_p > 0) ln_p = 0;

        // p * -ln_p is UQ0.16 * Q7.9 = Q7.25; shift by 11 with rounding to
        // accumulate the row entropy in Q2.14.
        uq7_25_t prod = (uq7_25_t)((uint32_t)p * (uint32_t)(-ln_p));
        uint32_t term = (prod + (1U << 10)) >> 11;
        entropy =
            (term > (uint32_t)(UINT16_MAX - entropy)) ? UINT16_MAX : (uq2_14_t)(entropy + term);
    }

    return entropy;
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
    uq8_56_t *frame_power = (uq8_56_t *)malloc((size_t)FFT_RES_LEN * sizeof(uq8_56_t));
    uq8_56_t *mel_entropy_power =
        (uq8_56_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uq8_56_t));
    q23_9_t *mel_db = (q23_9_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(q23_9_t));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!timedata || !cx_out || !frame_power || !mel_entropy_power || !mel_db || !cfg) {
        free(timedata);
        free(cx_out);
        free(frame_power);
        free(mel_entropy_power);
        free(mel_db);
        free(cfg);
        return;
    }

    q23_9_t max_db = INT32_MIN;
    for (int16_t f = 0; f < n_frames; f++) {
        int32_t frame_start = (int32_t)f * HOP_LEN;

        // Reflect-pad on the fly and apply the Q1.15 Hann window into Q2.30.
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
            timedata[n] = (kiss_fft_scalar)(windowed << 1);
        }

        kiss_fftr(cfg, timedata, cx_out);

        // Fixed KissFFT is scaled by 1/N_FFT, so Q4.28 keeps more fractional
        // precision while leaving headroom for the 64-bit mel-weight product.
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            q4_28_t re = (q4_28_t)(cx_out[k].r >> 2);
            q4_28_t im = (q4_28_t)(cx_out[k].i >> 2);
            uq8_56_t re_2 = (uq8_56_t)((int64_t)re * (int64_t)re);
            uq8_56_t im_2 = (uq8_56_t)((int64_t)im * (int64_t)im);
            uq8_56_t p = re_2 + im_2;
            frame_power[k] = p;
        }

        // Apply each sparse mel row. Basis weights are UQ1.15, so shift by 15.
        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uq8_56_t sum = 0;

            for (int16_t k = start; k <= end; k++) {
                uq1_15_t weight = fxp_mel_basis_q15[mel_idx][k - start];
                uq8_56_t term = _mel_weighted_power(frame_power[k], weight);
                sum = (term > (UINT64_MAX - sum)) ? UINT64_MAX : (uq8_56_t)(sum + term);
            }

            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            q23_9_t db = _mel_db(sum, MEL_DB_OFFSET);
            mel_entropy_power[idx] = sum;
            mel_db[idx] = db;
            if (db > max_db) max_db = db;
        }
    }

    // Clamp the spectrogram to max_db - 80 dB before row statistics. After the
    // clamp, values fit in q7_9_t (post-clip range is at most 80 dB wide).
    q23_9_t clip = (q23_9_t)(max_db - DB_80);

    for (int16_t m = 0; m < n_mels_needed; m++) {
        q21_11_t row_sum = 0;
        q7_9_t row_max = INT16_MIN;

        // First pass: clamp to the top_db floor and gather sum + max in one
        // sweep. Promoting to Q21.11 keeps headroom for n_frames * Q7.9.
        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            q7_9_t db = (q7_9_t)((mel_db[idx] < clip) ? clip : mel_db[idx]);
            mel_db[idx] = db;
            row_sum += (q21_11_t)db << 2; // Q7.9 to Q21.11 for mean calculation
            if (db > row_max) row_max = db;
        }

        q7_9_t mean = (q7_9_t)((row_sum / n_frames) >> 2);

        // Variance is accumulated as Q18.14, then restored to Q14.18 for sqrt -> Q7.9.
        // dev is Q7.9, so dev^2 is UQ14.18 and is shifted to UQ18.14 before
        // accumulation so the running sum cannot overflow.
        uq18_14_t sum = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            q7_9_t dev =
                (q7_9_t)((int32_t)mel_db[(size_t)m * (size_t)n_frames + (size_t)f] - (int32_t)mean);
            uq14_18_t dev_2 = (uq14_18_t)((int32_t)dev * (int32_t)dev);
            sum += (uq18_14_t)(dev_2 >> 4);
        }

        uq14_18_t var = (uq14_18_t)((sum / n_frames) << 4);
        q7_9_t std = (q7_9_t)fxp_sqrt32(var);

        uq2_14_t entrop = _mel_entropy(&mel_entropy_power[(size_t)m * (size_t)n_frames], n_frames);

        int16_t mel_bin = idxs_needed[m];
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin] = (fxp_feat_t)mean;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin] = (fxp_feat_t)std;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin] = (fxp_feat_t)row_max;
        feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin] = (fxp_feat_t)entrop;
    }

    free(timedata);
    free(cx_out);
    free(frame_power);
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
    free(probe->mel_db);
    probe->frame_power = NULL;
    probe->mel_power = NULL;
    probe->mel_db = NULL;
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
        (uq8_56_t *)malloc((size_t)n_frames * (size_t)FFT_RES_LEN * sizeof(uq8_56_t));
    probe->mel_power =
        (uq8_56_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uq8_56_t));
    probe->mel_db =
        (q23_9_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(q23_9_t));
    uq8_56_t *mel_entropy_power =
        (uq8_56_t *)malloc((size_t)n_mels_needed * (size_t)n_frames * sizeof(uq8_56_t));

    kiss_fft_scalar *timedata = (kiss_fft_scalar *)malloc((size_t)N_FFT * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)FFT_RES_LEN * sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0, 0, 0);

    if (!probe->frame_power || !probe->mel_power || !probe->mel_db || !mel_entropy_power ||
        !timedata || !cx_out || !cfg) {
        audio_mel_stage_probe_free(probe);
        free(mel_entropy_power);
        free(timedata);
        free(cx_out);
        free(cfg);
        memset(probe, 0, sizeof(*probe));
        return 0;
    }

    probe->stft_db_offset = STFT_DB_OFFSET;
    probe->mel_db_offset = MEL_DB_OFFSET;
    probe->frame_power_frac_bits = 56U;
    probe->mel_power_frac_bits = 56U;

    q23_9_t max_db = INT32_MIN;
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
            timedata[n] = (kiss_fft_scalar)(windowed << 1);
        }

        kiss_fftr(cfg, timedata, cx_out);

        uq8_56_t *frame_power = &probe->frame_power[(size_t)f * (size_t)FFT_RES_LEN];
        // Keep the per-frame STFT power visible to the stage harness.
        for (int16_t k = 0; k < FFT_RES_LEN; k++) {
            q4_28_t re = (q4_28_t)(cx_out[k].r >> 2);
            q4_28_t im = (q4_28_t)(cx_out[k].i >> 2);
            uq8_56_t re_2 = (uq8_56_t)((int64_t)re * (int64_t)re);
            uq8_56_t im_2 = (uq8_56_t)((int64_t)im * (int64_t)im);
            uq8_56_t p = re_2 + im_2;
            frame_power[k] = p;
        }

        // Apply the requested mel rows and keep both mel power and dB.
        for (int16_t m = 0; m < n_mels_needed; m++) {
            int16_t mel_idx = (int16_t)probe->idxs_needed[m];
            int16_t start = fxp_mel_nz_indexes[mel_idx][0];
            int16_t end = fxp_mel_nz_indexes[mel_idx][1];

            uq8_56_t sum = 0;
            for (int16_t k = start; k <= end; k++) {
                uq1_15_t weight = fxp_mel_basis_q15[mel_idx][k - start];
                uq8_56_t term = _mel_weighted_power(frame_power[k], weight);
                sum = (term > (UINT64_MAX - sum)) ? UINT64_MAX : (uq8_56_t)(sum + term);
            }

            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            q23_9_t db = _mel_db(sum, MEL_DB_OFFSET);
            probe->mel_power[idx] = sum;
            mel_entropy_power[idx] = sum;
            probe->mel_db[idx] = db;
            if (db > max_db) max_db = db;
        }
    }

    // Match audio_mel_features exactly for clipping and row statistics.
    q23_9_t clip = (q23_9_t)(max_db - DB_80);

    for (int16_t m = 0; m < n_mels_needed; m++) {
        q21_11_t row_sum = 0;
        q7_9_t row_max = INT16_MIN;

        for (int16_t f = 0; f < n_frames; f++) {
            size_t idx = (size_t)m * (size_t)n_frames + (size_t)f;
            q7_9_t db = (q7_9_t)((probe->mel_db[idx] < clip) ? clip : probe->mel_db[idx]);
            probe->mel_db[idx] = db;
            row_sum += (q21_11_t)db << 2; // Q7.9 to Q21.11 for mean calculation
            if (db > row_max) row_max = db;
        }

        q7_9_t mean = (q7_9_t)((row_sum / n_frames) >> 2);

        uq18_14_t sum = 0;
        for (int16_t f = 0; f < n_frames; f++) {
            q7_9_t dev =
                (q7_9_t)((int32_t)probe->mel_db[(size_t)m * (size_t)n_frames + (size_t)f] -
                         (int32_t)mean);
            uq14_18_t dev_2 = (uq14_18_t)((int32_t)dev * (int32_t)dev);
            sum += (uq18_14_t)(dev_2 >> 4);
        }
        uq14_18_t var = (uq14_18_t)((sum / n_frames) << 4);
        q7_9_t std = (q7_9_t)fxp_sqrt32(var);

        uint8_t mel_bin = probe->idxs_needed[m];
        probe->mean[mel_bin] = mean;
        probe->std[mel_bin] = std;
        probe->max[mel_bin] = row_max;
        probe->entropy[mel_bin] =
            _mel_entropy(&mel_entropy_power[(size_t)m * (size_t)n_frames], n_frames);
    }

    free(mel_entropy_power);
    free(timedata);
    free(cx_out);
    free(cfg);
    return 1;
}
#endif

#endif
