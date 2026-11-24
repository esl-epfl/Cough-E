/* fxp_stage_harness — per-kernel error harness driven by evaluate.py.
 *
 * Sweeps every analysis window of one recording and, for each window, runs
 * the float reference and the fixed-point pipeline side by side. The two
 * outputs are diffed at every kernel boundary (FFT magnitudes, mel power,
 * dB conversion, RMS, line length, AZC, etc.) and the per-kernel error
 * statistics are accumulated into the named_metric_t tables, then printed
 * at the end as the single source of truth for "how lossy is each FxP
 * kernel relative to its float reference."
 *
 * Unlike fxp_progressive_harness, this binary does not run the detection
 * loop or emit COUGH_SEG/N_PEAKS lines — its output is purely numerical
 * error metrics consumed by evaluate.py for the FxP-vs-float report.
 *
 * Audio + IMU input headers are injected via gcc -include so callers can
 * swap them per recording without quoting headaches in Make/shell. The
 * defaults below are used only when neither -include is provided.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if !defined(AUDIO_LEN)
#include <input_data/audio_input_55502_w0_9wnds.h>
#endif
#if !defined(IMU_LEN)
#include <input_data/imu_input_55502_w0_9wnds.h>
#endif

#include <audio_features.h>
#include <audio_model.h>
#include <audio/audio_pipeline_fxp.h>
#include <azc.h>
#include <core/fxp_core.h>
#define audio_features fxp_audio_features
#define imu_features fxp_imu_features
#include <feature_extraction.h>
#undef audio_features
#undef imu_features
#include <frequency_features.h>
#include <helpers.h>
#include <imu/imu_pipeline.h>
#include <imu_features.h>
#include <imu_model.h>
#include <mfcc_module.h>
#include <time_domain_feat.h>

#include "fxp_metrics.h"

#if !defined(FXP_MODE) || !defined(FIXED_POINT)
int main(void)
{
    fprintf(stderr, "fxp_stage_harness requires -DFXP_MODE and -DFIXED_POINT=32.\n");
    return 1;
}
#else

/* Pretty-printers used only for the final per-kernel report header. They map
 * the audio/IMU feature enum back to the human-readable kernel grouping that
 * evaluate.py expects to see in the output. */
static const char *audio_feature_name(int idx)
{
    switch (idx) {
        case SPECTRAL_DECREASE: return "SPECTRAL_DECREASE";
        case SPECTRAL_SLOPE: return "SPECTRAL_SLOPE";
        case SPECTRAL_ROLLOFF: return "SPECTRAL_ROLLOFF";
        case SPECTRAL_CENTROID: return "SPECTRAL_CENTROID";
        case SPECTRAL_SPREAD: return "SPECTRAL_SPREAD";
        case SPECTRAL_KURTOSIS: return "SPECTRAL_KURTOSIS";
        case SPECTRAL_SKEW: return "SPECTRAL_SKEW";
        case SPECTRAL_FLATNESS: return "SPECTRAL_FLATNESS";
        case SPECTRAL_STD: return "SPECTRAL_STD";
        case SPECTRAL_ENTROPY: return "SPECTRAL_ENTROPY";
        case DOMINANT_FREQUENCY: return "DOMINANT_FREQUENCY";
        case ZERO_CROSSING_RATE: return "ZERO_CROSSING_RATE";
        case ROOT_MEANS_SQUARED: return "ROOT_MEANS_SQUARED";
        case CREST_FACTOR: return "CREST_FACTOR";
        default: break;
    }

    if (idx >= POWER_SPECTRAL_DENSITY &&
        idx < POWER_SPECTRAL_DENSITY + N_PSD) {
        return "PSD";
    }

    if (idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT &&
        idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC) {
        return "MEL_MEAN";
    }
    if (idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC &&
        idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC)) {
        return "MEL_STD";
    }
    if (idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) &&
        idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC)) {
        return "MEL_MAX";
    }
    if (idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) &&
        idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (4 * N_MFCC)) {
        return "MEL_ENTROPY";
    }

    if (idx >= ENERGY_ENVELOPE_PEAK_DETECT &&
        idx < Number_AUDIO_Features) {
        return "EEPD";
    }

    return "AUDIO_FEATURE";
}

static const char *imu_signal_name(int sig)
{
    switch (sig) {
        case 0: return "ACCEL_X";
        case 1: return "ACCEL_Y";
        case 2: return "ACCEL_Z";
        case 3: return "GYRO_Y";
        case 4: return "GYRO_P";
        case 5: return "GYRO_R";
        case 6: return "ACCEL_COMBO";
        case 7: return "GYRO_COMBO";
        default: return "IMU";
    }
}

static const char *imu_family_name(int fam)
{
    switch (fam) {
        case LINE_LENGTH: return "LINE_LENGTH";
        case ZERO_CROSSING_RATE_IMU: return "ZERO_CROSSING_RATE";
        case KURTOSIS: return "KURTOSIS";
        case ROOT_MEANS_SQUARED_IMU: return "RMS";
        case CREST_FACTOR_IMU: return "CREST_FACTOR";
        default: return (fam >= APPROXIMATE_ZERO_CROSSING &&
                         fam < APPROXIMATE_ZERO_CROSSING + N_AZC) ? "AZC" : "FEATURE";
    }
}

static void imu_feature_name(int idx, char *buf, size_t n)
{
    int sig = idx / Num_imu_feat_families;
    int fam = idx % Num_imu_feat_families;
    snprintf(buf, n, "%s_%s", imu_signal_name(sig), imu_family_name(fam));
}

/* Float reference feature extractor. Mirrors the kernel call sequence that
 * fxp_audio_features uses internally, but in float, so we can compare each
 * kernel's output against its FxP counterpart at the same point in the chain
 * (the kernel-error helpers below don't have access to the reference Python
 * pipeline, so we recompute it here). */
static void compute_audio_float_features(const int8_t *selector,
                                         const float *sig,
                                         int16_t len,
                                         int16_t fs,
                                         float *feats)
{
    memset(feats, 0, (size_t)Number_AUDIO_Features * sizeof(float));

    int need_fft = selector[SPECTRAL_ROLLOFF] || selector[SPECTRAL_CENTROID] ||
                   selector[SPECTRAL_SPREAD] || selector[SPECTRAL_KURTOSIS];
    if (need_fft) {
        int16_t fft_len = (int16_t)((len / 2) + 1);
        float *mags = (float *)malloc((size_t)fft_len * sizeof(float));
        float *freqs = (float *)malloc((size_t)fft_len * sizeof(float));
        if (mags && freqs) {
            float sum_mags = 0.0f;
            compute_rfft(sig, len, fs, mags, freqs, &sum_mags);
            if (sum_mags > 0.0f) {
                float centroid = compute_centroid(mags, freqs, fft_len, sum_mags);
                float spread = compute_spread(mags, freqs, fft_len, sum_mags, centroid);
                if (selector[SPECTRAL_ROLLOFF]) feats[SPECTRAL_ROLLOFF] = compute_rolloff(mags, freqs, fft_len, sum_mags);
                if (selector[SPECTRAL_CENTROID]) feats[SPECTRAL_CENTROID] = centroid;
                if (selector[SPECTRAL_SPREAD]) feats[SPECTRAL_SPREAD] = spread;
                if (selector[SPECTRAL_KURTOSIS]) feats[SPECTRAL_KURTOSIS] = compute_kurt(mags, freqs, fft_len, sum_mags, centroid, spread);
            }
        }
        free(mags);
        free(freqs);
    }

    int need_psd = selector[SPECTRAL_FLATNESS] || selector[DOMINANT_FREQUENCY];
    for (int i = 0; i < N_PSD; i++) {
        if (selector[POWER_SPECTRAL_DENSITY + i]) need_psd = 1;
    }
    if (need_psd) {
        int16_t psd_len = (int16_t)((NPERSEG / 2) + 1);
        float *psd = (float *)malloc((size_t)psd_len * sizeof(float));
        float *freqs = (float *)malloc((size_t)psd_len * sizeof(float));
        if (psd && freqs) {
            compute_periodogram(sig, len, fs, psd, freqs);
            if (selector[SPECTRAL_FLATNESS]) feats[SPECTRAL_FLATNESS] = compute_flatness(psd, psd_len);
            if (selector[DOMINANT_FREQUENCY]) feats[DOMINANT_FREQUENCY] = get_domiant_freq(psd, freqs, psd_len);

            int8_t psd_selector[N_PSD] = {0};
            float band_powers[N_PSD] = {0.0f};
            int any_band = 0;
            for (int i = 0; i < N_PSD; i++) {
                psd_selector[i] = selector[POWER_SPECTRAL_DENSITY + i];
                if (psd_selector[i]) any_band = 1;
            }
            if (any_band) {
                normalized_bandpowers(psd, freqs, psd_len, psd_selector, band_powers);
                for (int i = 0; i < N_PSD; i++) {
                    if (psd_selector[i]) feats[POWER_SPECTRAL_DENSITY + i] = band_powers[i];
                }
            }
        }
        free(psd);
        free(freqs);
    }

    uint8_t idx_needed[N_MFCC];
    uint8_t n_mels_needed = 0U;
    for (uint8_t i = 0; i < N_MFCC; i++) {
        if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] ||
            selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] ||
            selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] ||
            selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i]) {
            idx_needed[n_mels_needed++] = i;
        }
    }
    if (n_mels_needed > 0U) {
        float *mean = (float *)malloc((size_t)n_mels_needed * sizeof(float));
        float *std = (float *)malloc((size_t)n_mels_needed * sizeof(float));
        float *max = (float *)malloc((size_t)n_mels_needed * sizeof(float));
        float *ent = (float *)malloc((size_t)n_mels_needed * sizeof(float));
        if (mean && std && max && ent) {
            get_mel_spectrogram_features(sig, len, idx_needed, n_mels_needed, mean, std, max, ent);
            for (uint8_t k = 0; k < n_mels_needed; k++) {
                uint8_t i = idx_needed[k];
                if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i]) {
                    feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = mean[k];
                }
                if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i]) {
                    feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = std[k];
                }
                if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i]) {
                    feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] = max[k];
                }
                if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i]) {
                    feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i] = ent[k];
                }
            }
        }
        free(mean);
        free(std);
        free(max);
        free(ent);
    }

    if (selector[ZERO_CROSSING_RATE] || selector[ROOT_MEANS_SQUARED] || selector[CREST_FACTOR]) {
        float *centered = (float *)malloc((size_t)len * sizeof(float));
        if (centered) {
            sub_mean(sig, centered, len);
            float rms = get_rms(centered, len);
            if (selector[ZERO_CROSSING_RATE]) feats[ZERO_CROSSING_RATE] = compute_zrc(centered, len);
            if (selector[ROOT_MEANS_SQUARED]) feats[ROOT_MEANS_SQUARED] = rms;
            if (selector[CREST_FACTOR]) feats[CREST_FACTOR] = (rms > 0.0f) ? (get_max(centered, len) / rms) : 0.0f;
        }
        free(centered);
    }

    if (selector[ENERGY_ENVELOPE_PEAK_DETECT]) {
        int8_t eepd_selector[N_EEPD] = {0};
        int16_t eepd_out[N_EEPD] = {0};
        eepd_selector[0] = 1;
        eepd(sig, len, fs, eepd_selector, eepd_out);
        feats[ENERGY_ENVELOPE_PEAK_DETECT] = (float)eepd_out[0];
    }
}

static void compute_imu_float_features(const int8_t *selector,
                                       const float sig[][Num_IMU_signals],
                                       int16_t len,
                                       float *feats)
{
    memset(feats, 0, (size_t)Number_IMU_Features * sizeof(float));

    float *axis = (float *)malloc((size_t)len * sizeof(float));
    float *l2a = (float *)malloc((size_t)len * sizeof(float));
    float *l2g = (float *)malloc((size_t)len * sizeof(float));
    if (!axis || !l2a || !l2g) {
        free(axis);
        free(l2a);
        free(l2g);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        l2a[i] = sqrtf(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1] + sig[i][2] * sig[i][2]);
        l2g[i] = sqrtf(sig[i][3] * sig[i][3] + sig[i][4] * sig[i][4] + sig[i][5] * sig[i][5]);
    }

    const int base[8] = {
        ACCEL_X_FEAT, ACCEL_Y_FEAT, ACCEL_Z_FEAT,
        GYRO_Y_FEAT, GYRO_P_FEAT, GYRO_R_FEAT,
        ACCEL_COMBO, GYRO_COMBO
    };

    for (int s = 0; s < 8; s++) {
        float *src = axis;
        if (s < 6) {
            for (int16_t i = 0; i < len; i++) axis[i] = sig[i][s];
        } else {
            src = (s == 6) ? l2a : l2g;
        }

        int b = base[s];
        if (selector[b + LINE_LENGTH]) feats[b + LINE_LENGTH] = get_line_length(src, len);
        if (selector[b + ZERO_CROSSING_RATE_IMU]) feats[b + ZERO_CROSSING_RATE_IMU] = compute_zrc(src, len);
        if (selector[b + KURTOSIS]) feats[b + KURTOSIS] = get_kurtosis(src, len);
        if (selector[b + ROOT_MEANS_SQUARED_IMU]) feats[b + ROOT_MEANS_SQUARED_IMU] = get_rms(src, len);
        if (selector[b + CREST_FACTOR_IMU]) {
            float rms = get_rms(src, len);
            feats[b + CREST_FACTOR_IMU] = (rms > 0.0f) ? (get_max(src, len) / rms) : 0.0f;
        }
        for (int azc = 0; azc < N_AZC; azc++) {
            int idx = b + APPROXIMATE_ZERO_CROSSING + azc;
            if (selector[idx]) {
                float eps = EPSILON_START + (EPSILON_STEP * (float)azc);
                feats[idx] = (float)azc_computation(src, len, eps);
            }
        }
    }

    free(axis);
    free(l2a);
    free(l2g);
}

#define MAX_KERNELS 64

/* One slot per kernel name; fxp_metric_acc_t accumulates ref/fxp pairs and
 * computes summary statistics (max abs error, RMSE, etc.) on demand. The
 * harness adds samples to a slot keyed by kernel name during the window
 * sweep, then prints all slots once at the end. */
typedef struct {
    char name[48];
    fxp_metric_acc_t acc;
} named_metric_t;

static int find_or_add(named_metric_t *table, int *count, const char *name)
{
    for (int i = 0; i < *count; i++) {
        if (strcmp(table[i].name, name) == 0) return i;
    }
    if (*count >= MAX_KERNELS) return -1;
    int idx = (*count)++;
    snprintf(table[idx].name, sizeof(table[idx].name), "%s", name);
    fxp_metric_init(&table[idx].acc);
    return idx;
}

static void add_metric(named_metric_t *table,
                       int *count,
                       const char *name,
                       double ref,
                       double fxp)
{
    int slot = find_or_add(table, count, name);
    if (slot >= 0) fxp_metric_add(&table[slot].acc, ref, fxp);
}

static float audio_feat_to_float(fxp_feat_t value, uint16_t feature_idx);
static float imu_feat_to_float(fxp_feat_t value, uint16_t feature_idx);

static float fxp_scaled_fft_power_to_float(uint64_t power, uint8_t frac_bits, int16_t fft_len)
{
    double p = ldexp((double)power, -(int)frac_bits);
    return (float)(p * (double)fft_len * (double)fft_len);
}

static double welch_psd_scale(int16_t fs, int16_t steps)
{
    double win_sum = 0.0;
    for (int16_t i = 0; i < NPERSEG; i++) {
        win_sum += (double)hann_window[i] * (double)hann_window[i];
    }
    if (fs <= 0 || steps <= 0 || win_sum <= 0.0) return 0.0;
    return ((double)NPERSEG * (double)NPERSEG) / ((double)fs * win_sum * (double)steps);
}

/* Audio FFT kernel diff. Uses audio_fft_stage_probe (a debug entry point in
 * audio_pipeline_fxp.c, only compiled when FXP_STAGE_PROBES is set) to
 * expose the FxP magnitudes and frequency bins in addition to the final
 * features, so we can record errors at every internal stage of the FFT
 * subpipeline rather than only on the final feature values. The mag_scale
 * normalization is needed because the float and FxP RFFTs use different
 * normalizations; we collapse that out so the reported error reflects bin
 * shape only, not absolute amplitude. */
static void add_audio_fft_kernel_metrics(named_metric_t *table,
                                         int *count,
                                         const int8_t *selector,
                                         const float *sig,
                                         const int16_t *sig_q14,
                                         int16_t len,
                                         int16_t fs)
{
    int need_rolloff = selector[SPECTRAL_ROLLOFF];
    int need_centroid = selector[SPECTRAL_CENTROID] ||
                        selector[SPECTRAL_SPREAD] ||
                        selector[SPECTRAL_KURTOSIS];
    int need_spread = selector[SPECTRAL_SPREAD] ||
                      selector[SPECTRAL_KURTOSIS];
    int need_kurt = selector[SPECTRAL_KURTOSIS];
    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    int16_t fft_len = (int16_t)((len / 2) + 1);
    float *mags = (float *)malloc((size_t)fft_len * sizeof(float));
    float *freqs = (float *)malloc((size_t)fft_len * sizeof(float));
    uq4_28_t *fxp_mags_q28 = (uq4_28_t *)malloc((size_t)fft_len * sizeof(uq4_28_t));
    uq12_20_t *fxp_freqs_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    fxp_feat_t fxp_feats[Number_AUDIO_Features];
    if (!mags || !freqs || !fxp_mags_q28 || !fxp_freqs_q20) {
        free(mags);
        free(freqs);
        free(fxp_mags_q28);
        free(fxp_freqs_q20);
        return;
    }

    memset(fxp_feats, 0, sizeof(fxp_feats));

    float sum_mags = 0.0f;
    compute_rfft(sig, len, fs, mags, freqs, &sum_mags);
    uq7_25_t fxp_sum_q25 = 0;
    int have_fxp_fft = audio_fft_stage_probe(sig_q14, len, fs,
                                             fxp_mags_q28,
                                             fxp_freqs_q20,
                                             &fxp_sum_q25);

    int8_t probe_selector[Number_AUDIO_Features] = {0};
    if (need_rolloff) probe_selector[SPECTRAL_ROLLOFF] = 1;
    if (need_centroid) probe_selector[SPECTRAL_CENTROID] = 1;
    if (need_spread) probe_selector[SPECTRAL_SPREAD] = 1;
    if (need_kurt) probe_selector[SPECTRAL_KURTOSIS] = 1;
    audio_fft_features(probe_selector, sig_q14, len, fs, fxp_feats);

    if (have_fxp_fft) {
        (void)fxp_sum_q25;
        for (int16_t i = 0; i < fft_len; i++) {
            add_metric(table, count, "compute_rfft",
                       mags[i],
                       FXP_TO_FLOAT(fxp_mags_q28[i], 28) * (float)len);
        }
    }

    if (sum_mags > 0.0f) {
        float centroid = need_centroid ? compute_centroid(mags, freqs, fft_len, sum_mags) : 0.0f;
        float spread = need_spread ? compute_spread(mags, freqs, fft_len, sum_mags, centroid) : 0.0f;

        if (need_rolloff) {
            float ref = compute_rolloff(mags, freqs, fft_len, sum_mags);
            float fxp = audio_feat_to_float(fxp_feats[SPECTRAL_ROLLOFF], SPECTRAL_ROLLOFF);
            add_metric(table, count, "compute_rolloff", ref, fxp);
        }
        if (need_centroid) {
            float fxp = audio_feat_to_float(fxp_feats[SPECTRAL_CENTROID], SPECTRAL_CENTROID);
            add_metric(table, count, "compute_centroid", centroid, fxp);
        }
        if (need_spread) {
            float fxp = audio_feat_to_float(fxp_feats[SPECTRAL_SPREAD], SPECTRAL_SPREAD);
            add_metric(table, count, "compute_spread", spread, fxp);
        }
        if (need_kurt) {
            float ref = compute_kurt(mags, freqs, fft_len, sum_mags, centroid, spread);
            float fxp = audio_feat_to_float(fxp_feats[SPECTRAL_KURTOSIS], SPECTRAL_KURTOSIS);
            add_metric(table, count, "compute_kurt", ref, fxp);
        }
    }

    free(mags);
    free(freqs);
    free(fxp_mags_q28);
    free(fxp_freqs_q20);
}

static void add_audio_psd_kernel_metrics(named_metric_t *table,
                                         int *count,
                                         const int8_t *selector,
                                         const float *sig,
                                         const int16_t *sig_q14,
                                         int16_t len,
                                         int16_t fs)
{
    int need_psd = selector[SPECTRAL_FLATNESS] || selector[DOMINANT_FREQUENCY];
    for (int i = 0; i < N_PSD; i++) {
        if (selector[POWER_SPECTRAL_DENSITY + i]) need_psd = 1;
    }
    if (!need_psd) return;

    int16_t psd_len = (int16_t)((NPERSEG / 2) + 1);
    float *ref_psd = (float *)malloc((size_t)psd_len * sizeof(float));
    float *ref_freqs = (float *)malloc((size_t)psd_len * sizeof(float));
    uq9_23_t *fxp_acc_power = (uq9_23_t *)malloc((size_t)psd_len * sizeof(uq9_23_t));
    uq12_20_t *fxp_freqs = (uq12_20_t *)malloc((size_t)psd_len * sizeof(uq12_20_t));
    if (!ref_psd || !ref_freqs || !fxp_acc_power || !fxp_freqs) {
        free(ref_psd);
        free(ref_freqs);
        free(fxp_acc_power);
        free(fxp_freqs);
        return;
    }

    compute_periodogram(sig, len, fs, ref_psd, ref_freqs);

    int16_t steps = 0;
    if (audio_psd_stage_probe(sig_q14, len, fs, fxp_acc_power, fxp_freqs, &steps)) {
        double scale = welch_psd_scale(fs, steps);
        for (int16_t i = 0; i < psd_len; i++) {
            float fxp_psd = (float)((double)FXP_TO_FLOAT(fxp_acc_power[i], 23) * scale);
            add_metric(table, count, "compute_periodogram", ref_psd[i], fxp_psd);
        }
    }

    free(ref_psd);
    free(ref_freqs);
    free(fxp_acc_power);
    free(fxp_freqs);
}

static int audio_mel_feature_selected(const int8_t *selector)
{
    for (uint16_t i = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT;
         i < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (N_MFCC * N_MFCC_FAMILIES);
         i++) {
        if (selector[i]) return 1;
    }
    return 0;
}

/* Audio mel kernel diff. Linear STFT and mel powers are scaled back to the
 * float domain before comparison. The dB-domain comparison is reserved for
 * power_to_dB and the final dB feature aggregation. */
static void add_audio_mel_kernel_metrics(named_metric_t *table,
                                         int *count,
                                         const int8_t *selector,
                                         const float *sig,
                                         const int16_t *sig_q14,
                                         int16_t len)
{
    if (!audio_mel_feature_selected(selector)) return;

    audio_mel_stage_probe_t probe;
    if (!audio_mel_stage_probe(selector, sig_q14, len, &probe)) return;

    int16_t n_frames = probe.n_frames;
    size_t frame_count = (size_t)n_frames * (size_t)FFT_RES_LEN;
    size_t mel_count = (size_t)probe.n_mels * (size_t)n_frames;

    float *ref_frame_power = (float *)malloc(frame_count * sizeof(float));
    float *ref_mel_power = (float *)malloc(mel_count * sizeof(float));
    float *ref_mel_db = (float *)malloc(mel_count * sizeof(float));
    float *ref_mean = (float *)malloc((size_t)probe.n_mels * sizeof(float));
    float *ref_std = (float *)malloc((size_t)probe.n_mels * sizeof(float));
    float *ref_max = (float *)malloc((size_t)probe.n_mels * sizeof(float));
    float *ref_entropy = (float *)malloc((size_t)probe.n_mels * sizeof(float));

    if (!ref_frame_power || !ref_mel_power || !ref_mel_db ||
        !ref_mean || !ref_std || !ref_max || !ref_entropy) {
        free(ref_frame_power);
        free(ref_mel_power);
        free(ref_mel_db);
        free(ref_mean);
        free(ref_std);
        free(ref_max);
        free(ref_entropy);
        audio_mel_stage_probe_free(&probe);
        return;
    }

    memset(ref_mel_power, 0, mel_count * sizeof(float));

    stft(sig, len, n_frames, ref_frame_power);
    mel_spectrogram(sig, len, n_frames, probe.idxs_needed, ref_mel_power);
    power_to_dB(ref_mel_power, (int16_t)mel_count, ref_mel_db);

    float *entropy_input = (float *)malloc(mel_count * sizeof(float));
    if (entropy_input) {
        memcpy(entropy_input, ref_mel_power, mel_count * sizeof(float));
        entropy(entropy_input, probe.n_mels, n_frames, ref_entropy);
        free(entropy_input);
    } else {
        memset(ref_entropy, 0, (size_t)probe.n_mels * sizeof(float));
    }

    for (size_t i = 0; i < frame_count; i++) {
        float fxp = fxp_scaled_fft_power_to_float(probe.frame_power[i],
                                                  probe.frame_power_frac_bits,
                                                  N_FFT);
        add_metric(table, count, "stft", ref_frame_power[i], fxp);
    }

    for (size_t i = 0; i < mel_count; i++) {
        float fxp_mel_power = fxp_scaled_fft_power_to_float(probe.mel_power[i],
                                                            probe.mel_power_frac_bits,
                                                            N_FFT);
        add_metric(table, count, "mel_spectrogram", ref_mel_power[i], fxp_mel_power);
        add_metric(table, count, "power_to_dB", ref_mel_db[i], FXP_TO_FLOAT(probe.mel_db[i], 9));
    }

    for (int16_t m = 0; m < probe.n_mels; m++) {
        uint8_t mel_bin = probe.idxs_needed[m];
        ref_mean[m] = vect_mean(&ref_mel_db[(size_t)m * (size_t)n_frames], n_frames);
        ref_std[m] = vect_std(&ref_mel_db[(size_t)m * (size_t)n_frames], n_frames);
        ref_max[m] = vect_max_value(&ref_mel_db[(size_t)m * (size_t)n_frames], n_frames);

        if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + mel_bin]) {
            add_metric(table, count, "feature_aggregation",
                       ref_mean[m],
                       FXP_TO_FLOAT(probe.mean[mel_bin], 9));
        }
        if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + mel_bin]) {
            add_metric(table, count, "feature_aggregation",
                       ref_std[m],
                       FXP_TO_FLOAT(probe.std[mel_bin], 9));
        }
        if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + mel_bin]) {
            add_metric(table, count, "feature_aggregation",
                       ref_max[m],
                       FXP_TO_FLOAT(probe.max[mel_bin], 9));
        }
        if (selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + mel_bin]) {
            add_metric(table, count, "entropy",
                       ref_entropy[m],
                       FXP_TO_FLOAT(probe.entropy[mel_bin], 14));
        }
    }

    free(ref_frame_power);
    free(ref_mel_power);
    free(ref_mel_db);
    free(ref_mean);
    free(ref_std);
    free(ref_max);
    free(ref_entropy);
    audio_mel_stage_probe_free(&probe);
}

static void add_audio_scalar_kernel_metrics(named_metric_t *table,
                                           int *count,
                                           const int8_t *selector,
                                           const float *sig,
                                           const int16_t *sig_q14,
                                           const fxp_feat_t *fxp_feats,
                                           int16_t len)
{
    int need_rms = selector[ROOT_MEANS_SQUARED] || selector[CREST_FACTOR];
    int need_max = selector[CREST_FACTOR];
    if (!need_rms && !need_max) return;

    float *centered = (float *)malloc((size_t)len * sizeof(float));
    if (!centered) return;

    sub_mean(sig, centered, len);
    float ref_rms = get_rms(centered, len);
    float ref_max = need_max ? get_max(centered, len) : 0.0f;
    float ref_crest = (ref_rms > 0.0f) ? (ref_max / ref_rms) : 0.0f;

    int64_t sum_q14 = 0;
    for (int16_t i = 0; i < len; i++) {
        sum_q14 += (int64_t)sig_q14[i];
    }

    int32_t mean_q14 = (sum_q14 >= 0)
        ? (int32_t)((sum_q14 + (len / 2)) / len)
        : (int32_t)(-(((-sum_q14) + (len / 2)) / len));

    int32_t max_q14 = (int32_t)sig_q14[0] - mean_q14;
    uint64_t sum_sq_q28 = 0;
    for (int16_t i = 0; i < len; i++) {
        int32_t cur = (int32_t)sig_q14[i] - mean_q14;
        if (cur > max_q14) max_q14 = cur;
        sum_sq_q28 += (uint64_t)((int64_t)cur * (int64_t)cur);
    }

    uint64_t mean_sq_q28 = (sum_sq_q28 + ((uint64_t)len >> 1U)) / (uint64_t)len;
    int32_t rms_q14 = (int32_t)fxp_sqrt64(mean_sq_q28);

    if (need_rms) {
        add_metric(table, count, "audio_get_rms", ref_rms, FXP_TO_FLOAT(rms_q14, FXP_FRAC_AUDIO_INPUT));
    }
    if (need_max) {
        add_metric(table, count, "audio_get_max", ref_max, FXP_TO_FLOAT(max_q14, FXP_FRAC_AUDIO_INPUT));
        add_metric(table, count, "audio_crest_factor", ref_crest,
                   audio_feat_to_float(fxp_feats[CREST_FACTOR], CREST_FACTOR));
    }

    free(centered);
}

/* For features whose final value is what we want to attribute back to a
 * named kernel, this function maps the audio feature enum to the kernel
 * slot(s) that contributed to it. Spectral features handled by the FFT/mel
 * probes return early because their per-stage errors were already recorded
 * by add_audio_fft_kernel_metrics / add_audio_mel_kernel_metrics. */
static void add_audio_kernel_errors(named_metric_t *table,
                                    int *count,
                                    int feature_idx,
                                    double ref,
                                    double fxp)
{
    switch (feature_idx) {
        case SPECTRAL_ROLLOFF:
            return;
        case SPECTRAL_CENTROID:
        case SPECTRAL_SPREAD:
            return;
        case SPECTRAL_KURTOSIS:
            return;
        case SPECTRAL_FLATNESS:
            add_metric(table, count, "compute_flatness", ref, fxp);
            return;
        case DOMINANT_FREQUENCY:
            add_metric(table, count, "get_dominant_freq", ref, fxp);
            return;
        case ROOT_MEANS_SQUARED:
        case CREST_FACTOR:
            return;
        default:
            break;
    }

    if (feature_idx >= POWER_SPECTRAL_DENSITY &&
        feature_idx < POWER_SPECTRAL_DENSITY + N_PSD) {
        add_metric(table, count, "normalized_bandpowers", ref, fxp);
        return;
    }

    if (feature_idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT &&
        feature_idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (4 * N_MFCC)) {
        (void)ref;
        (void)fxp;
        return;
    }
}

/* IMU equivalent of add_audio_kernel_errors. The L2-norm signals (ACCEL_COMBO,
 * GYRO_COMBO) get an extra slot because their values are derived from a sqrt
 * across three axes that's worth tracking on its own. */
static void add_imu_kernel_errors(named_metric_t *table,
                                  int *count,
                                  int feature_idx,
                                  double ref,
                                  double fxp)
{
    int sig = feature_idx / Num_imu_feat_families;
    int fam = feature_idx % Num_imu_feat_families;

    if (sig >= 6) {
        add_metric(table, count, "L2_norm", ref, fxp);
    }

    switch (fam) {
        case LINE_LENGTH:
            add_metric(table, count, "get_line_length", ref, fxp);
            return;
        case KURTOSIS:
            add_metric(table, count, "get_kurtosis", ref, fxp);
            return;
        case ROOT_MEANS_SQUARED_IMU:
            add_metric(table, count, "get_rms", ref, fxp);
            return;
        case CREST_FACTOR_IMU:
            add_metric(table, count, "get_rms", ref, fxp);
            add_metric(table, count, "get_max", ref, fxp);
            return;
        default:
            if (fam >= APPROXIMATE_ZERO_CROSSING &&
                fam < APPROXIMATE_ZERO_CROSSING + N_AZC) {
                add_metric(table, count, "azc_computation", ref, fxp);
            }
            return;
    }
}

static float audio_feat_to_float(fxp_feat_t value, uint16_t feature_idx)
{
    if (audio_feature_is_signed(feature_idx)) {
        return FXP_TO_FLOAT((int32_t)value, audio_feature_frac_bits(feature_idx));
    }
    return FXP_TO_FLOAT(value, audio_feature_frac_bits(feature_idx));
}

static float imu_feat_to_float(fxp_feat_t value, uint16_t feature_idx)
{
    if (imu_feature_is_signed(feature_idx)) {
        return FXP_TO_FLOAT((int32_t)value, imu_feature_frac_bits(feature_idx));
    }
    return FXP_TO_FLOAT(value, imu_feature_frac_bits(feature_idx));
}

/* Sweep every analysis window of this recording, run float + FxP side by
 * side, and accumulate per-kernel error stats. The audio and IMU passes are
 * independent — the harness does not interleave them through the FSM, since
 * the goal is per-kernel quantization error, not detection-loop behavior. */
int main(void)
{
    named_metric_t audio_table[MAX_KERNELS];
    named_metric_t imu_table[MAX_KERNELS];
    int audio_n = 0;
    int imu_n = 0;

    int n_audio_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    int n_imu_wins = ((IMU_LEN - WINDOW_SAMP_IMU) / IMU_STEP) + 1;

    float *audio_ref_feats = (float *)malloc((size_t)Number_AUDIO_Features * sizeof(float));
    fxp_feat_t *audio_fxp_feats = (fxp_feat_t *)malloc((size_t)Number_AUDIO_Features * sizeof(fxp_feat_t));
    int16_t *audio_q14 = (int16_t *)malloc((size_t)WINDOW_SAMP_AUDIO * sizeof(int16_t));
    if (!audio_ref_feats || !audio_fxp_feats || !audio_q14) {
        fprintf(stderr, "audio harness allocation failed.\n");
        free(audio_ref_feats);
        free(audio_fxp_feats);
        free(audio_q14);
        return 1;
    }

    for (int w = 0; w < n_audio_wins; w++) {
        const float *sig = &audio_in.air[w * AUDIO_STEP];
        memset(audio_fxp_feats, 0, (size_t)Number_AUDIO_Features * sizeof(fxp_feat_t));

        for (int i = 0; i < WINDOW_SAMP_AUDIO; i++) {
            audio_q14[i] = FXP_AUDIO_FROM_FLOAT(sig[i]);
        }

        compute_audio_float_features(audio_features_selector, sig, WINDOW_SAMP_AUDIO, AUDIO_FS, audio_ref_feats);
        fxp_audio_features(audio_features_selector, audio_q14, WINDOW_SAMP_AUDIO, AUDIO_FS, audio_fxp_feats);
        add_audio_fft_kernel_metrics(audio_table, &audio_n,
                                     audio_features_selector,
                                     sig,
                                     audio_q14,
                                     WINDOW_SAMP_AUDIO,
                                     AUDIO_FS);
        add_audio_psd_kernel_metrics(audio_table, &audio_n,
                                     audio_features_selector,
                                     sig,
                                     audio_q14,
                                     WINDOW_SAMP_AUDIO,
                                     AUDIO_FS);
        add_audio_scalar_kernel_metrics(audio_table, &audio_n,
                                        audio_features_selector,
                                        sig,
                                        audio_q14,
                                        audio_fxp_feats,
                                        WINDOW_SAMP_AUDIO);
        add_audio_mel_kernel_metrics(audio_table, &audio_n,
                                     audio_features_selector,
                                     sig,
                                     audio_q14,
                                     WINDOW_SAMP_AUDIO);

        for (int i = 0; i < Number_AUDIO_Features; i++) {
            if (!audio_features_selector[i]) continue;
            float fxp_v = audio_feat_to_float(audio_fxp_feats[i], (uint16_t)i);
            add_audio_kernel_errors(audio_table, &audio_n, i, audio_ref_feats[i], fxp_v);
        }
    }

    free(audio_ref_feats);
    free(audio_fxp_feats);
    free(audio_q14);

    float imu_ref_feats[Number_IMU_Features];
    fxp_feat_t imu_fxp_feats[Number_IMU_Features];
    q11_5_t (*imu_q5)[Num_IMU_signals] = malloc((size_t)WINDOW_SAMP_IMU * sizeof(*imu_q5));
    if (!imu_q5) {
        fprintf(stderr, "imu harness allocation failed.\n");
        return 1;
    }

    for (int w = 0; w < n_imu_wins; w++) {
        const float (*sig)[Num_IMU_signals] = &imu_in[w * IMU_STEP];
        memset(imu_fxp_feats, 0, sizeof(imu_fxp_feats));

        for (int i = 0; i < WINDOW_SAMP_IMU; i++) {
            for (int ax = 0; ax < Num_IMU_signals; ax++) {
                imu_q5[i][ax] = FXP_IMU_RAW_FROM_FLOAT(sig[i][ax]);
            }
        }

        compute_imu_float_features(imu_features_selector, sig, WINDOW_SAMP_IMU, imu_ref_feats);
        fxp_imu_features(imu_features_selector, imu_q5, WINDOW_SAMP_IMU, imu_fxp_feats);

        for (int i = 0; i < Number_IMU_Features; i++) {
            if (!imu_features_selector[i]) continue;
            float fxp_v = imu_feat_to_float(imu_fxp_feats[i], (uint16_t)i);
            add_imu_kernel_errors(imu_table, &imu_n, i, imu_ref_feats[i], fxp_v);
        }
    }

    free(imu_q5);

    for (int i = 0; i < audio_n; i++) {
        if (audio_table[i].acc.n <= 0) continue;
        fxp_metric_print_kernel_acc("audio", audio_table[i].name, &audio_table[i].acc);
    }
    for (int i = 0; i < imu_n; i++) {
        if (imu_table[i].acc.n <= 0) continue;
        fxp_metric_print_kernel_acc("imu", imu_table[i].name, &imu_table[i].acc);
    }

    return 0;
}

#endif
