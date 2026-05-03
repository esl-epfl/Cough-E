#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Audio + IMU input headers are injected via gcc -include so callers can
 * swap them per recording without quoting headaches in Make/shell. The
 * defaults below are used only when neither -include is provided. */
#if !defined(AUDIO_LEN)
#include <input_data/audio_input_55502_w0_9wnds.h>
#endif
#if !defined(IMU_LEN)
#include <input_data/imu_input_55502_w0_9wnds.h>
#endif

#include <audio_features.h>
#include <audio_model.h>
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
#include <time_domain_feat.h>

#include "fxp_metrics.h"

#if !defined(FXP_MODE) || !defined(FIXED_POINT)
int main(void)
{
    fprintf(stderr, "fxp_stage_harness requires -DFXP_MODE and -DFIXED_POINT=16 or 32.\n");
    return 1;
}
#else

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

static void add_audio_kernel_errors(named_metric_t *table,
                                    int *count,
                                    int feature_idx,
                                    double ref,
                                    double fxp)
{
    switch (feature_idx) {
        case SPECTRAL_ROLLOFF:
            add_metric(table, count, "compute_rfft", ref, fxp);
            add_metric(table, count, "compute_rolloff", ref, fxp);
            return;
        case SPECTRAL_SPREAD:
            add_metric(table, count, "compute_rfft", ref, fxp);
            add_metric(table, count, "compute_centroid", ref, fxp);
            add_metric(table, count, "compute_spread", ref, fxp);
            return;
        case SPECTRAL_KURTOSIS:
            add_metric(table, count, "compute_rfft", ref, fxp);
            add_metric(table, count, "compute_centroid", ref, fxp);
            add_metric(table, count, "compute_spread", ref, fxp);
            add_metric(table, count, "compute_kurt", ref, fxp);
            return;
        case SPECTRAL_FLATNESS:
            add_metric(table, count, "compute_periodogram", ref, fxp);
            add_metric(table, count, "compute_flatness", ref, fxp);
            return;
        case DOMINANT_FREQUENCY:
            add_metric(table, count, "compute_periodogram", ref, fxp);
            add_metric(table, count, "get_dominant_freq", ref, fxp);
            return;
        case CREST_FACTOR:
            add_metric(table, count, "audio_get_rms", ref, fxp);
            add_metric(table, count, "audio_get_max", ref, fxp);
            add_metric(table, count, "audio_crest_factor", ref, fxp);
            return;
        default:
            break;
    }

    if (feature_idx >= POWER_SPECTRAL_DENSITY &&
        feature_idx < POWER_SPECTRAL_DENSITY + N_PSD) {
        add_metric(table, count, "compute_periodogram", ref, fxp);
        add_metric(table, count, "normalized_bandpowers", ref, fxp);
        return;
    }

    if (feature_idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT &&
        feature_idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC)) {
        add_metric(table, count, "stft", ref, fxp);
        add_metric(table, count, "mel_spectrogram", ref, fxp);
        add_metric(table, count, "power_to_dB", ref, fxp);
        add_metric(table, count, "feature_aggregation", ref, fxp);
        return;
    }

    if (feature_idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) &&
        feature_idx < MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (4 * N_MFCC)) {
        add_metric(table, count, "stft", ref, fxp);
        add_metric(table, count, "mel_spectrogram", ref, fxp);
        add_metric(table, count, "entropy", ref, fxp);
        add_metric(table, count, "feature_aggregation", ref, fxp);
    }
}

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
