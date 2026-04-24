#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef AUDIO_HEADER
#define AUDIO_HEADER <input_data/audio_input_55502_w0_9wnds.h>
#endif
#ifndef IMU_HEADER
#define IMU_HEADER <input_data/imu_input_55502_w0_9wnds.h>
#endif
#ifndef BIO_HEADER
#define BIO_HEADER <input_data/bio_input_55502.h>
#endif

#include AUDIO_HEADER
#include IMU_HEADER
#include BIO_HEADER

#include <audio_features.h>
#include <audio_model.h>
#include <azc.h>
#include <core/fxp_convert.h>
#include <feature_extraction.h>
#include <frequency_features.h>
#include <helpers.h>
#include <imu/imu_pipeline.h>
#include <imu_features.h>
#include <imu_model.h>
#include <kiss_fftr.h>
#include <model_fxp.h>
#include <time_domain_feat.h>

#include "fxp_metrics.h"

#if !defined(FXP_MODE) || !defined(FIXED_POINT)
int main(void)
{
    fprintf(stderr, "fxp_stage_harness requires -DFXP_MODE and -DFIXED_POINT=16 or 32.\n");
    return 1;
}
#else

#define MAX_AUDIO_TRACE_FEATURES 16
#define MAX_IMU_TRACE_FEATURES 16

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(FIXED_POINT) && (FIXED_POINT == 32)
#define KISS_SCALAR_SCALE 2147483647.0
#else
#define KISS_SCALAR_SCALE 32767.0
#endif

static const char *k_imu_signal_name[8] = {
    "ACCEL_X", "ACCEL_Y", "ACCEL_Z",
    "GYRO_Y", "GYRO_P", "GYRO_R",
    "ACCEL_COMBO", "GYRO_COMBO"
};

static const char *k_imu_family_name[Num_imu_feat_families] = {
    "LINE_LENGTH",
    "ZERO_CROSSING_RATE",
    "KURTOSIS",
    "RMS",
    "CREST_FACTOR",
    "AZC_0",
    "AZC_1",
    "AZC_2",
    "AZC_3",
    "AZC_4",
    "AZC_5",
    "AZC_6",
    "AZC_7"
};

typedef enum {
    FXP_HARNESS_BACKEND_FLOAT = 0,
    FXP_HARNESS_BACKEND_FXP = 1
} fxp_harness_backend_t;

typedef struct {
    fxp_harness_backend_t audio_features;
    fxp_harness_backend_t audio_model;
    fxp_harness_backend_t imu_features;
    fxp_harness_backend_t imu_model;
} fxp_harness_backend_config_t;

typedef struct {
    const char *block;
    const char *kernel;
    const char *stage;
    const char *qformat;
    fxp_harness_backend_t backend;
} fxp_harness_stage_desc_t;

static const char *arg_string(int argc, char **argv, const char *name, const char *fallback);

static const char *backend_name(fxp_harness_backend_t backend)
{
    return (backend == FXP_HARNESS_BACKEND_FXP) ? "fxp" : "float";
}

static fxp_harness_backend_t parse_backend_name(const char *name, fxp_harness_backend_t fallback)
{
    (void)fallback;
    if (strcmp(name, "float") == 0) return FXP_HARNESS_BACKEND_FLOAT;
    if (strcmp(name, "fxp") == 0) return FXP_HARNESS_BACKEND_FXP;
    fprintf(stderr, "Unknown backend '%s'; expected 'float' or 'fxp'.\n", name);
    exit(2);
}

static void parse_backend_config(int argc, char **argv, fxp_harness_backend_config_t *cfg)
{
    cfg->audio_features = FXP_HARNESS_BACKEND_FXP;
    cfg->audio_model = FXP_HARNESS_BACKEND_FXP;
    cfg->imu_features = FXP_HARNESS_BACKEND_FXP;
    cfg->imu_model = FXP_HARNESS_BACKEND_FXP;

    cfg->audio_features = parse_backend_name(arg_string(argc, argv, "--audio-features-backend", backend_name(cfg->audio_features)), cfg->audio_features);
    cfg->audio_model = parse_backend_name(arg_string(argc, argv, "--audio-model-backend", backend_name(cfg->audio_model)), cfg->audio_model);
    cfg->imu_features = parse_backend_name(arg_string(argc, argv, "--imu-features-backend", backend_name(cfg->imu_features)), cfg->imu_features);
    cfg->imu_model = parse_backend_name(arg_string(argc, argv, "--imu-model-backend", backend_name(cfg->imu_model)), cfg->imu_model);
}

static int is_hybrid_config(const fxp_harness_backend_config_t *cfg)
{
    return cfg->audio_features != FXP_HARNESS_BACKEND_FXP ||
           cfg->audio_model != FXP_HARNESS_BACKEND_FXP ||
           cfg->imu_features != FXP_HARNESS_BACKEND_FXP ||
           cfg->imu_model != FXP_HARNESS_BACKEND_FXP;
}

static void print_descriptor(const fxp_harness_stage_desc_t *desc)
{
    printf("FXP_DESCRIPTOR,block=%s,kernel=%s,stage=%s,backend=%s,qformat=%s\n",
           desc->block,
           desc->kernel,
           desc->stage,
           backend_name(desc->backend),
           desc->qformat);
}

static void print_descriptor_table(void)
{
    static const fxp_harness_stage_desc_t descs[] = {
        {"audio", "source", "input_float_to_Q1.14", "Q1.14", FXP_HARNESS_BACKEND_FXP},
        {"imu", "source", "input_float_to_Q11.5", "Q11.5", FXP_HARNESS_BACKEND_FXP},
        {"audio", "features", "selected_feature_outputs_to_Q16", "Q16", FXP_HARNESS_BACKEND_FXP},
        {"imu", "features", "selected_feature_outputs_to_Q16", "Q16", FXP_HARNESS_BACKEND_FXP},
        {"imu", "l2_norm_accel", "raw_square_sum_sqrt", "UQ10.6", FXP_HARNESS_BACKEND_FXP},
        {"imu", "l2_norm_gyro", "raw_square_sum_sqrt", "UQ5.11", FXP_HARNESS_BACKEND_FXP},
        {"audio", "model", "model_logit", "Q16", FXP_HARNESS_BACKEND_FXP},
        {"imu", "model", "model_logit", "Q16", FXP_HARNESS_BACKEND_FXP},
        {"kissfft", "rfft", "output_bins_q15_vs_q31", "native", FXP_HARNESS_BACKEND_FXP},
        {"postprocessing", "events", "not_yet_isolated", "n/a", FXP_HARNESS_BACKEND_FLOAT},
    };

    for (size_t i = 0; i < sizeof(descs) / sizeof(descs[0]); i++) {
        print_descriptor(&descs[i]);
    }
}

static int arg_value(int argc, char **argv, const char *name, int fallback)
{
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return atoi(argv[i + 1]);
    }
    return fallback;
}

static int arg_flag(int argc, char **argv, const char *name)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static const char *arg_string(int argc, char **argv, const char *name, const char *fallback)
{
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

static float q_to_float_signed(int32_t x, int frac)
{
    return (float)((double)x / (double)(1ULL << frac));
}

static int32_t quant_signed(float x, int frac, int bits)
{
    double scaled = (double)x * (double)(1ULL << frac);
    scaled += (scaled >= 0.0) ? 0.5 : -0.5;

    int64_t max_v = ((int64_t)1 << (bits - 1)) - 1;
    int64_t min_v = -((int64_t)1 << (bits - 1));
    if (scaled > (double)max_v) return (int32_t)max_v;
    if (scaled < (double)min_v) return (int32_t)min_v;
    return (int32_t)scaled;
}

static int32_t quant_signed_metric(float x, int frac, int bits, fxp_metric_acc_t *metric)
{
    double scaled = (double)x * (double)(1ULL << frac);
    scaled += (scaled >= 0.0) ? 0.5 : -0.5;

    int64_t max_v = ((int64_t)1 << (bits - 1)) - 1;
    int64_t min_v = -((int64_t)1 << (bits - 1));
    if (scaled > (double)max_v) {
        fxp_metric_count_saturation(metric);
        return (int32_t)max_v;
    }
    if (scaled < (double)min_v) {
        fxp_metric_count_saturation(metric);
        return (int32_t)min_v;
    }
    return (int32_t)scaled;
}

static void audio_feature_name(int idx, char *buf, size_t n)
{
    switch (idx) {
        case SPECTRAL_DECREASE: snprintf(buf, n, "SPECTRAL_DECREASE"); return;
        case SPECTRAL_SLOPE: snprintf(buf, n, "SPECTRAL_SLOPE"); return;
        case SPECTRAL_ROLLOFF: snprintf(buf, n, "SPECTRAL_ROLLOFF"); return;
        case SPECTRAL_CENTROID: snprintf(buf, n, "SPECTRAL_CENTROID"); return;
        case SPECTRAL_SPREAD: snprintf(buf, n, "SPECTRAL_SPREAD"); return;
        case SPECTRAL_KURTOSIS: snprintf(buf, n, "SPECTRAL_KURTOSIS"); return;
        case SPECTRAL_SKEW: snprintf(buf, n, "SPECTRAL_SKEW"); return;
        case SPECTRAL_FLATNESS: snprintf(buf, n, "SPECTRAL_FLATNESS"); return;
        case SPECTRAL_STD: snprintf(buf, n, "SPECTRAL_STD"); return;
        case SPECTRAL_ENTROPY: snprintf(buf, n, "SPECTRAL_ENTROPY"); return;
        case DOMINANT_FREQUENCY: snprintf(buf, n, "DOMINANT_FREQUENCY"); return;
        case ZERO_CROSSING_RATE: snprintf(buf, n, "ZERO_CROSSING_RATE"); return;
        case ROOT_MEANS_SQUARED: snprintf(buf, n, "ROOT_MEANS_SQUARED"); return;
        case CREST_FACTOR: snprintf(buf, n, "CREST_FACTOR"); return;
        default: break;
    }

    if (idx >= POWER_SPECTRAL_DENSITY && idx < POWER_SPECTRAL_DENSITY + N_PSD) {
        snprintf(buf, n, "PSD_%d", idx - POWER_SPECTRAL_DENSITY);
        return;
    }

    if (idx >= MEL_FREQUENCY_CEPSTRAL_COEFFICIENT && idx < ZERO_CROSSING_RATE) {
        int local = idx - MEL_FREQUENCY_CEPSTRAL_COEFFICIENT;
        int fam = local / N_MFCC;
        int bin = local % N_MFCC;
        static const char *names[4] = {"MEL_MEAN", "MEL_STD", "MEL_MAX", "MEL_ENTROPY"};
        snprintf(buf, n, "%s_%d", names[fam], bin);
        return;
    }

    if (idx >= ENERGY_ENVELOPE_PEAK_DETECT && idx < Number_AUDIO_Features) {
        snprintf(buf, n, "EEPD_%d", idx - ENERGY_ENVELOPE_PEAK_DETECT);
        return;
    }

    snprintf(buf, n, "AUDIO_%d", idx);
}

static void imu_feature_name(int idx, char *buf, size_t n)
{
    int sig = idx / Num_imu_feat_families;
    int fam = idx % Num_imu_feat_families;
    if (sig >= 0 && sig < 8 && fam >= 0 && fam < Num_imu_feat_families) {
        snprintf(buf, n, "%s_%s", k_imu_signal_name[sig], k_imu_family_name[fam]);
    } else {
        snprintf(buf, n, "IMU_%d", idx);
    }
}

static float audio_model_logit_ref(const float *feats)
{
    float score = 0.0f;

    for (int16_t t = 0; t < AUD_N_TREES; t++) {
        int16_t current_node = 0;
        int16_t child_type = 0;

        for (int16_t n = 0; n < AUD_MAX_NODES; n++) {
            if (feats[audio_feat_comp[t][current_node]] < FXP_TO_FLOAT(audio_values_comp_q16[t][current_node], FXP_PIPE_FRAC)) {
                child_type = audio_children[t][current_node].child_left.type;
                current_node = audio_children[t][current_node].child_left.id;
            } else {
                child_type = audio_children[t][current_node].child_right.type;
                current_node = audio_children[t][current_node].child_right.id;
            }

            if (child_type == AUD_LEAF_T) {
                score += FXP_TO_FLOAT(audio_scores_q16[t][current_node], FXP_PIPE_FRAC);
                break;
            }
        }
    }

    return score;
}

static float imu_model_logit_ref(const float *feats)
{
    float score = 0.0f;

    for (int16_t t = 0; t < IMU_N_TREES; t++) {
        int16_t current_node = 0;
        int16_t child_type = 0;

        for (int16_t n = 0; n < IMU_MAX_NODES; n++) {
            if (feats[imu_feat_comp[t][current_node]] < FXP_TO_FLOAT(imu_values_comp_q16[t][current_node], FXP_PIPE_FRAC)) {
                child_type = imu_children[t][current_node].child_left.type;
                current_node = imu_children[t][current_node].child_left.id;
            } else {
                child_type = imu_children[t][current_node].child_right.type;
                current_node = imu_children[t][current_node].child_right.id;
            }

            if (child_type == IMU_LEAF_T) {
                score += FXP_TO_FLOAT(imu_scores_q16[t][current_node], FXP_PIPE_FRAC);
                break;
            }
        }
    }

    return score;
}

static void build_audio_model_features_f32(const float *feature_array, float *model_feats)
{
    int16_t j = 0;
    for (int16_t i = 0; i < Number_AUDIO_Features; i++) {
        if (audio_features_selector[i]) model_feats[j++] = feature_array[i];
    }
    if (audio_bio_feats_selector[0]) model_feats[j++] = gender;
    if (audio_bio_feats_selector[1]) model_feats[j++] = bmi;
}

static void build_audio_model_features_q16(const fxp_q16_t *feature_array, fxp_q16_t *model_feats)
{
    int16_t j = 0;
    for (int16_t i = 0; i < Number_AUDIO_Features; i++) {
        if (audio_features_selector[i]) model_feats[j++] = feature_array[i];
    }
    if (audio_bio_feats_selector[0]) model_feats[j++] = FXP_FROM_FLOAT(gender, FXP_PIPE_FRAC);
    if (audio_bio_feats_selector[1]) model_feats[j++] = FXP_FROM_FLOAT(bmi, FXP_PIPE_FRAC);
}

static void quantize_audio_model_features_q16(const float *model_feats,
                                              fxp_q16_t *model_feats_q16,
                                              fxp_metric_acc_t *bridge_metric)
{
    for (int i = 0; i < TOT_FEATURES_AUDIO_MODEL_AUDIO; i++) {
        model_feats_q16[i] = (fxp_q16_t)quant_signed_metric(model_feats[i], FXP_PIPE_FRAC, 32, bridge_metric);
        fxp_metric_add(bridge_metric, model_feats[i], FXP_TO_FLOAT(model_feats_q16[i], FXP_PIPE_FRAC));
    }
}

static void dequantize_audio_model_features_f32(const fxp_q16_t *model_feats_q16,
                                                const float *ref_model_feats,
                                                float *model_feats,
                                                fxp_metric_acc_t *bridge_metric)
{
    for (int i = 0; i < TOT_FEATURES_AUDIO_MODEL_AUDIO; i++) {
        model_feats[i] = FXP_TO_FLOAT(model_feats_q16[i], FXP_PIPE_FRAC);
        fxp_metric_add(bridge_metric, ref_model_feats[i], model_feats[i]);
    }
}

static void build_imu_model_features_f32(const float *feature_array, float *model_feats)
{
    int16_t j = 0;
    for (int16_t i = 0; i < Number_IMU_Features; i++) {
        if (imu_features_selector[i]) model_feats[j++] = feature_array[i];
    }
    if (imu_bio_feats_selector[0]) model_feats[j++] = gender;
    if (imu_bio_feats_selector[1]) model_feats[j++] = bmi;
}

static void build_imu_model_features_q16(const fxp_q16_t *feature_array, fxp_q16_t *model_feats)
{
    int16_t j = 0;
    for (int16_t i = 0; i < Number_IMU_Features; i++) {
        if (imu_features_selector[i]) model_feats[j++] = feature_array[i];
    }
    if (imu_bio_feats_selector[0]) model_feats[j++] = FXP_FROM_FLOAT(gender, FXP_PIPE_FRAC);
    if (imu_bio_feats_selector[1]) model_feats[j++] = FXP_FROM_FLOAT(bmi, FXP_PIPE_FRAC);
}

static void quantize_imu_model_features_q16(const float *model_feats,
                                            fxp_q16_t *model_feats_q16,
                                            fxp_metric_acc_t *bridge_metric)
{
    for (int i = 0; i < TOT_FEATURES_IMU_MODEL_IMU; i++) {
        model_feats_q16[i] = (fxp_q16_t)quant_signed_metric(model_feats[i], FXP_PIPE_FRAC, 32, bridge_metric);
        fxp_metric_add(bridge_metric, model_feats[i], FXP_TO_FLOAT(model_feats_q16[i], FXP_PIPE_FRAC));
    }
}

static void dequantize_imu_model_features_f32(const fxp_q16_t *model_feats_q16,
                                              const float *ref_model_feats,
                                              float *model_feats,
                                              fxp_metric_acc_t *bridge_metric)
{
    for (int i = 0; i < TOT_FEATURES_IMU_MODEL_IMU; i++) {
        model_feats[i] = FXP_TO_FLOAT(model_feats_q16[i], FXP_PIPE_FRAC);
        fxp_metric_add(bridge_metric, ref_model_feats[i], model_feats[i]);
    }
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

static void trace_value(int enabled,
                        int window,
                        const char *domain,
                        const char *stage,
                        double ref,
                        double fxp)
{
    if (!enabled) return;
    printf("FXP_TRACE,window=%d,domain=%s,stage=%s,ref=%.17g,fxp=%.17g,pct=%.9g\n",
           window, domain, stage, ref, fxp, fxp_metric_sample_pct(ref, fxp));
}

static kiss_fft_scalar kiss_scalar_from_unit_float(float x)
{
    double y = (double)x * KISS_SCALAR_SCALE;
    if (y > KISS_SCALAR_SCALE) y = KISS_SCALAR_SCALE;
    if (y < -KISS_SCALAR_SCALE) y = -KISS_SCALAR_SCALE;
    return (kiss_fft_scalar)llround(y);
}

static double kiss_scalar_to_double(kiss_fft_scalar x)
{
    return (double)x / KISS_SCALAR_SCALE;
}

static uint32_t kiss_lcg_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static float kiss_noise_sample(uint32_t *state)
{
    uint32_t v = kiss_lcg_next(state);
    double unit = (double)(v >> 8) / (double)0x00FFFFFFu;
    return (float)(0.85 * ((2.0 * unit) - 1.0));
}

static float kiss_signal_sample(const char *signal, int i, int nfft, uint32_t *noise_state)
{
    const double amp = 0.85;
    double t = (double)i;
    double n = (double)nfft;

    if (strcmp(signal, "impulse") == 0) {
        return (i == 0) ? (float)amp : 0.0f;
    }
    if (strcmp(signal, "tone_bin7") == 0) {
        return (float)(amp * sin((2.0 * M_PI * 7.0 * t) / n));
    }
    if (strcmp(signal, "dual_5_37") == 0) {
        return (float)((0.6 * amp * sin((2.0 * M_PI * 5.0 * t) / n)) +
                       (0.4 * amp * sin(((2.0 * M_PI * 37.0 * t) / n) + 0.3)));
    }
    if (strcmp(signal, "chirp") == 0) {
        double f = 3.0 + ((120.0 - 3.0) * (t / n));
        return (float)(amp * sin((2.0 * M_PI * f * t) / n));
    }
    if (strcmp(signal, "noise") == 0) {
        return kiss_noise_sample(noise_state);
    }

    return 0.0f;
}

static int run_kissfft_bins(int argc, char **argv)
{
    int nfft = arg_value(argc, argv, "--nfft", 900);
    const char *signal = arg_string(argc, argv, "--signal", "impulse");

    if (nfft <= 0 || (nfft % 2) != 0) {
        fprintf(stderr, "KissFFT harness requires a positive even --nfft value.\n");
        return 2;
    }

    kiss_fft_scalar *time_data = (kiss_fft_scalar *)malloc((size_t)nfft * sizeof(*time_data));
    kiss_fft_cpx *freq_data = (kiss_fft_cpx *)malloc((size_t)(nfft / 2 + 1) * sizeof(*freq_data));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!time_data || !freq_data || !cfg) {
        fprintf(stderr, "KissFFT harness allocation failed (nfft=%d).\n", nfft);
        free(time_data);
        free(freq_data);
        free(cfg);
        return 1;
    }

    uint32_t noise_state = 12345u + (uint32_t)nfft;
    for (int i = 0; i < nfft; i++) {
        time_data[i] = kiss_scalar_from_unit_float(kiss_signal_sample(signal, i, nfft, &noise_state));
    }

    kiss_fftr(cfg, time_data, freq_data);
    for (int k = 0; k <= nfft / 2; k++) {
        printf("KISSFFT_BIN,nfft=%d,signal=%s,bin=%d,re=%.17g,im=%.17g\n",
               nfft, signal, k,
               kiss_scalar_to_double(freq_data[k].r),
               kiss_scalar_to_double(freq_data[k].i));
    }

    free(time_data);
    free(freq_data);
    free(cfg);
    return 0;
}

static void run_qformat_sensitivity_audio(int max_windows)
{
    static const int input_frac[] = {12, 13, 14, 15};
    static const int feature_frac[] = {12, 14, 16, 18, 20};
    const int n_input = (int)(sizeof(input_frac) / sizeof(input_frac[0]));
    const int n_feature = (int)(sizeof(feature_frac) / sizeof(feature_frac[0]));

    fxp_metric_acc_t input_scores[n_input];
    fxp_metric_acc_t feature_scores[n_feature];
    for (int i = 0; i < n_input; i++) fxp_metric_init(&input_scores[i]);
    for (int i = 0; i < n_feature; i++) fxp_metric_init(&feature_scores[i]);

    int n_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    if (max_windows > 0 && max_windows < n_wins) n_wins = max_windows;

    float *ref_feats = (float *)malloc((size_t)Number_AUDIO_Features * sizeof(float));
    float *tmp_feats = (float *)malloc((size_t)Number_AUDIO_Features * sizeof(float));
    float *win_q = (float *)malloc((size_t)WINDOW_SAMP_AUDIO * sizeof(float));
    float ref_model[TOT_FEATURES_AUDIO_MODEL_AUDIO];
    float tmp_model[TOT_FEATURES_AUDIO_MODEL_AUDIO];

    if (!ref_feats || !tmp_feats || !win_q) goto done;

    for (int w = 0; w < n_wins; w++) {
        const float *sig = &audio_in.air[w * AUDIO_STEP];
        compute_audio_float_features(audio_features_selector, sig, WINDOW_SAMP_AUDIO, AUDIO_FS, ref_feats);
        build_audio_model_features_f32(ref_feats, ref_model);
        float ref_score = audio_model_logit_ref(ref_model);

        for (int c = 0; c < n_input; c++) {
            for (int i = 0; i < WINDOW_SAMP_AUDIO; i++) {
                int32_t q = quant_signed(sig[i], input_frac[c], 16);
                win_q[i] = q_to_float_signed(q, input_frac[c]);
            }
            compute_audio_float_features(audio_features_selector, win_q, WINDOW_SAMP_AUDIO, AUDIO_FS, tmp_feats);
            build_audio_model_features_f32(tmp_feats, tmp_model);
            fxp_metric_add(&input_scores[c], ref_score, audio_model_logit_ref(tmp_model));
        }

        for (int c = 0; c < n_feature; c++) {
            memcpy(tmp_model, ref_model, sizeof(tmp_model));
            for (int i = 0; i < TOT_FEATURES_AUDIO_MODEL_AUDIO; i++) {
                int32_t q = quant_signed(tmp_model[i], feature_frac[c], 32);
                tmp_model[i] = q_to_float_signed(q, feature_frac[c]);
            }
            fxp_metric_add(&feature_scores[c], ref_score, audio_model_logit_ref(tmp_model));
        }
    }

    for (int c = 0; c < n_input; c++) {
        printf("FXP_QSWEEP,domain=audio,location=input,candidate_frac=%d,n=%d,score_rel_rmse_pct=%.9g,score_wape_pct=%.9g,score_max_abs_pct=%.9g\n",
               input_frac[c], input_scores[c].n,
               fxp_metric_rel_rmse_pct(&input_scores[c]),
               fxp_metric_wape_pct(&input_scores[c]),
               fxp_metric_max_abs_pct(&input_scores[c]));
    }
    for (int c = 0; c < n_feature; c++) {
        printf("FXP_QSWEEP,domain=audio,location=model_feature_pipe,candidate_frac=%d,n=%d,score_rel_rmse_pct=%.9g,score_wape_pct=%.9g,score_max_abs_pct=%.9g\n",
               feature_frac[c], feature_scores[c].n,
               fxp_metric_rel_rmse_pct(&feature_scores[c]),
               fxp_metric_wape_pct(&feature_scores[c]),
               fxp_metric_max_abs_pct(&feature_scores[c]));
    }

done:
    free(ref_feats);
    free(tmp_feats);
    free(win_q);
}

static void run_qformat_sensitivity_imu(int max_windows)
{
    static const int input_frac[] = {3, 4, 5, 6, 7};
    static const int feature_frac[] = {12, 14, 16, 18, 20};
    const int n_input = (int)(sizeof(input_frac) / sizeof(input_frac[0]));
    const int n_feature = (int)(sizeof(feature_frac) / sizeof(feature_frac[0]));

    fxp_metric_acc_t input_scores[n_input];
    fxp_metric_acc_t feature_scores[n_feature];
    for (int i = 0; i < n_input; i++) fxp_metric_init(&input_scores[i]);
    for (int i = 0; i < n_feature; i++) fxp_metric_init(&feature_scores[i]);

    int n_wins = ((IMU_LEN - WINDOW_SAMP_IMU) / IMU_STEP) + 1;
    if (max_windows > 0 && max_windows < n_wins) n_wins = max_windows;

    float (*win_q)[Num_IMU_signals] = malloc((size_t)WINDOW_SAMP_IMU * sizeof(*win_q));
    float ref_feats[Number_IMU_Features];
    float tmp_feats[Number_IMU_Features];
    float ref_model[TOT_FEATURES_IMU_MODEL_IMU];
    float tmp_model[TOT_FEATURES_IMU_MODEL_IMU];
    if (!win_q) return;

    for (int w = 0; w < n_wins; w++) {
        const float (*sig)[Num_IMU_signals] = &imu_in[w * IMU_STEP];
        compute_imu_float_features(imu_features_selector, sig, WINDOW_SAMP_IMU, ref_feats);
        build_imu_model_features_f32(ref_feats, ref_model);
        float ref_score = imu_model_logit_ref(ref_model);

        for (int c = 0; c < n_input; c++) {
            for (int i = 0; i < WINDOW_SAMP_IMU; i++) {
                for (int ax = 0; ax < Num_IMU_signals; ax++) {
                    int32_t q = quant_signed(sig[i][ax], input_frac[c], 16);
                    win_q[i][ax] = q_to_float_signed(q, input_frac[c]);
                }
            }
            compute_imu_float_features(imu_features_selector, win_q, WINDOW_SAMP_IMU, tmp_feats);
            build_imu_model_features_f32(tmp_feats, tmp_model);
            fxp_metric_add(&input_scores[c], ref_score, imu_model_logit_ref(tmp_model));
        }

        for (int c = 0; c < n_feature; c++) {
            memcpy(tmp_model, ref_model, sizeof(tmp_model));
            for (int i = 0; i < TOT_FEATURES_IMU_MODEL_IMU; i++) {
                int32_t q = quant_signed(tmp_model[i], feature_frac[c], 32);
                tmp_model[i] = q_to_float_signed(q, feature_frac[c]);
            }
            fxp_metric_add(&feature_scores[c], ref_score, imu_model_logit_ref(tmp_model));
        }
    }

    for (int c = 0; c < n_input; c++) {
        printf("FXP_QSWEEP,domain=imu,location=input,candidate_frac=%d,n=%d,score_rel_rmse_pct=%.9g,score_wape_pct=%.9g,score_max_abs_pct=%.9g\n",
               input_frac[c], input_scores[c].n,
               fxp_metric_rel_rmse_pct(&input_scores[c]),
               fxp_metric_wape_pct(&input_scores[c]),
               fxp_metric_max_abs_pct(&input_scores[c]));
    }
    for (int c = 0; c < n_feature; c++) {
        printf("FXP_QSWEEP,domain=imu,location=model_feature_pipe,candidate_frac=%d,n=%d,score_rel_rmse_pct=%.9g,score_wape_pct=%.9g,score_max_abs_pct=%.9g\n",
               feature_frac[c], feature_scores[c].n,
               fxp_metric_rel_rmse_pct(&feature_scores[c]),
               fxp_metric_wape_pct(&feature_scores[c]),
               fxp_metric_max_abs_pct(&feature_scores[c]));
    }

    free(win_q);
}

int main(int argc, char **argv)
{
    if (arg_flag(argc, argv, "--kissfft-bins")) {
        return run_kissfft_bins(argc, argv);
    }
    if (arg_flag(argc, argv, "--list-descriptors")) {
        print_descriptor_table();
        return 0;
    }

    int max_windows = arg_value(argc, argv, "--max-windows", 4);
    int trace_limit = arg_value(argc, argv, "--trace-limit", 0);
    int do_sweep = arg_flag(argc, argv, "--sweep");
    fxp_harness_backend_config_t backend_cfg;
    parse_backend_config(argc, argv, &backend_cfg);
    int hybrid_mode = is_hybrid_config(&backend_cfg);

    fxp_metric_acc_t audio_input_metric;
    fxp_metric_acc_t imu_input_metric;
    fxp_metric_acc_t audio_feature_metrics[Number_AUDIO_Features];
    fxp_metric_acc_t imu_feature_metrics[Number_IMU_Features];
    fxp_metric_acc_t audio_feature_block;
    fxp_metric_acc_t imu_feature_block;
    fxp_metric_acc_t imu_l2a_metric;
    fxp_metric_acc_t imu_l2g_metric;
    fxp_metric_acc_t audio_score_metric;
    fxp_metric_acc_t imu_score_metric;
    fxp_metric_acc_t audio_model_bridge_metric;
    fxp_metric_acc_t imu_model_bridge_metric;
    fxp_metric_acc_t audio_hybrid_score_metric;
    fxp_metric_acc_t imu_hybrid_score_metric;

    fxp_metric_init(&audio_input_metric);
    fxp_metric_init(&imu_input_metric);
    fxp_metric_init(&audio_feature_block);
    fxp_metric_init(&imu_feature_block);
    fxp_metric_init(&imu_l2a_metric);
    fxp_metric_init(&imu_l2g_metric);
    fxp_metric_init(&audio_score_metric);
    fxp_metric_init(&imu_score_metric);
    fxp_metric_init(&audio_model_bridge_metric);
    fxp_metric_init(&imu_model_bridge_metric);
    fxp_metric_init(&audio_hybrid_score_metric);
    fxp_metric_init(&imu_hybrid_score_metric);
    for (int i = 0; i < Number_AUDIO_Features; i++) fxp_metric_init(&audio_feature_metrics[i]);
    for (int i = 0; i < Number_IMU_Features; i++) fxp_metric_init(&imu_feature_metrics[i]);

    int n_audio_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    int n_imu_wins = ((IMU_LEN - WINDOW_SAMP_IMU) / IMU_STEP) + 1;
    if (max_windows > 0 && max_windows < n_audio_wins) n_audio_wins = max_windows;
    if (max_windows > 0 && max_windows < n_imu_wins) n_imu_wins = max_windows;

    float *audio_ref_feats = (float *)malloc((size_t)Number_AUDIO_Features * sizeof(float));
    fxp_q16_t *audio_fxp_feats = (fxp_q16_t *)malloc((size_t)Number_AUDIO_Features * sizeof(fxp_q16_t));
    int16_t *audio_q14 = (int16_t *)malloc((size_t)WINDOW_SAMP_AUDIO * sizeof(int16_t));

    for (int w = 0; w < n_audio_wins; w++) {
        const float *sig = &audio_in.air[w * AUDIO_STEP];
        memset(audio_fxp_feats, 0, (size_t)Number_AUDIO_Features * sizeof(fxp_q16_t));

        for (int i = 0; i < WINDOW_SAMP_AUDIO; i++) {
            audio_q14[i] = (int16_t)quant_signed_metric(sig[i], FXP_FRAC_AUDIO_INPUT, 16, &audio_input_metric);
            fxp_metric_add(&audio_input_metric, sig[i], FXP_TO_FLOAT(audio_q14[i], FXP_FRAC_AUDIO_INPUT));
        }

        compute_audio_float_features(audio_features_selector, sig, WINDOW_SAMP_AUDIO, AUDIO_FS, audio_ref_feats);
        audio_features_fxp_q16_from_q14(audio_features_selector, audio_q14, WINDOW_SAMP_AUDIO, AUDIO_FS, audio_fxp_feats);

        int traced = (w < trace_limit);
        int traced_count = 0;
        for (int i = 0; i < Number_AUDIO_Features; i++) {
            if (!audio_features_selector[i]) continue;
            float fxp_v = FXP_TO_FLOAT(audio_fxp_feats[i], FXP_PIPE_FRAC);
            fxp_metric_add(&audio_feature_metrics[i], audio_ref_feats[i], fxp_v);
            fxp_metric_add(&audio_feature_block, audio_ref_feats[i], fxp_v);
            if (traced && traced_count < MAX_AUDIO_TRACE_FEATURES) {
                char name[64];
                audio_feature_name(i, name, sizeof(name));
                trace_value(1, w, "audio", name, audio_ref_feats[i], fxp_v);
                traced_count++;
            }
        }

        float ref_model[TOT_FEATURES_AUDIO_MODEL_AUDIO];
        fxp_q16_t fxp_model[TOT_FEATURES_AUDIO_MODEL_AUDIO];
        build_audio_model_features_f32(audio_ref_feats, ref_model);
        build_audio_model_features_q16(audio_fxp_feats, fxp_model);
        float ref_score = audio_model_logit_ref(ref_model);
        fxp_metric_add(&audio_score_metric,
                       ref_score,
                       FXP_TO_FLOAT(audio_predict_q16(fxp_model), FXP_PIPE_FRAC));

        if (hybrid_mode) {
            float hybrid_model_f32[TOT_FEATURES_AUDIO_MODEL_AUDIO];
            fxp_q16_t hybrid_model_q16[TOT_FEATURES_AUDIO_MODEL_AUDIO];
            double hybrid_score = 0.0;

            if (backend_cfg.audio_features == FXP_HARNESS_BACKEND_FLOAT &&
                backend_cfg.audio_model == FXP_HARNESS_BACKEND_FLOAT) {
                hybrid_score = audio_model_logit_ref(ref_model);
            } else if (backend_cfg.audio_features == FXP_HARNESS_BACKEND_FLOAT &&
                       backend_cfg.audio_model == FXP_HARNESS_BACKEND_FXP) {
                quantize_audio_model_features_q16(ref_model, hybrid_model_q16, &audio_model_bridge_metric);
                hybrid_score = FXP_TO_FLOAT(audio_predict_q16(hybrid_model_q16), FXP_PIPE_FRAC);
            } else if (backend_cfg.audio_features == FXP_HARNESS_BACKEND_FXP &&
                       backend_cfg.audio_model == FXP_HARNESS_BACKEND_FLOAT) {
                dequantize_audio_model_features_f32(fxp_model, ref_model, hybrid_model_f32, &audio_model_bridge_metric);
                hybrid_score = audio_model_logit_ref(hybrid_model_f32);
            } else {
                memcpy(hybrid_model_q16, fxp_model, sizeof(hybrid_model_q16));
                hybrid_score = FXP_TO_FLOAT(audio_predict_q16(hybrid_model_q16), FXP_PIPE_FRAC);
            }

            fxp_metric_add(&audio_hybrid_score_metric, ref_score, hybrid_score);
        }
    }

    free(audio_ref_feats);
    free(audio_fxp_feats);
    free(audio_q14);

    float imu_ref_feats[Number_IMU_Features];
    fxp_q16_t imu_fxp_feats[Number_IMU_Features];
    q11_5_t (*imu_q5)[Num_IMU_signals] = malloc((size_t)WINDOW_SAMP_IMU * sizeof(*imu_q5));

    for (int w = 0; w < n_imu_wins; w++) {
        const float (*sig)[Num_IMU_signals] = &imu_in[w * IMU_STEP];
        memset(imu_fxp_feats, 0, sizeof(imu_fxp_feats));

        for (int i = 0; i < WINDOW_SAMP_IMU; i++) {
            for (int ax = 0; ax < Num_IMU_signals; ax++) {
                imu_q5[i][ax] = (q11_5_t)quant_signed_metric(sig[i][ax], FXP_FRAC_IMU_RAW, 16, &imu_input_metric);
                fxp_metric_add(&imu_input_metric, sig[i][ax], FXP_TO_FLOAT(imu_q5[i][ax], FXP_FRAC_IMU_RAW));
            }

            float ref_l2a = sqrtf(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1] + sig[i][2] * sig[i][2]);
            float ref_l2g = sqrtf(sig[i][3] * sig[i][3] + sig[i][4] * sig[i][4] + sig[i][5] * sig[i][5]);
            uq10_6_t l2a_q6 = fxp_l2_norm_accel_from_raw(imu_q5[i][0], imu_q5[i][1], imu_q5[i][2]);
            uq5_11_t l2g_q11 = fxp_l2_norm_gyro_from_raw(imu_q5[i][3], imu_q5[i][4], imu_q5[i][5]);
            fxp_metric_add(&imu_l2a_metric, ref_l2a, FXP_TO_FLOAT(l2a_q6, FXP_FRAC_IMU_L2A));
            fxp_metric_add(&imu_l2g_metric, ref_l2g, FXP_TO_FLOAT(l2g_q11, FXP_FRAC_IMU_L2G));
        }

        compute_imu_float_features(imu_features_selector, sig, WINDOW_SAMP_IMU, imu_ref_feats);
        imu_features_fxp_q16_from_raw(imu_features_selector, imu_q5, WINDOW_SAMP_IMU, imu_fxp_feats);

        int traced = (w < trace_limit);
        int traced_count = 0;
        for (int i = 0; i < Number_IMU_Features; i++) {
            if (!imu_features_selector[i]) continue;
            float fxp_v = FXP_TO_FLOAT(imu_fxp_feats[i], FXP_PIPE_FRAC);
            fxp_metric_add(&imu_feature_metrics[i], imu_ref_feats[i], fxp_v);
            fxp_metric_add(&imu_feature_block, imu_ref_feats[i], fxp_v);
            if (traced && traced_count < MAX_IMU_TRACE_FEATURES) {
                char name[64];
                imu_feature_name(i, name, sizeof(name));
                trace_value(1, w, "imu", name, imu_ref_feats[i], fxp_v);
                traced_count++;
            }
        }

        float ref_model[TOT_FEATURES_IMU_MODEL_IMU];
        fxp_q16_t fxp_model[TOT_FEATURES_IMU_MODEL_IMU];
        build_imu_model_features_f32(imu_ref_feats, ref_model);
        build_imu_model_features_q16(imu_fxp_feats, fxp_model);
        float ref_score = imu_model_logit_ref(ref_model);
        fxp_metric_add(&imu_score_metric,
                       ref_score,
                       FXP_TO_FLOAT(imu_predict_q16(fxp_model), FXP_PIPE_FRAC));

        if (hybrid_mode) {
            float hybrid_model_f32[TOT_FEATURES_IMU_MODEL_IMU];
            fxp_q16_t hybrid_model_q16[TOT_FEATURES_IMU_MODEL_IMU];
            double hybrid_score = 0.0;

            if (backend_cfg.imu_features == FXP_HARNESS_BACKEND_FLOAT &&
                backend_cfg.imu_model == FXP_HARNESS_BACKEND_FLOAT) {
                hybrid_score = imu_model_logit_ref(ref_model);
            } else if (backend_cfg.imu_features == FXP_HARNESS_BACKEND_FLOAT &&
                       backend_cfg.imu_model == FXP_HARNESS_BACKEND_FXP) {
                quantize_imu_model_features_q16(ref_model, hybrid_model_q16, &imu_model_bridge_metric);
                hybrid_score = FXP_TO_FLOAT(imu_predict_q16(hybrid_model_q16), FXP_PIPE_FRAC);
            } else if (backend_cfg.imu_features == FXP_HARNESS_BACKEND_FXP &&
                       backend_cfg.imu_model == FXP_HARNESS_BACKEND_FLOAT) {
                dequantize_imu_model_features_f32(fxp_model, ref_model, hybrid_model_f32, &imu_model_bridge_metric);
                hybrid_score = imu_model_logit_ref(hybrid_model_f32);
            } else {
                memcpy(hybrid_model_q16, fxp_model, sizeof(hybrid_model_q16));
                hybrid_score = FXP_TO_FLOAT(imu_predict_q16(hybrid_model_q16), FXP_PIPE_FRAC);
            }

            fxp_metric_add(&imu_hybrid_score_metric, ref_score, hybrid_score);
        }
    }

    free(imu_q5);

    fxp_metric_print_stage("FXP_STAGE", "conversion", "audio", "input_float_to_Q1.14", "Q1.14", &audio_input_metric);
    fxp_metric_print_stage("FXP_STAGE", "conversion", "imu", "input_float_to_Q11.5", "Q11.5", &imu_input_metric);
    fxp_metric_print_stage("FXP_STAGE", "block", "audio", "selected_feature_outputs_to_Q16", "Q16", &audio_feature_block);
    fxp_metric_print_stage("FXP_STAGE", "block", "imu", "selected_feature_outputs_to_Q16", "Q16", &imu_feature_block);
    fxp_metric_print_stage_ex("FXP_STAGE", "intermediate", "imu", "l2_norm_accel", "raw_square_sum_sqrt", "fxp", "UQ10.6", &imu_l2a_metric);
    fxp_metric_print_stage_ex("FXP_STAGE", "intermediate", "imu", "l2_norm_gyro", "raw_square_sum_sqrt", "fxp", "UQ5.11", &imu_l2g_metric);
    fxp_metric_print_stage("FXP_STAGE", "end_to_end", "audio", "model_logit", "Q16", &audio_score_metric);
    fxp_metric_print_stage("FXP_STAGE", "end_to_end", "imu", "model_logit", "Q16", &imu_score_metric);

    if (hybrid_mode) {
        printf("FXP_HYBRID_CONFIG,audio_features=%s,audio_model=%s,imu_features=%s,imu_model=%s\n",
               backend_name(backend_cfg.audio_features),
               backend_name(backend_cfg.audio_model),
               backend_name(backend_cfg.imu_features),
               backend_name(backend_cfg.imu_model));

        if (audio_model_bridge_metric.n > 0) {
            fxp_metric_print_stage_ex("FXP_STAGE",
                                      "hybrid-bridge",
                                      "audio",
                                      "model_input",
                                      "model_feature_bridge",
                                      backend_name(backend_cfg.audio_model),
                                      "Q16",
                                      &audio_model_bridge_metric);
        }
        if (imu_model_bridge_metric.n > 0) {
            fxp_metric_print_stage_ex("FXP_STAGE",
                                      "hybrid-bridge",
                                      "imu",
                                      "model_input",
                                      "model_feature_bridge",
                                      backend_name(backend_cfg.imu_model),
                                      "Q16",
                                      &imu_model_bridge_metric);
        }
        fxp_metric_print_stage_ex("FXP_STAGE",
                                  "hybrid",
                                  "audio",
                                  "model",
                                  "model_logit",
                                  backend_name(backend_cfg.audio_model),
                                  (backend_cfg.audio_model == FXP_HARNESS_BACKEND_FXP) ? "Q16" : "float",
                                  &audio_hybrid_score_metric);
        fxp_metric_print_stage_ex("FXP_STAGE",
                                  "hybrid",
                                  "imu",
                                  "model",
                                  "model_logit",
                                  backend_name(backend_cfg.imu_model),
                                  (backend_cfg.imu_model == FXP_HARNESS_BACKEND_FXP) ? "Q16" : "float",
                                  &imu_hybrid_score_metric);
    }

    for (int i = 0; i < Number_AUDIO_Features; i++) {
        if (!audio_features_selector[i] || audio_feature_metrics[i].n <= 0) continue;
        char name[64];
        audio_feature_name(i, name, sizeof(name));
        fxp_metric_print_stage("FXP_STAGE", "single-kernel", "audio", name, "Q16", &audio_feature_metrics[i]);
    }
    for (int i = 0; i < Number_IMU_Features; i++) {
        if (!imu_features_selector[i] || imu_feature_metrics[i].n <= 0) continue;
        char name[64];
        imu_feature_name(i, name, sizeof(name));
        fxp_metric_print_stage("FXP_STAGE", "single-kernel", "imu", name, "Q16", &imu_feature_metrics[i]);
    }

    if (do_sweep) {
        run_qformat_sensitivity_audio(max_windows);
        run_qformat_sensitivity_imu(max_windows);
    }

    return 0;
}

#endif
