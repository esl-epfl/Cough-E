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
#if !defined(PROGRESSIVE_INPUTS_INJECTED)
#include <input_data/bio_input_55502.h>
#endif

#include <audio_features.h>
#include <audio_model.h>
#include <core/fxp_core.h>
#include <feature_extraction.h>
#include <imu_features.h>
#include <imu_model.h>
#include <postprocessing.h>

void audio_crest_factor(const int8_t *features_selector,
                        const int16_t *sig,
                        int16_t len,
                        fxp_feat_t *feats);

#define TIME_DEADLINE_OUTPUT_NUM 3U
#define TIME_DEADLINE_OUTPUT_DEN 2U
#define TIME_DEADLINE_OUTPUT_TICKS ((uint32_t)(((uint64_t)TIME_DEADLINE_OUTPUT_NUM * AUDIO_FS) / TIME_DEADLINE_OUTPUT_DEN))
#define N_MAX_WIND_AUD 4U
#define AUDIO_TH 0.3f
#define IMU_TH 0.05f

typedef enum {
    IMU_MODEL_LOCAL,
    AUDIO_MODEL_LOCAL,
} local_model_t;

typedef enum {
    NON_COUGH_OUT_LOCAL,
    COUGH_OUT_LOCAL,
} local_class_t;

typedef struct {
    local_model_t model;
    local_class_t model_cls_out;
    uint32_t last_output_tick;
    uint32_t ticks_from_last_output;
    uint32_t window_start_tick;
    uint32_t last_window_start_tick;
    uint8_t n_winds_aud;
} local_fsm_t;

static local_fsm_t fsm_state;

typedef enum {
    BLOCK_AUDIO_FFT,
    BLOCK_AUDIO_PSD,
    BLOCK_AUDIO_MEL,
    BLOCK_AUDIO_SCALAR,
    BLOCK_AUDIO_ALL,
    BLOCK_IMU_RAW,
    BLOCK_IMU_L2,
    BLOCK_IMU_ALL,
} progressive_block_t;

extern void audio_fft_features(const int8_t *features_selector,
                               const int16_t *sig_q14,
                               int16_t len,
                               int16_t fs,
                               fxp_feat_t *feats);
extern void audio_psd_features(const int8_t *features_selector,
                               const int16_t *sig_q14,
                               int16_t sig_len,
                               int16_t fs,
                               fxp_feat_t *feats);
extern void audio_mel_features(const int8_t *features_selector,
                               const int16_t *sig_q14,
                               int16_t len,
                               fxp_feat_t *feats);
extern void imu_run_raw_features(const int8_t *features_selector,
                                 const q11_5_t *sig,
                                 int16_t len,
                                 fxp_feat_t *feats);
extern void imu_run_l2a_features(const int8_t *features_selector,
                                 const uq10_6_t *sig,
                                 int16_t len,
                                 fxp_feat_t *feats);
extern void imu_run_l2g_features(const int8_t *features_selector,
                                 const uq5_11_t *sig,
                                 int16_t len,
                                 fxp_feat_t *feats);
extern uq10_6_t imu_l2a(q11_5_t ax, q11_5_t ay, q11_5_t az);
extern uq5_11_t imu_l2g(q11_5_t gx, q11_5_t gy, q11_5_t gz);

static int parse_block(const char *name, progressive_block_t *block)
{
    if (!name || !block) return 0;
    if (strcmp(name, "audio_fft") == 0) *block = BLOCK_AUDIO_FFT;
    else if (strcmp(name, "audio_psd") == 0) *block = BLOCK_AUDIO_PSD;
    else if (strcmp(name, "audio_mel") == 0) *block = BLOCK_AUDIO_MEL;
    else if (strcmp(name, "audio_scalar") == 0) *block = BLOCK_AUDIO_SCALAR;
    else if (strcmp(name, "audio_all") == 0) *block = BLOCK_AUDIO_ALL;
    else if (strcmp(name, "imu_raw") == 0) *block = BLOCK_IMU_RAW;
    else if (strcmp(name, "imu_l2") == 0) *block = BLOCK_IMU_L2;
    else if (strcmp(name, "imu_all") == 0) *block = BLOCK_IMU_ALL;
    else return 0;
    return 1;
}

static int block_is_audio(progressive_block_t block)
{
    return block == BLOCK_AUDIO_FFT || block == BLOCK_AUDIO_PSD ||
           block == BLOCK_AUDIO_MEL || block == BLOCK_AUDIO_SCALAR ||
           block == BLOCK_AUDIO_ALL;
}

static int block_is_imu(progressive_block_t block)
{
    return block == BLOCK_IMU_RAW || block == BLOCK_IMU_L2 || block == BLOCK_IMU_ALL;
}

static int any_required(const int8_t *selector, uint16_t start, uint16_t end)
{
    for (uint16_t i = start; i <= end; i++) {
        if (selector[i] == 1) return 1;
    }
    return 0;
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

static void copy_audio_feature_range(float *dst, const fxp_feat_t *src, uint16_t start, uint16_t end)
{
    for (uint16_t i = start; i <= end; i++) {
        dst[i] = audio_feat_to_float(src[i], i);
    }
}

static void apply_audio_block(progressive_block_t block,
                              const float *sig,
                              int16_t len,
                              int16_t fs,
                              float *features)
{
    if (!block_is_audio(block)) return;

    int16_t *sig_q14 = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    fxp_feat_t *fxp_feats = (fxp_feat_t *)calloc((size_t)Number_AUDIO_Features, sizeof(fxp_feat_t));
    if (!sig_q14 || !fxp_feats) {
        free(sig_q14);
        free(fxp_feats);
        return;
    }

    for (int16_t i = 0; i < len; i++) sig_q14[i] = FXP_AUDIO_FROM_FLOAT(sig[i]);

    if (block == BLOCK_AUDIO_FFT || block == BLOCK_AUDIO_ALL) {
        audio_fft_features(audio_features_selector, sig_q14, len, fs, fxp_feats);
        if (audio_features_selector[SPECTRAL_ROLLOFF]) features[SPECTRAL_ROLLOFF] = audio_feat_to_float(fxp_feats[SPECTRAL_ROLLOFF], SPECTRAL_ROLLOFF);
        if (audio_features_selector[SPECTRAL_SPREAD]) features[SPECTRAL_SPREAD] = audio_feat_to_float(fxp_feats[SPECTRAL_SPREAD], SPECTRAL_SPREAD);
        if (audio_features_selector[SPECTRAL_KURTOSIS]) features[SPECTRAL_KURTOSIS] = audio_feat_to_float(fxp_feats[SPECTRAL_KURTOSIS], SPECTRAL_KURTOSIS);
    }

    if (block == BLOCK_AUDIO_PSD || block == BLOCK_AUDIO_ALL) {
        audio_psd_features(audio_features_selector, sig_q14, len, fs, fxp_feats);
        if (audio_features_selector[SPECTRAL_FLATNESS]) features[SPECTRAL_FLATNESS] = audio_feat_to_float(fxp_feats[SPECTRAL_FLATNESS], SPECTRAL_FLATNESS);
        if (audio_features_selector[DOMINANT_FREQUENCY]) features[DOMINANT_FREQUENCY] = audio_feat_to_float(fxp_feats[DOMINANT_FREQUENCY], DOMINANT_FREQUENCY);
        copy_audio_feature_range(features, fxp_feats, POWER_SPECTRAL_DENSITY, POWER_SPECTRAL_DENSITY + N_PSD - 1);
    }

    if (block == BLOCK_AUDIO_MEL || block == BLOCK_AUDIO_ALL) {
        audio_mel_features(audio_features_selector, sig_q14, len, fxp_feats);
        copy_audio_feature_range(features, fxp_feats, MEL_FREQUENCY_CEPSTRAL_COEFFICIENT, ZERO_CROSSING_RATE - 1);
    }

    if (block == BLOCK_AUDIO_SCALAR || block == BLOCK_AUDIO_ALL) {
        audio_crest_factor(audio_features_selector, sig_q14, len, fxp_feats);
        if (audio_features_selector[CREST_FACTOR]) features[CREST_FACTOR] = audio_feat_to_float(fxp_feats[CREST_FACTOR], CREST_FACTOR);
    }

    free(sig_q14);
    free(fxp_feats);
}

static void copy_imu_local(float *dst, const fxp_feat_t *src, int base)
{
    for (uint16_t local = 0; local < Num_imu_feat_families; local++) {
        if (imu_features_selector[base + local]) {
            dst[base + local] = imu_feat_to_float(src[local], (uint16_t)(base + local));
        }
    }
}

static void apply_imu_block(progressive_block_t block,
                            const float sig[][Num_IMU_signals],
                            int16_t len,
                            float *features)
{
    if (!block_is_imu(block)) return;

    q11_5_t (*raw_q5)[Num_IMU_signals] = malloc((size_t)len * sizeof(*raw_q5));
    q11_5_t *axis = (q11_5_t *)malloc((size_t)len * sizeof(q11_5_t));
    uq10_6_t *l2a = (uq10_6_t *)malloc((size_t)len * sizeof(uq10_6_t));
    uq5_11_t *l2g = (uq5_11_t *)malloc((size_t)len * sizeof(uq5_11_t));
    fxp_feat_t local_feats[Num_imu_feat_families];
    if (!raw_q5 || !axis || !l2a || !l2g) {
        free(raw_q5);
        free(axis);
        free(l2a);
        free(l2g);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        for (int ax = 0; ax < Num_IMU_signals; ax++) raw_q5[i][ax] = FXP_IMU_RAW_FROM_FLOAT(sig[i][ax]);
        l2a[i] = imu_l2a(raw_q5[i][0], raw_q5[i][1], raw_q5[i][2]);
        l2g[i] = imu_l2g(raw_q5[i][3], raw_q5[i][4], raw_q5[i][5]);
    }

    if (block == BLOCK_IMU_RAW || block == BLOCK_IMU_ALL) {
        static const int axes[Num_IMU_signals] = {
            ACCELEROMETER_X, ACCELEROMETER_Y, ACCELEROMETER_Z,
            GYROSCOPE_Y, GYROSCOPE_P, GYROSCOPE_R
        };
        static const int bases[Num_IMU_signals] = {
            ACCEL_X_FEAT, ACCEL_Y_FEAT, ACCEL_Z_FEAT,
            GYRO_Y_FEAT, GYRO_P_FEAT, GYRO_R_FEAT
        };
        for (int s = 0; s < Num_IMU_signals; s++) {
            int base = bases[s];
            if (!any_required(imu_features_selector, base, (uint16_t)(base + Num_imu_feat_families - 1))) continue;
            for (int16_t i = 0; i < len; i++) axis[i] = raw_q5[i][axes[s]];
            memset(local_feats, 0, sizeof(local_feats));
            imu_run_raw_features(&imu_features_selector[base], axis, len, local_feats);
            copy_imu_local(features, local_feats, base);
        }
    }

    if (block == BLOCK_IMU_L2 || block == BLOCK_IMU_ALL) {
        if (any_required(imu_features_selector, ACCEL_COMBO, (uint16_t)(ACCEL_COMBO + Num_imu_feat_families - 1))) {
            memset(local_feats, 0, sizeof(local_feats));
            imu_run_l2a_features(&imu_features_selector[ACCEL_COMBO], l2a, len, local_feats);
            copy_imu_local(features, local_feats, ACCEL_COMBO);
        }
        if (any_required(imu_features_selector, GYRO_COMBO, (uint16_t)(GYRO_COMBO + Num_imu_feat_families - 1))) {
            memset(local_feats, 0, sizeof(local_feats));
            imu_run_l2g_features(&imu_features_selector[GYRO_COMBO], l2g, len, local_feats);
            copy_imu_local(features, local_feats, GYRO_COMBO);
        }
    }

    free(raw_q5);
    free(axis);
    free(l2a);
    free(l2g);
}

static uint8_t is_cough(float score, float threshold)
{
    return (score >= threshold) ? 1U : 0U;
}

static void init_state_local(void)
{
    fsm_state.model = IMU_MODEL_LOCAL;
    fsm_state.model_cls_out = NON_COUGH_OUT_LOCAL;
    fsm_state.last_output_tick = 0U;
    fsm_state.ticks_from_last_output = 0U;
    fsm_state.window_start_tick = 0U;
    fsm_state.last_window_start_tick = 0U;
    fsm_state.n_winds_aud = 0U;
}

static uint32_t get_idx_window_local(void)
{
    if (fsm_state.model == IMU_MODEL_LOCAL) {
        return (uint32_t)(((uint64_t)fsm_state.window_start_tick * IMU_FS) / AUDIO_FS);
    }
    return fsm_state.window_start_tick;
}

static void update_local(void)
{
    fsm_state.last_window_start_tick = fsm_state.window_start_tick;

    if (fsm_state.model_cls_out == COUGH_OUT_LOCAL) {
        if (fsm_state.model == IMU_MODEL_LOCAL) {
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + IMU_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.model = AUDIO_MODEL_LOCAL;
        } else {
            fsm_state.n_winds_aud++;
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + AUDIO_WINDOW_TICKS - fsm_state.last_output_tick;
            if (fsm_state.n_winds_aud >= N_MAX_WIND_AUD) {
                fsm_state.n_winds_aud = 0U;
                fsm_state.model = IMU_MODEL_LOCAL;
                fsm_state.window_start_tick += AUDIO_WINDOW_TICKS;
            } else {
                fsm_state.model = AUDIO_MODEL_LOCAL;
                fsm_state.window_start_tick += AUDIO_STEP_TICKS;
            }
        }
    } else {
        if (fsm_state.model == IMU_MODEL_LOCAL) {
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + IMU_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.model = IMU_MODEL_LOCAL;
            fsm_state.window_start_tick += IMU_STEP_TICKS;
        } else {
            fsm_state.model = IMU_MODEL_LOCAL;
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + AUDIO_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.window_start_tick += AUDIO_WINDOW_TICKS;
        }
    }
}

static uint8_t check_postprocessing_local(void)
{
    if (fsm_state.ticks_from_last_output >= TIME_DEADLINE_OUTPUT_TICKS) {
        if (fsm_state.model == IMU_MODEL_LOCAL) {
            fsm_state.last_output_tick = fsm_state.last_window_start_tick + IMU_WINDOW_TICKS;
        } else {
            fsm_state.last_output_tick = fsm_state.last_window_start_tick + AUDIO_WINDOW_TICKS;
        }
        fsm_state.ticks_from_last_output = 0U;
        return 1U;
    }
    return 0U;
}

int main(int argc, char **argv)
{
    progressive_block_t block;
    if (argc != 2 || !parse_block(argv[1], &block)) {
        fprintf(stderr, "usage: %s audio_fft|audio_psd|audio_mel|audio_scalar|audio_all|imu_raw|imu_l2|imu_all\n", argv[0]);
        return 2;
    }

    int16_t *indexes_audio_f = (int16_t *)malloc((size_t)N_AUDIO_FEATURES * sizeof(int16_t));
    int8_t *indexes_imu_f = (int8_t *)malloc((size_t)N_IMU_FEATURES * sizeof(int8_t));
    float *audio_feature_array = (float *)calloc((size_t)Number_AUDIO_Features, sizeof(float));
    float *imu_feature_array = (float *)calloc((size_t)Number_IMU_Features, sizeof(float));
    float *features_audio_model = (float *)malloc((size_t)TOT_FEATURES_AUDIO_MODEL_AUDIO * sizeof(float));
    float *features_imu_model = (float *)malloc((size_t)TOT_FEATURES_IMU_MODEL_IMU * sizeof(float));
    uint16_t *starts = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    uint16_t *ends = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    uint16_t *locs = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    postproc_peak_t *peaks = (postproc_peak_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(postproc_peak_t));
    float *audio_confidence = (float *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(float));
    if (!indexes_audio_f || !indexes_imu_f || !audio_feature_array || !imu_feature_array ||
        !features_audio_model || !features_imu_model || !starts || !ends || !locs || !peaks ||
        !audio_confidence) {
        return 1;
    }

    int16_t idx = 0;
    for (int16_t i = 0; i < Number_AUDIO_Features; i++) if (audio_features_selector[i]) indexes_audio_f[idx++] = i;
    idx = 0;
    for (int8_t i = 0; i < Number_IMU_Features; i++) if (imu_features_selector[i]) indexes_imu_f[idx++] = i;

    uint16_t n_peaks = 0;
    uint16_t n_idxs_above_th = 0;
    uint16_t new_added = 0;
    uint32_t idx_start_window = 0;
    int debug_cnt = 0;

    init_state_local();

    while (1) {
        idx_start_window = get_idx_window_local();

        if (fsm_state.model == IMU_MODEL_LOCAL) {
            if (idx_start_window + WINDOW_SAMP_IMU >= IMU_LEN) {
                init_state_local();
                idx_start_window = get_idx_window_local();
            }

            imu_features(imu_features_selector, &imu_in[idx_start_window], WINDOW_SAMP_IMU, imu_feature_array);
            apply_imu_block(block, &imu_in[idx_start_window], WINDOW_SAMP_IMU, imu_feature_array);

            for (int16_t j = 0; j < N_IMU_FEATURES; j++) features_imu_model[j] = imu_feature_array[indexes_imu_f[j]];
            if (imu_bio_feats_selector[0]) features_imu_model[N_IMU_FEATURES] = gender;
            if (imu_bio_feats_selector[1]) features_imu_model[N_IMU_FEATURES + 1] = bmi;

            float imu_score = imu_predict(features_imu_model);
            fsm_state.model_cls_out = is_cough(imu_score, IMU_TH) ? COUGH_OUT_LOCAL : NON_COUGH_OUT_LOCAL;
        } else {
            if (idx_start_window + WINDOW_SAMP_AUDIO >= AUDIO_LEN) break;

            audio_features(audio_features_selector, &audio_in.air[idx_start_window], WINDOW_SAMP_AUDIO, AUDIO_FS, audio_feature_array);
            apply_audio_block(block, &audio_in.air[idx_start_window], WINDOW_SAMP_AUDIO, AUDIO_FS, audio_feature_array);

            for (int16_t j = 0; j < N_AUDIO_FEATURES; j++) features_audio_model[j] = audio_feature_array[indexes_audio_f[j]];
            if (audio_bio_feats_selector[0]) features_audio_model[N_AUDIO_FEATURES] = gender;
            if (audio_bio_feats_selector[1]) features_audio_model[N_AUDIO_FEATURES + 1] = bmi;

            float audio_score = audio_predict(features_audio_model);
            fsm_state.model_cls_out = is_cough(audio_score, AUDIO_TH) ? COUGH_OUT_LOCAL : NON_COUGH_OUT_LOCAL;

            _get_cough_peaks(&audio_in.air[idx_start_window], WINDOW_SAMP_AUDIO, AUDIO_FS,
                             &starts[n_peaks], &ends[n_peaks], &locs[n_peaks], &peaks[n_peaks], &new_added);

            for (uint16_t j = 0; j < new_added; j++) {
                starts[n_peaks + j] += idx_start_window;
                ends[n_peaks + j] += idx_start_window;
                locs[n_peaks + j] += idx_start_window;
                audio_confidence[n_peaks + j] = audio_score;
            }
            n_peaks += new_added;
        }

        update_local();

        if (check_postprocessing_local()) {
            uint16_t n_peaks_final = 0;
            if (n_peaks > 0) {
                uint16_t *idxs_above_th = (uint16_t *)malloc((size_t)n_peaks * sizeof(uint16_t));
                for (uint16_t i = 0; i < n_peaks; i++) {
                    if (is_cough(audio_confidence[i], AUDIO_TH)) idxs_above_th[n_idxs_above_th++] = i;
                }

                uint16_t *final_starts = (uint16_t *)malloc((size_t)n_idxs_above_th * sizeof(uint16_t));
                uint16_t *final_ends = (uint16_t *)malloc((size_t)n_idxs_above_th * sizeof(uint16_t));
                uint16_t *above_locs = (uint16_t *)malloc((size_t)n_idxs_above_th * sizeof(uint16_t));
                postproc_peak_t *above_peaks = (postproc_peak_t *)malloc((size_t)n_idxs_above_th * sizeof(postproc_peak_t));

                for (uint16_t i = 0; i < n_idxs_above_th; i++) {
                    final_starts[i] = starts[idxs_above_th[i]];
                    final_ends[i] = ends[idxs_above_th[i]];
                    above_locs[i] = locs[idxs_above_th[i]];
                    above_peaks[i] = peaks[idxs_above_th[i]];
                }

                n_peaks_final = _clean_cough_segments(final_starts, final_ends, above_locs, above_peaks, n_idxs_above_th, AUDIO_FS);
                for (uint16_t k = 0; k < n_peaks_final; k++) {
                    printf("COUGH_SEG: %u %u\n", final_starts[k], final_ends[k]);
                }

                free(idxs_above_th);
                free(final_starts);
                free(final_ends);
                free(above_locs);
                free(above_peaks);
            }
            printf("N_PEAKS FINAL: %d\n", n_peaks_final);
            n_peaks = 0;
            n_idxs_above_th = 0;
        }

        debug_cnt++;
        if (debug_cnt == 100) break;
    }

    free(indexes_audio_f);
    free(indexes_imu_f);
    free(audio_feature_array);
    free(imu_feature_array);
    free(features_audio_model);
    free(features_imu_model);
    free(starts);
    free(ends);
    free(locs);
    free(peaks);
    free(audio_confidence);
    return 0;
}
