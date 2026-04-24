#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <main.h>

#include <fsm_control.h>
#include <feature_extraction.h>
#include <audio_features.h>
#include <imu_features.h>
#include <postprocessing.h>

#ifdef FXP_MODE
#include <core/cough_backend.h>
#include <core/fxp_convert.h>
#define FEAT_THRESHOLD_AUDIO ((feat_t)FXP_AUDIO_SCORE_TH_Q16)
#define FEAT_THRESHOLD_IMU ((feat_t)FXP_IMU_SCORE_TH_Q16)
#else
#define FEAT_THRESHOLD_AUDIO ((feat_t)AUDIO_TH)
#define FEAT_THRESHOLD_IMU ((feat_t)IMU_TH)
#endif

static inline uint8_t is_cough(feat_t score, feat_t threshold)
{
    return (score >= threshold) ? 1U : 0U;
}

int main(void)
{
    int16_t *indexes_audio_f = (int16_t *)malloc((size_t)N_AUDIO_FEATURES * sizeof(int16_t));
    int8_t *indexes_imu_f = (int8_t *)malloc((size_t)N_IMU_FEATURES * sizeof(int8_t));

    int16_t idx = 0;
    for (int16_t i = 0; i < Number_AUDIO_Features; i++) {
        if (audio_features_selector[i] == 1) {
            indexes_audio_f[idx] = i;
            idx++;
        }
    }
    idx = 0;
    for (int8_t i = 0; i < Number_IMU_Features; i++) {
        if (imu_features_selector[i] == 1) {
            indexes_imu_f[idx] = i;
            idx++;
        }
    }

    feat_t *audio_feature_array = (feat_t *)malloc((size_t)Number_AUDIO_Features * sizeof(feat_t));
    memset(audio_feature_array, 0, (size_t)Number_AUDIO_Features * sizeof(feat_t));

    feat_t *imu_feature_array = (feat_t *)malloc((size_t)Number_IMU_Features * sizeof(feat_t));
    memset(imu_feature_array, 0, (size_t)Number_IMU_Features * sizeof(feat_t));

    feat_t *features_audio_model = (feat_t *)malloc((size_t)TOT_FEATURES_AUDIO_MODEL_AUDIO * sizeof(feat_t));
    feat_t *features_imu_model = (feat_t *)malloc((size_t)TOT_FEATURES_IMU_MODEL_IMU * sizeof(feat_t));

    feat_t audio_score = 0;
    feat_t imu_score = 0;

    uint16_t *starts = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    uint16_t *ends = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    uint16_t *locs = (uint16_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(uint16_t));
    postproc_peak_t *peaks = (postproc_peak_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(postproc_peak_t));

    uint16_t n_peaks = 0;
    uint16_t new_added = 0;

    feat_t *audio_confidence = (feat_t *)malloc((size_t)MAX_PEAKS_EXPECTED * sizeof(feat_t));

    uint32_t idx_start_window = 0;
    uint16_t n_idxs_above_th = 0;
    int debug_cnt = 0;

    feat_t gender_feature = 0;
    feat_t bmi_feature = 0;

#ifdef FXP_MODE
    int16_t *audio_in_q14 = (int16_t *)malloc((size_t)AUDIO_LEN * sizeof(int16_t));
    q11_5_t (*imu_in_q5)[Num_IMU_signals] = (q11_5_t(*)[Num_IMU_signals])malloc((size_t)IMU_LEN * sizeof(*imu_in_q5));
    const audio_sample_t *audio_runtime_in = NULL;
    const imu_sample_t (*imu_runtime_in)[Num_IMU_signals] = NULL;

    if (!audio_in_q14 || !imu_in_q5) {
        free(audio_in_q14);
        free(imu_in_q5);
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
        return 1;
    }

    /* Single runtime boundary: source float samples are converted once to FxP carriers. */
    for (int32_t i = 0; i < AUDIO_LEN; i++) {
        audio_in_q14[i] = cough_source_audio_sample(audio_in.air[i]);
    }
    for (int32_t i = 0; i < IMU_LEN; i++) {
        for (int8_t ax = 0; ax < Num_IMU_signals; ax++) {
            imu_in_q5[i][ax] = cough_source_imu_sample(imu_in[i][ax]);
        }
    }

    audio_runtime_in = audio_in_q14;
    imu_runtime_in = imu_in_q5;
    gender_feature = cough_source_feat(gender);
    bmi_feature = cough_source_feat(bmi);
#else
    const audio_sample_t *audio_runtime_in = audio_in.air;
    const imu_sample_t (*imu_runtime_in)[Num_IMU_signals] = imu_in;
    gender_feature = (feat_t)gender;
    bmi_feature = (feat_t)bmi;
#endif

    init_state();

    while (1) {
        idx_start_window = get_idx_window();

        if (fsm_state.model == IMU_MODEL) {
            if (idx_start_window + WINDOW_SAMP_IMU >= IMU_LEN) {
                init_state();
                idx_start_window = get_idx_window();
            }

            imu_features(imu_features_selector,
                         &imu_runtime_in[idx_start_window],
                         WINDOW_SAMP_IMU,
                         imu_feature_array);

            for (int16_t j = 0; j < N_IMU_FEATURES; j++) {
                features_imu_model[j] = imu_feature_array[indexes_imu_f[j]];
            }
            if (imu_bio_feats_selector[0] == 1) {
                features_imu_model[N_IMU_FEATURES] = gender_feature;
            }
            if (imu_bio_feats_selector[1] == 1) {
                features_imu_model[N_IMU_FEATURES + 1] = bmi_feature;
            }

            imu_score = imu_predict(features_imu_model);
            fsm_state.model_cls_out = is_cough(imu_score, FEAT_THRESHOLD_IMU) ? COUGH_OUT : NON_COUGH_OUT;
        } else {
            if (idx_start_window + WINDOW_SAMP_AUDIO >= AUDIO_LEN) {
                break;
            }

            audio_features(audio_features_selector,
                           &audio_runtime_in[idx_start_window],
                           WINDOW_SAMP_AUDIO,
                           AUDIO_FS,
                           audio_feature_array);

            for (int16_t j = 0; j < N_AUDIO_FEATURES; j++) {
                features_audio_model[j] = audio_feature_array[indexes_audio_f[j]];
            }
            if (audio_bio_feats_selector[0] == 1) {
                features_audio_model[N_AUDIO_FEATURES] = gender_feature;
            }
            if (audio_bio_feats_selector[1] == 1) {
                features_audio_model[N_AUDIO_FEATURES + 1] = bmi_feature;
            }

            audio_score = audio_predict(features_audio_model);
            fsm_state.model_cls_out = is_cough(audio_score, FEAT_THRESHOLD_AUDIO) ? COUGH_OUT : NON_COUGH_OUT;

            _get_cough_peaks(&audio_runtime_in[idx_start_window], WINDOW_SAMP_AUDIO, AUDIO_FS,
                             &starts[n_peaks], &ends[n_peaks], &locs[n_peaks], &peaks[n_peaks], &new_added);

            for (uint16_t j = 0; j < new_added; j++) {
                starts[n_peaks + j] += idx_start_window;
                ends[n_peaks + j] += idx_start_window;
                locs[n_peaks + j] += idx_start_window;
                audio_confidence[n_peaks + j] = audio_score;
            }
            n_peaks += new_added;
        }

        update();

        if (check_postprocessing()) {
            uint16_t n_peaks_final = 0;

            if (n_peaks > 0) {
                uint16_t *idxs_above_th = (uint16_t *)malloc((size_t)n_peaks * sizeof(uint16_t));

                for (uint16_t i = 0; i < n_peaks; i++) {
                    if (is_cough(audio_confidence[i], FEAT_THRESHOLD_AUDIO)) {
                        idxs_above_th[n_idxs_above_th] = i;
                        n_idxs_above_th++;
                    }
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

#ifdef EVALUATION_MODE
                for (uint16_t k = 0; k < n_peaks_final; k++) {
                    printf("COUGH_SEG: %u %u\n", final_starts[k], final_ends[k]);
                }
#endif

                free(idxs_above_th);
                free(final_starts);
                free(final_ends);
                free(above_locs);
                free(above_peaks);
            }

#ifdef EVALUATION_MODE
            printf("N_PEAKS FINAL: %d\n", n_peaks_final);
#endif
            (void)n_peaks_final;

            n_peaks = 0;
            n_idxs_above_th = 0;
        }

        debug_cnt++;
        if (debug_cnt == 100) {
            break;
        }
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

#ifdef FXP_MODE
    free(audio_in_q14);
    free(imu_in_q5);
#endif

    return 0;
}
