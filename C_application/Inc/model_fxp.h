#pragma once

#ifdef FXP_MODE

#include <inttypes.h>

#include <audio_model.h>
#include <FxP/core/fxp_core.h>
#include <imu_model.h>

#define FXP_AUDIO_SCORE_TH_Q16 ((fxp_q16_t) - 55528) /* logit(0.30) in Q16 */
#define FXP_IMU_SCORE_TH_Q16 ((fxp_q16_t) - 192968)  /* logit(0.05) in Q16 */

fxp_q16_t audio_predict(const fxp_feat_t *feats);
fxp_q16_t imu_predict(const fxp_feat_t *feats);

static inline uint8_t audio_score_is_cough(fxp_q16_t score) {
    return (score >= FXP_AUDIO_SCORE_TH_Q16) ? 1U : 0U;
}

static inline uint8_t imu_score_is_cough(fxp_q16_t score) {
    return (score >= FXP_IMU_SCORE_TH_Q16) ? 1U : 0U;
}

static inline uint8_t score_is_cough(fxp_q16_t score, fxp_q16_t threshold) {
    return (score >= threshold) ? 1U : 0U;
}

#endif
