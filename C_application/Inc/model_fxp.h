#pragma once

#ifdef FXP_MODE

#include <inttypes.h>

#include <audio_model.h>
#include <imu_model.h>
#include <core/fxp_core.h>

/*
 * Class thresholds mapped from probability space to score space:
 *   score = log(p / (1-p))
 */
#define FXP_AUDIO_SCORE_TH_Q16 ((fxp_q16_t)-55528)   /* logit(0.30) in Q16 */
#define FXP_IMU_SCORE_TH_Q16 ((fxp_q16_t)-192968)    /* logit(0.05) in Q16 */

fxp_q16_t audio_predict_q16(const fxp_q16_t *feats_q16);
fxp_q16_t imu_predict_q16(const fxp_q16_t *feats_q16);

static inline fxp_q16_t audio_predict(const fxp_q16_t *feats_q16)
{
    return audio_predict_q16(feats_q16);
}

static inline fxp_q16_t imu_predict(const fxp_q16_t *feats_q16)
{
    return imu_predict_q16(feats_q16);
}

static inline uint8_t audio_score_is_cough(fxp_q16_t score_q16)
{
    return (score_q16 >= FXP_AUDIO_SCORE_TH_Q16) ? 1U : 0U;
}

static inline uint8_t imu_score_is_cough(fxp_q16_t score_q16)
{
    return (score_q16 >= FXP_IMU_SCORE_TH_Q16) ? 1U : 0U;
}

static inline uint8_t score_is_cough(fxp_q16_t score_q16, fxp_q16_t threshold_q16)
{
    return (score_q16 >= threshold_q16) ? 1U : 0U;
}

#endif
