#include <stdio.h>
#include <inttypes.h>

#include <imu_model.h>

#ifdef FXP_MODE
#include <model_fxp.h>
#include <core/fxp_core.h>

static uint8_t _imu_model_feature_less(fxp_feat_t value, fxp_feat_t threshold, int16_t model_idx)
{
    uint8_t feature_frac = 0U;
    uint8_t threshold_frac = 0U;
    uint8_t signed_feature = 0U;

    if (model_idx < 0 || model_idx >= TOT_FEATURES_IMU_MODEL_IMU) {
        return (value < threshold) ? 1U : 0U;
    }

    feature_frac = imu_model_feature_frac[model_idx];
    threshold_frac = FXP_PIPE_FRAC;
    signed_feature = imu_model_feature_signed[model_idx];

    if (signed_feature) {
        int64_t value_q16 = (int64_t)(int32_t)value;
        int64_t threshold_q16 = (int64_t)(int32_t)threshold;

        if (feature_frac < FXP_PIPE_FRAC) {
            value_q16 <<= (uint8_t)(FXP_PIPE_FRAC - feature_frac);
        } else if (feature_frac > FXP_PIPE_FRAC) {
            value_q16 = fxp_round_div_i64(value_q16, (int32_t)(1U << (feature_frac - FXP_PIPE_FRAC)));
        }

        if (threshold_frac > FXP_PIPE_FRAC) {
            threshold_q16 = fxp_round_div_i64(threshold_q16, (int32_t)(1U << (threshold_frac - FXP_PIPE_FRAC)));
        }

        return (value_q16 < threshold_q16) ? 1U : 0U;
    }

    uint64_t value_q16 = (uint64_t)value;
    uint64_t threshold_q16 = (uint64_t)threshold;

    if (feature_frac < FXP_PIPE_FRAC) {
        value_q16 <<= (uint8_t)(FXP_PIPE_FRAC - feature_frac);
    } else if (feature_frac > FXP_PIPE_FRAC) {
        value_q16 = fxp_round_shift_u64(value_q16, (uint32_t)(feature_frac - FXP_PIPE_FRAC));
    }

    if (threshold_frac > FXP_PIPE_FRAC) {
        threshold_q16 = fxp_round_shift_u64(threshold_q16, (uint32_t)(threshold_frac - FXP_PIPE_FRAC));
    }

    return (value_q16 < threshold_q16) ? 1U : 0U;
}

fxp_q16_t imu_predict_q16(const fxp_feat_t *feats)
{
    if (!feats) return 0;

    int16_t current_node = 0;
    int16_t child_type = 0;
    int64_t score_q16 = 0;

    for (int16_t t = 0; t < IMU_N_TREES; t++) {
        current_node = 0;
        child_type = 0;

        for (int16_t n = 0; n < IMU_MAX_NODES; n++) {
            int16_t feat_idx = imu_feat_comp[t][current_node];
            if (_imu_model_feature_less(feats[feat_idx], imu_values_comp_fxp[t][current_node], feat_idx)) {
                child_type = imu_children[t][current_node].child_left.type;
                current_node = imu_children[t][current_node].child_left.id;
            } else {
                child_type = imu_children[t][current_node].child_right.type;
                current_node = imu_children[t][current_node].child_right.id;
            }

            if (child_type == IMU_LEAF_T) {
                score_q16 += (int64_t)imu_scores_q16[t][current_node];
                break;
            }
        }
    }

    return fxp_sat_s32_from_s64(score_q16);
}

#else

#include <math.h>
#include <range_analysis.h>

float _imu_sigmoid(float score)
{
    if (score < 0.0f) {
        float z = expf(score);
        return z / (1.0f + z);
    }
    return (1.0f / (1.0f + expf(-score)));
}

float imu_predict(float *feats)
{
    RA_LOG_ARRAY("CLASSIFY", "imu_predict", "feats_input", feats, TOT_FEATURES_IMU_MODEL_IMU);

    float score = 0.0f;
    int16_t current_node = 0;
    int16_t child_type = 0;

    for (int16_t t = 0; t < IMU_N_TREES; t++) {
        current_node = 0;
        child_type = 0;

        for (int16_t n = 0; n < IMU_MAX_NODES; n++) {
            if (feats[imu_feat_comp[t][current_node]] < imu_values_comp[t][current_node]) {
                child_type = imu_children[t][current_node].child_left.type;
                current_node = imu_children[t][current_node].child_left.id;
            } else {
                child_type = imu_children[t][current_node].child_right.type;
                current_node = imu_children[t][current_node].child_right.id;
            }

            if (child_type == IMU_LEAF_T) {
                score += imu_scores[t][current_node];
                break;
            }
        }
    }

    RA_LOG_SCALAR("CLASSIFY", "imu_predict", "score", score);

    float res = _imu_sigmoid(score);
    RA_LOG_SCALAR("CLASSIFY", "_imu_sigmoid", "result", res);
    return res;
}

#endif
