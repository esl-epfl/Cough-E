#include <stdio.h>
#include <inttypes.h>

#include <imu_model.h>

#ifdef FXP_MODE
#include <model_fxp.h>
#include <core/fxp_core.h>

fxp_q16_t imu_predict_q16(const fxp_q16_t *feats_q16)
{
    if (!feats_q16) return 0;

    int16_t current_node = 0;
    int16_t child_type = 0;
    int64_t score_q16 = 0;

    for (int16_t t = 0; t < IMU_N_TREES; t++) {
        current_node = 0;
        child_type = 0;

        for (int16_t n = 0; n < IMU_MAX_NODES; n++) {
            int16_t feat_idx = imu_feat_comp[t][current_node];
            if (feats_q16[feat_idx] < imu_values_comp_q16[t][current_node]) {
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
