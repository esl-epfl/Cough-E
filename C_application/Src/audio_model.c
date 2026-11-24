#include <stdio.h>
#include <inttypes.h>

#include <audio_model.h>

#ifdef FXP_MODE
#include <model_fxp.h>
#include <core/fxp_core.h>

// Thresholds in audio_values_comp are pre-quantized to each feature's native
// Q-format by generate_fxp_tables.py, so the runtime only has to pick the
// right signed/unsigned compare based on the feature's signedness.
static inline uint8_t _feat_less(fxp_feat_t value, uint32_t threshold, uint8_t signed_feature)
{
    return signed_feature ? ((int32_t)value < (int32_t)threshold) : (value < threshold);
}

fxp_q16_t audio_predict(const fxp_feat_t *feats)
{
    if (!feats) return 0;

    int16_t current_node = 0;
    int16_t child_type = 0;
    int64_t score = 0;

    for (int16_t t = 0; t < AUD_N_TREES; t++) {
        current_node = 0;
        child_type = 0;

        for (int16_t n = 0; n < AUD_MAX_NODES; n++) {
            int16_t feat_idx = audio_feat_comp[t][current_node];
            if (_feat_less(feats[feat_idx], audio_values_comp[t][current_node], audio_model_feature_signed[feat_idx])) {
                child_type = audio_children[t][current_node].child_left.type;
                current_node = audio_children[t][current_node].child_left.id;
            } else {
                child_type = audio_children[t][current_node].child_right.type;
                current_node = audio_children[t][current_node].child_right.id;
            }

            if (child_type == AUD_LEAF_T) {
                score += (int64_t)audio_scores[t][current_node];
                break;
            }
        }
    }

    return (fxp_q16_t)score;
}

#else

#include <math.h>
#include <range_analysis.h>

/// @brief Computes the sigmoid value of a given score
/// @param score    :   the score for which to compute the sigmoid
/// @return The resulting value
float _audio_sigmoid(float score)
{
    if (score < 0.0f) {
        float z = expf(score);
        return z / (1.0f + z);
    }
    return (1.0f / (1.0f + expf(-score)));
}

float audio_predict(float *feats)
{
    RA_LOG_ARRAY("CLASSIFY", "audio_predict", "feats_input", feats, TOT_FEATURES_AUDIO_MODEL_AUDIO);

    float score = 0.0f;
    int16_t current_node = 0;
    int16_t child_type = 0;

    for (int16_t t = 0; t < AUD_N_TREES; t++) {
        current_node = 0;
        child_type = 0;

        for (int16_t n = 0; n < AUD_MAX_NODES; n++) {
            if (feats[audio_feat_comp[t][current_node]] < audio_values_comp[t][current_node]) {
                child_type = audio_children[t][current_node].child_left.type;
                current_node = audio_children[t][current_node].child_left.id;
            } else {
                child_type = audio_children[t][current_node].child_right.type;
                current_node = audio_children[t][current_node].child_right.id;
            }

            if (child_type == AUD_LEAF_T) {
                score += audio_scores[t][current_node];
                break;
            }
        }
    }

    RA_LOG_SCALAR("CLASSIFY", "audio_predict", "score", score);

    float res = _audio_sigmoid(score);
    RA_LOG_SCALAR("CLASSIFY", "_audio_sigmoid", "result", res);
    return res;
}

#endif
