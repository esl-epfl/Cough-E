#include <stdio.h>
#include <inttypes.h>
#include <math.h>

#include <audio_model.h>

/// @brief Computes the sigmoid value of a given score
/// @param score    :   the score for which to compute the sigmoid
/// @return The resulting value
float _audio_sigmoid(float score){
    if(score < 0.0){
        float z = expf(score);
        return z / (1.0 + z);
    }
    return (1.0 / (1.0 + expf(-score)));
}



float audio_predict(float *feats){

    float score = 0.0;

    int16_t current_node = 0;
    int16_t child_type = 0;

    for(int16_t t=0; t<AUD_N_TREES; t++){

        current_node = 0;
        child_type = 0;

        for(int16_t n=0; n<AUD_MAX_NODES; n++){
            
            if(feats[audio_feat_comp[t][current_node]] < audio_values_comp[t][current_node]){
                child_type = audio_children[t][current_node].child_left.type;
                current_node = audio_children[t][current_node].child_left.id;
            } else {
                child_type = audio_children[t][current_node].child_right.type;
                current_node = audio_children[t][current_node].child_right.id;
            }

            if(child_type == AUD_LEAF_T){
                score += audio_scores[t][current_node];
                break;
            }
        }
    }

    float res = _audio_sigmoid(score);
    
    return res;
}