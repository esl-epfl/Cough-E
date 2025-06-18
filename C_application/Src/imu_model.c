#include <stdio.h>
#include <inttypes.h>
#include <math.h>

#include <imu_model.h>

float _imu_sigmoid(float score){
    if(score < 0.0){
        float z = expf(score);
        return z / (1.0 + z);
    }
    return (1.0 / (1.0 + expf(-score)));
}

float imu_predict(float *feats){

    float score = 0.0;

    int16_t current_node = 0;
    int16_t child_type = 0;

    for(int16_t t=0; t<IMU_N_TREES; t++){

        current_node = 0;
        child_type = 0;

        for(int16_t n=0; n<IMU_MAX_NODES; n++){
            
            if(feats[imu_feat_comp[t][current_node]] < imu_values_comp[t][current_node]){
                child_type = imu_children[t][current_node].child_left.type;
                current_node = imu_children[t][current_node].child_left.id;
            } else {
                child_type = imu_children[t][current_node].child_right.type;
                current_node = imu_children[t][current_node].child_right.id;
            }

            if(child_type == IMU_LEAF_T){
                score += imu_scores[t][current_node];
                break;
            }
        }
    }

    float res = _imu_sigmoid(score);
    return res;
}