#ifndef _IMU_MODEL_H_
#define _IMU_MODEL_H_

#include <inttypes.h>
#include <types.h>

#include <bio_features.h>
#include <imu_features.h>

#define N_IMU_FEATURES 34
#define N_BIO_FEATURES_IMU 2

#define TOT_FEATURES_IMU_MODEL_IMU (N_IMU_FEATURES + N_BIO_FEATURES_IMU)

/*
    Features selector vectors.
    Each array is a one-hot econding of the features to be extracted.
    1 --> extract the corresponding feature
    0 --> non extract the corresponding feature
*/
static const int8_t imu_features_selector[Number_IMU_Features] = {
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
};
static const int8_t imu_bio_feats_selector[Number_bio_features] = {
    1, // Gender
    1  // BMI
};

#ifdef FXP_MODE
static const uint8_t imu_model_feature_signed[35] = {
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
static const uint8_t imu_model_feature_frac[35] = {
    9U, 3U, 9U, 3U, 0U, 0U, 0U, 0U, 0U,  0U, 0U, 0U, 0U, 0U, 0U, 9U, 9U, 22U,
    0U, 0U, 0U, 3U, 0U, 0U, 0U, 9U, 14U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 16U};
#endif

//////////////////////////////////////////////////////////////////////////////////////
// 					Here is the model parameter's and
// description					//
//////////////////////////////////////////////////////////////////////////////////////

/**
 * Number of trees and max number of NODES and LEAVES among the trees
 */
#define IMU_N_TREES 100
#define IMU_MAX_NODES 60
#define IMU_MAX_LEAVES 61

/**
 * Defines to identify a NODE and a LEAF type
 */
#define IMU_NODE_T -1
#define IMU_LEAF_T -2

/**
 * Returns the model confidence for the current window to be COUGH
 */
#ifndef FXP_MODE
imu_score_t imu_predict(imu_data_t *feats);
#endif

/**
 * Structure to store type and ID of a generic node element
 * (it can be either NODE or LEAF)
 */
typedef struct imu_node {
    int16_t type;
    int16_t id;
} imu_NODE_T;

/**
 * Structure to store the info regarding the left and right child of each node
 */
typedef struct imu_node_children {
    imu_NODE_T child_left;
    imu_NODE_T child_right;
} imu_node_children_t;

// 6000 values * 8 bytes = 48000 bytes ~ 46.9kB
extern const imu_node_children_t DINTL0
    imu_children[IMU_N_TREES][IMU_MAX_NODES];
// 6100 values * 4 bytes = 24400 bytes ~ 23.9kB
extern imu_score_t CARUS11 imu_scores[IMU_N_TREES][IMU_MAX_LEAVES];
extern const imu_score_t FLASH1 imu_scores_src[IMU_N_TREES][IMU_MAX_LEAVES];
// 6000 values * 4 bytes = 24000 bytes ~ 23.5kB
extern imu_score_t CARUS11 imu_values_comp[IMU_N_TREES][IMU_MAX_NODES];
extern const imu_score_t FLASH1 imu_values_comp_src[IMU_N_TREES][IMU_MAX_NODES];
// 6000 values * 2 bytes = 12000 bytes ~ 11.8kB
extern int16_t CARUS10 imu_feat_comp[IMU_N_TREES][IMU_MAX_NODES];
extern const int16_t FLASH0 imu_feat_comp_src[IMU_N_TREES][IMU_MAX_NODES];

#endif
