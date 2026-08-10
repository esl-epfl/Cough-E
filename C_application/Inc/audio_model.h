#ifndef _AUDIO_MODEL_
#define _AUDIO_MODEL_

#include <inttypes.h>
#include <types.h>

#include <audio_features.h>
#include <bio_features.h>

#define N_AUDIO_FEATURES 82
#define N_BIO_FEATURES_AUDIO 2

#define TOT_FEATURES_AUDIO_MODEL_AUDIO (N_AUDIO_FEATURES + N_BIO_FEATURES_AUDIO)

/*
    Features selector vectors.
    Each array is a one-hot econding of the features to be extracted.
    1 --> extract the corresponding feature
    0 --> non extract the corresponding feature
*/
static const int8_t audio_features_selector[Number_AUDIO_Features] = {
    0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
static const int8_t audio_bio_feats_selector[Number_bio_features] = {
    1, // Gender
    0  // BMI
};

#ifdef FXP_MODE
static const uint8_t audio_model_feature_signed[83] = {
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U,
    1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
static const uint8_t audio_model_feature_frac[83] = {
    20U, 5U,  15U, 16U, 20U, 16U, 16U, 16U, 9U,  9U,  9U,  9U,  9U, 9U,
    9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U, 9U,
    9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U, 9U,
    9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U, 9U,
    9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U,  9U, 14U,
    14U, 14U, 14U, 14U, 14U, 14U, 14U, 14U, 14U, 14U, 14U, 16U, 16U};
#endif

//////////////////////////////////////////////////////////////////////////////////////
// 					Here is the model parameter's and
// description					//
//////////////////////////////////////////////////////////////////////////////////////

/**
 * Number of trees and max number of NODES and LEAVES among the trees
 */
#define AUD_N_TREES 100
#define AUD_MAX_NODES 54
#define MAX_LEAVES 55

/**
 * Defines to identify a NODE and a LEAF type
 */
#define AUD_NODE_T -1
#define AUD_LEAF_T -2

/**
 * Returns the model confidence for the current window to be COUGH
 *
 * @param *feat 	:	pointer to the features to be processed
 */
#ifndef FXP_MODE
audio_score_t audio_predict(audio_data_t *feats);
#endif

/**
 * Structure to store type and ID of a generic node element
 * (it can be either NODE or LEAF)
 */
typedef struct node {
    int16_t type;
    int16_t id;
} audio_NODE_T;

/**
 * Structure to store the info regarding the left and right child of each node
 */
typedef struct node_children {
    audio_NODE_T child_left;
    audio_NODE_T child_right;
} audio_node_children_t;

// 5400 values * 8 bytes = 43200 bytes ~ 42.2kB
extern const audio_node_children_t DINTL0
    audio_children[AUD_N_TREES][AUD_MAX_NODES];
// 5500 values * 4 bytes = 22000 bytes ~ 21.5kB
extern audio_score_t CARUS01 audio_scores[AUD_N_TREES][MAX_LEAVES];
extern const audio_score_t FLASH1 audio_scores_src[AUD_N_TREES][MAX_LEAVES];
// 5400 values * 4 bytes = 21600 bytes ~ 21.1kB
extern audio_score_t CARUS01 audio_values_comp[AUD_N_TREES][AUD_MAX_NODES];
extern const audio_score_t FLASH1 audio_values_comp_src[AUD_N_TREES][AUD_MAX_NODES];
// 5400 values * 2 bytes = 10800 bytes ~ 10.6kB
extern int16_t CARUS00 audio_feat_comp[AUD_N_TREES][AUD_MAX_NODES];
extern const int16_t FLASH0 audio_feat_comp_src[AUD_N_TREES][AUD_MAX_NODES];

#endif
