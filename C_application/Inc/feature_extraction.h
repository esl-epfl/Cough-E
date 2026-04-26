#ifndef _FEATURE_EXTRACTION_H
#define _FEATURE_EXTRACTION_H

#include <stdlib.h>
#include <imu_features.h>
#include <core/cough_backend.h>

typedef cough_feat_t feat_t;
typedef cough_audio_sample_t audio_sample_t;
typedef cough_imu_sample_t imu_sample_t;

/**
    Computes features from the AUDIO signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the signal to be processed    
    @param len                  :   the length of the signal
    @param fs                   :   the sampling frequency
    @param *feats               :   array to be filled with the extracted features
*/
void audio_features(const int8_t *features_selector,
                    const audio_sample_t *sig,
                    int16_t len,
                    int16_t fs,
                    feat_t *feats);

/**
    Computes features from the IMU signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the IMU signal to be processed, containing the 3 axial accelerometer and angles
    @param len                  :   the length of the signal
    @param *feats               :   array to be filled with the extracted features
*/
void imu_features(const int8_t *features_selector,
                  const imu_sample_t sig[][Num_IMU_signals],
                  int16_t len,
                  feat_t *feats);

#endif
