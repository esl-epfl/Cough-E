#ifndef _FEATURE_EXTRACTION_H
#define _FEATURE_EXTRACTION_H

#include <stdlib.h>
#include <imu_features.h>

#ifdef FXP_MODE
#include <fxp.h>
// Fixed-point IMU signal features dispatcher.
// sig is a typed FxP buffer (q11_5_t*, uq10_6_t*, or uq5_11_t*) cast to void*.
// Results are written to feats[] as float (model trees stay float).
void imu_signal_features_fxp(const int8_t *features_selector, const void *sig,
                              int16_t len, fxp_sig_type_t sig_type, float *feats);
#endif


/**
    Computes features from the AUDIO signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the signal to be processed    
    @param len                  :   the length of the signal
    @param fs                   :   the sampling frequency
    @param *feats               :   array to be filled with the extracted features
*/
void audio_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats);

/**
    Computes features from the IMU signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the IMU signal to be processed, containing the 3 axial accelerometer and angles
    @param len                  :   the length of the signal
    @param *feats               :   array to be filled with the extracted features
*/
void imu_features(const int8_t *features_selector, const float sig[][Num_IMU_signals], int16_t len, float *feats);

#endif