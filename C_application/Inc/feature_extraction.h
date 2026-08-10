#ifndef _FEATURE_EXTRACTION_H
#define _FEATURE_EXTRACTION_H

#include <imu_features.h>
#include <types.h>

#ifdef FXP_MODE
#include <FxP/core/fxp_core.h>
#endif

/**
    Computes features from the AUDIO signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the signal to be processed
    @param len                  :   the length of the signal
    @param fs                   :   the sampling frequency
    @param *feats               :   array to be filled with the extracted features
*/
void audio_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                    audio_feat_t *feats);

/**
    Computes features from the IMU signal, passed as parameter.

    @param *features_selector   :   one-hot vector for the features to extract (0: DO NOT extract, 1: DO extract)
    @param *sig                 :   the IMU signal to be processed, containing the 3 axial accelerometer and angles
    @param len                  :   the length of the signal
    @param *feats               :   array to be filled with the extracted features
*/
void imu_features(const int8_t *features_selector, const imu_data_t sig[][Num_IMU_signals], int16_t len,
                  imu_feat_t *feats);

//////////////////////////////////////////////////////////////////////////////////
/*                      Local functions declaration                             */
//////////////////////////////////////////////////////////////////////////////////

/**
    Given the features selector vector and two indexes (start and end), it
    returns 1 if feature_selector has at least a 1 in the range specified
    by the indexes, 0 otherwise

    @param *features_selector   :   one-hot vector for which features to extract
    @param start_index          :   starting index from which to check the `features_selector`
    @param end_index            :   end index to check in the `features_selector`
*/
int is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index);

/**
    Computes the required FFT-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param fs                   :   sampling frequency
    @param *feats               :   array of extracted features
*/
void fft_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                        audio_data_t *feats);

/**
    Computes the required periodogram-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void periodogram_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                                audio_data_t *feats);

/**
    Computes the required MFCC features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mfcc_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, audio_data_t *feats);

/**
    Computes the required Mel Spectrogram features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mel_spectrogram_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len,
                              audio_data_t *feats);

/**
    Computes the required mean-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mean_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, audio_data_t *feats);

/**
    Computes the required EEPD features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signals
    @param fs                   :   sampling frequency
    @param *feats               :   array of extracted features
*/
void eepd_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                   audio_data_t *feats);

/**
    This function triggers the feature extraction process for a specific IMU feature family.
    First it checks the the features has to be computed, by means of the features_selector array.
    Then it retrieves the proper data and it calls the feature extraction function.

    The discrimination between different IMU signal here is done through the use of the two
    input parameters "signal_idx" and "sig_feat_idx".

    @param *features_selector   :   one-hot vector for which features to extract
    @param signal               :   signal to process
    @param len                  :   length of the signal
    @param signal_idx           :   index of the specific IMU signal
    @param sig_feat_idx         :   starting index of the features for the IMU signal inside the features_selector
   vector
    @param *feats               :   array of extracted features
*/
void compute_imu_family(const int8_t *features_selector, const imu_data_t signal[][Num_IMU_signals], int16_t len,
                        int8_t signal_idx, int8_t sig_feat_idx, imu_data_t *feats);

void compute_imu_l2(const int8_t *features_selector, const imu_data_t signal[][Num_IMU_signals], int16_t len,
                    int8_t signal_idx, int8_t sig_feat_idx, imu_data_t *feats);

#endif
