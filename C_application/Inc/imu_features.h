#ifndef _IMU_FEATURES_H_
#define _IMU_FEATURES_H_

#include <inttypes.h>

#include <core/fxp_core.h>

/* Defines all the parameters of IMU features extraction */

#define WIND_LEN_IMU_NUM 1U
#define WIND_LEN_IMU_DEN 2U
#ifndef FXP_MODE
#define WIND_LEN_IMU    ((float)WIND_LEN_IMU_NUM / (float)WIND_LEN_IMU_DEN)
#endif
#define OVERLAP_IMU     50

#define WINDOW_SAMP_IMU     ((int16_t)(((uint32_t)IMU_FS * WIND_LEN_IMU_NUM) / WIND_LEN_IMU_DEN))
#define IMU_OVERLAP_SAMP    ((int16_t)(((uint32_t)WINDOW_SAMP_IMU * OVERLAP_IMU) / 100U))
#define IMU_STEP            ((int16_t)(WINDOW_SAMP_IMU - IMU_OVERLAP_SAMP))

#define IMU_WINDOW_TICKS    ((uint32_t)(((uint64_t)WINDOW_SAMP_IMU * AUDIO_FS) / IMU_FS))
#define IMU_STEP_TICKS      ((uint32_t)(((uint64_t)IMU_STEP * AUDIO_FS) / IMU_FS))

#ifndef FXP_MODE
#define IMU_OVERLAP_SEC     (float)(WIND_LEN_IMU * (OVERLAP_IMU / 100.0))
#define IMU_STEP_SEC        (float)(WIND_LEN_IMU - IMU_OVERLAP_SEC)
#endif

/**
 * Epsilon tolerance value for the AZC computation
*/
#define EPSILON_START   0.3 
#define EPSILON_END     1.0
#define EPSILON_STEP    0.1

// #define N_AZC ((EPSILON_END - EPSILON_START) / EPSILON_STEP) + 1
#define N_AZC 8

/*
    Different IMU signals that are used for the feature extraction
*/
enum imu_signals {
    ACCELEROMETER_X,
    ACCELEROMETER_Y,
    ACCELEROMETER_Z,
    GYROSCOPE_Y,
    GYROSCOPE_P,
    GYROSCOPE_R,
    Num_IMU_signals
};


/*
    Featues types that can be extracted from each IMU signal
*/
enum imu_features_families {
    LINE_LENGTH,
    ZERO_CROSSING_RATE_IMU,
    KURTOSIS,
    ROOT_MEANS_SQUARED_IMU,
    CREST_FACTOR_IMU,
    APPROXIMATE_ZERO_CROSSING,
    Num_imu_feat_families = APPROXIMATE_ZERO_CROSSING + N_AZC  // total number of different feature types we have for IMU
};


/*
    Indexes all the possibile IMU features that we can have.
    Basically, for every IMU signal we can have each one of
    the features families.
    This enum is primarly used to have base indexes for the features_selector array
*/
enum imu_signal_features {
    ACCEL_X_FEAT,
    ACCEL_Y_FEAT = ACCEL_X_FEAT + Num_imu_feat_families,
    ACCEL_Z_FEAT = ACCEL_Y_FEAT + Num_imu_feat_families,
    
    GYRO_Y_FEAT = ACCEL_Z_FEAT + Num_imu_feat_families,
    GYRO_P_FEAT = GYRO_Y_FEAT + Num_imu_feat_families,
    GYRO_R_FEAT = GYRO_P_FEAT + Num_imu_feat_families,

    ACCEL_COMBO = GYRO_R_FEAT + Num_imu_feat_families,
    GYRO_COMBO = ACCEL_COMBO + Num_imu_feat_families,

    Number_IMU_Features = GYRO_COMBO + Num_imu_feat_families
};

static inline uint8_t imu_feature_frac_bits(uint16_t feature_idx)
{
    uint16_t base;
    uint16_t local;

    if (feature_idx >= Number_IMU_Features) return FXP_PIPE_FRAC;

    base = (uint16_t)((feature_idx / Num_imu_feat_families) * Num_imu_feat_families);
    local = (uint16_t)(feature_idx - base);

    if (local >= APPROXIMATE_ZERO_CROSSING && local < Num_imu_feat_families) {
        return 0U;
    }

    if (base == GYRO_COMBO) {
        if (local == LINE_LENGTH) return FXP_FRAC_IMU_LINE_LENGTH_L2G;
        if (local == ROOT_MEANS_SQUARED_IMU) return FXP_FRAC_IMU_RMS_L2G;
        if (local == CREST_FACTOR_IMU) return FXP_FRAC_IMU_CREST_L2G;
    } else if (base == ACCEL_COMBO) {
        if (local == ROOT_MEANS_SQUARED_IMU) return FXP_FRAC_IMU_RMS_L2A;
    } else {
        if (local == LINE_LENGTH) return FXP_FRAC_IMU_LINE_LENGTH_RAW;
        if (local == KURTOSIS) return FXP_FRAC_IMU_KURTOSIS_RAW;
        if (local == ROOT_MEANS_SQUARED_IMU) return FXP_FRAC_IMU_RMS_RAW;
    }

    return FXP_PIPE_FRAC;
}

static inline uint8_t imu_feature_is_signed(uint16_t feature_idx)
{
    uint16_t local;

    if (feature_idx >= Number_IMU_Features) return 0U;
    local = (uint16_t)(feature_idx % Num_imu_feat_families);
    return (local == KURTOSIS) ? 1U : 0U;
}


#endif
