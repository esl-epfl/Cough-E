#ifndef _IMU_FEATURES_H_
#define _IMU_FEATURES_H_

#include <inttypes.h>

/* Defines all the parameters of IMU features extraction */

#define WIND_LEN_IMU    0.5
#define OVERLAP_IMU     50

#define WINDOW_SAMP_IMU     (int16_t)(WIND_LEN_IMU * IMU_FS)
#define IMU_STEP            (int16_t)(WINDOW_SAMP_IMU * (1.0 - (OVERLAP_IMU / 100.0)))

#define IMU_OVERLAP_SEC     (float)(WIND_LEN_IMU * (OVERLAP_IMU / 100.0))
#define IMU_STEP_SEC        (float)(WIND_LEN_IMU - IMU_OVERLAP_SEC)

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


#endif