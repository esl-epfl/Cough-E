#pragma once

#include <inttypes.h>

#include <imu_features.h>

#ifndef FXP_MODE
typedef struct {
    float *data;
    int16_t len;
} imu_float_view_t;

void imu_run_float_features(const int8_t *features_selector,
                            imu_float_view_t sig,
                            float *feats);
#endif
