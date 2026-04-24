#pragma once

/* Shared scalar carriers selected by the existing FXP_MODE build flag. */

#ifdef FXP_MODE
#include <core/fxp_convert.h>
#include <core/fxp_core.h>

typedef int16_t cough_audio_sample_t; /* Q14 */
typedef q11_5_t cough_imu_sample_t;   /* Q11.5 raw IMU */
typedef fxp_q16_t cough_feat_t;       /* Q16 cross-module feature carrier */

static inline cough_audio_sample_t cough_source_audio_sample(float x)
{
    return FXP_AUDIO_FROM_FLOAT(x);
}

static inline cough_imu_sample_t cough_source_imu_sample(float x)
{
    return FXP_IMU_RAW_FROM_FLOAT(x);
}

static inline cough_feat_t cough_source_feat(float x)
{
    return FXP_FROM_FLOAT(x, FXP_PIPE_FRAC);
}

#else

typedef float cough_audio_sample_t;
typedef float cough_imu_sample_t;
typedef float cough_feat_t;

static inline cough_audio_sample_t cough_source_audio_sample(float x)
{
    return x;
}

static inline cough_imu_sample_t cough_source_imu_sample(float x)
{
    return x;
}

static inline cough_feat_t cough_source_feat(float x)
{
    return x;
}

#endif
