#ifndef RANGE_ANALYSIS_H
#define RANGE_ANALYSIS_H

#include <stdio.h>
#include <math.h>
#include <inttypes.h>

#ifdef RANGE_ANALYSIS

/*
 * Output format:
 *   RANGE|<section>|<function>|<variable>|<len>|<min>|<max>|<absmax>
 *
 * Sections:
 *   IMU_RAW        – raw single-axis signals (accel_x/y/z, gyro_y/p/r)
 *   IMU_L2_ACCEL   – L2 norm combo of accelerometer axes
 *   IMU_L2_GYRO    – L2 norm combo of gyroscope axes
 *   AUDIO_PREPROC, AUDIO_FFT, AUDIO_PSD, AUDIO_MEL,
 *   AUDIO_TIME, AUDIO_EEPD, CLASSIFY, POSTPROC
 */

/* Global context for IMU signal path tagging.
 * Set before calling imu_signal_features() to differentiate
 * raw vs. L2-accel vs. L2-gyro paths through the same kernels. */
extern const char *_ra_imu_signal_ctx;
extern int _ra_imu_active;

#define RA_SET_IMU_CTX(ctx) do { _ra_imu_signal_ctx = (ctx); _ra_imu_active = 1; } while(0)
#define RA_CLEAR_IMU_CTX()  do { _ra_imu_active = 0; } while(0)

static inline void _ra_log_array(const char *section, const char *func, const char *var,
                                  const float *arr, int16_t len) {
    float mn = arr[0], mx = arr[0], absmx = fabsf(arr[0]);
    for (int16_t i = 1; i < len; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        float a = fabsf(arr[i]);
        if (a > absmx) absmx = a;
    }
    printf("RANGE|%s|%s|%s|%d|%e|%e|%e\n", section, func, var, len, mn, mx, absmx);
}

static inline void _ra_log_scalar(const char *section, const char *func, const char *var,
                                   float val) {
    printf("RANGE|%s|%s|%s|1|%e|%e|%e\n", section, func, var, val, val, fabsf(val));
}

#define RA_LOG_ARRAY(section, func, var, arr, len)  _ra_log_array(section, func, var, arr, len)
#define RA_LOG_SCALAR(section, func, var, val)      _ra_log_scalar(section, func, var, val)
/* IMU-specific versions that use the global context tag (only fire when IMU context is active) */
#define RA_IMU_LOG_ARRAY(func, var, arr, len)  do { if(_ra_imu_active) _ra_log_array(_ra_imu_signal_ctx, func, var, arr, len); } while(0)
#define RA_IMU_LOG_SCALAR(func, var, val)      do { if(_ra_imu_active) _ra_log_scalar(_ra_imu_signal_ctx, func, var, val); } while(0)

#else

#define RA_SET_IMU_CTX(ctx)
#define RA_CLEAR_IMU_CTX()
#define RA_LOG_ARRAY(section, func, var, arr, len)
#define RA_LOG_SCALAR(section, func, var, val)
#define RA_IMU_LOG_ARRAY(func, var, arr, len)
#define RA_IMU_LOG_SCALAR(func, var, val)

#endif /* RANGE_ANALYSIS */

#endif /* RANGE_ANALYSIS_H */
