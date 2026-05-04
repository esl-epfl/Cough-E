#pragma once

#include <inttypes.h>

#include <imu_features.h>

#ifdef FXP_MODE
#include <core/fxp_core.h>

/* -------------------------------------------------------------------------- */
/*  IMU feature dispatch entry points                                          */
/* -------------------------------------------------------------------------- */

/* Runs selected features for one already-prepared IMU signal family.
 * features_selector, sig, and feats are local to that family:
 * raw axis, accel L2A, or gyro L2G.
 */
void imu_run_raw_features(const int8_t *features_selector,
                          const q11_5_t *sig,
                          int16_t len,
                          fxp_feat_t *feats);
void imu_run_l2a_features(const int8_t *features_selector,
                          const uq10_6_t *sig,
                          int16_t len,
                          fxp_feat_t *feats);
void imu_run_l2g_features(const int8_t *features_selector,
                          const uq5_11_t *sig,
                          int16_t len,
                          fxp_feat_t *feats);

/* Builds per-sample L2 norm signals from raw Q11.5 axes. */
uq10_6_t imu_l2a(q11_5_t ax, q11_5_t ay, q11_5_t az);
uq5_11_t imu_l2g(q11_5_t gx, q11_5_t gy, q11_5_t gz);

#endif
