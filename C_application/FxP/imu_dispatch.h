#pragma once

// =============================================================================
// imu_dispatch.h — Type-generic IMU signal dispatch via C11 _Generic
//
// Float mode:  imu_sig_float_t  ->  original float kernels
// FxP mode:    imu_sig_raw_t    ->  fxp_get_rms_raw, fxp_get_line_length_raw, …
//              imu_sig_l2a_t    ->  fxp_get_rms_l2a, …
//              imu_sig_l2g_t    ->  fxp_get_rms_l2g, fxp_get_line_length_l2g, …
//
// Requires C11 (-std=c11) for _Generic.
// =============================================================================

#include <inttypes.h>
#include <imu_features.h>

#ifdef FXP_MODE
#include <fxp.h>
#endif

#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>

// =============================================================================
// Section 1 — Tagged signal structs
//
// Each struct is a (typed pointer, length) pair.
// =============================================================================

typedef struct { float    *data; int16_t len; } imu_sig_float_t;

#ifdef FXP_MODE
typedef struct { q11_5_t  *data; int16_t len; } imu_sig_raw_t;   // single axis, Q11.5
typedef struct { uq10_6_t *data; int16_t len; } imu_sig_l2a_t;   // COMBO_ACCEL, UQ10.6
typedef struct { uq5_11_t *data; int16_t len; } imu_sig_l2g_t;   // COMBO_GYRO,  UQ5.11
#endif

// =============================================================================
// Section 2 — Thin wrapper functions  (static inline)
//
// _Generic requires each arm to name a function (not a macro).  These wrappers
// accept a tagged struct, call the underlying kernel, and return float. The
// feature vector is always float regardless of build mode.
//
// Naming: imu_dispatch_<kernel>_<suffix>(sig_struct) -> float
// =============================================================================

// ── Float wrappers ──────────────────────────────────────────────────────────

static inline float imu_dispatch_line_length_float(imu_sig_float_t s) {
    return get_line_length(s.data, s.len);
}
static inline float imu_dispatch_zcr_float(imu_sig_float_t s) {
    return compute_zrc(s.data, s.len);
}
static inline float imu_dispatch_kurtosis_float(imu_sig_float_t s) {
    return get_kurtosis(s.data, s.len);
}
static inline float imu_dispatch_rms_float(imu_sig_float_t s) {
    return get_rms(s.data, s.len);
}
static inline float imu_dispatch_max_float(imu_sig_float_t s) {
    return get_max(s.data, s.len);
}
static inline float imu_dispatch_azc_float(imu_sig_float_t s, float eps) {
    return (float)azc_computation(s.data, s.len, eps);
}

// ── FxP wrappers ────────────────────────────────────────────────────────────
#ifdef FXP_MODE

// --- RAW (Q11.5) ---
static inline float imu_dispatch_line_length_raw(imu_sig_raw_t s) {
    return FXP_TO_FLOAT(fxp_get_line_length_raw(s.data, s.len), 23);  // uq9_23_t: 23 frac bits
}
static inline float imu_dispatch_zcr_raw(imu_sig_raw_t s) {
    return fxp_compute_zcr_raw(s.data, s.len);
}
static inline float imu_dispatch_kurtosis_raw(imu_sig_raw_t s) {
    return FXP_TO_FLOAT(fxp_get_kurtosis_raw(s.data, s.len), 30);
}
static inline float imu_dispatch_rms_raw(imu_sig_raw_t s) {
    return FXP_TO_FLOAT(fxp_get_rms_raw(s.data, s.len), 16);
}
// RAW max and crest factor: not present in the IMU model — return 0 intentionally.
static inline float imu_dispatch_max_raw(imu_sig_raw_t s) {
    (void)s; return 0.0f;
}
static inline float imu_dispatch_azc_raw(imu_sig_raw_t s, float eps) {
    return (float)fxp_azc_computation_raw(s.data, s.len, eps);
}

// --- L2_A (UQ10.6) ---
// line_length, zcr, kurtosis: not present in the IMU model for COMBO_ACCEL.
static inline float imu_dispatch_line_length_l2a(imu_sig_l2a_t s) {
    (void)s; return 0.0f;
}
static inline float imu_dispatch_zcr_l2a(imu_sig_l2a_t s) {
    (void)s; return 0.0f;  // unsigned signal: ZCR = 0 by definition
}
static inline float imu_dispatch_kurtosis_l2a(imu_sig_l2a_t s) {
    (void)s; return 0.0f;
}
static inline float imu_dispatch_rms_l2a(imu_sig_l2a_t s) {
    return FXP_TO_FLOAT(fxp_get_rms_l2a(s.data, s.len), 3);
}
static inline float imu_dispatch_max_l2a(imu_sig_l2a_t s) {
    (void)s; return 0.0f;  // not used for COMBO_ACCEL
}
static inline float imu_dispatch_azc_l2a(imu_sig_l2a_t s, float eps) {
    return (float)fxp_azc_computation_l2a(s.data, s.len, eps);
}

// --- L2_G (UQ5.11) ---
static inline float imu_dispatch_line_length_l2g(imu_sig_l2g_t s) {
    return FXP_TO_FLOAT(fxp_get_line_length_l2g(s.data, s.len), 9);
}
static inline float imu_dispatch_zcr_l2g(imu_sig_l2g_t s) {
    (void)s; return 0.0f;  // unsigned signal: ZCR = 0 by definition
}
static inline float imu_dispatch_kurtosis_l2g(imu_sig_l2g_t s) {
    (void)s; return 0.0f;  // kurtosis not used for COMBO_GYRO
}
static inline float imu_dispatch_rms_l2g(imu_sig_l2g_t s) {
    return FXP_TO_FLOAT(fxp_get_rms_l2g(s.data, s.len), 9);
}
static inline float imu_dispatch_max_l2g(imu_sig_l2g_t s) {
    return FXP_TO_FLOAT(fxp_get_max_l2g(s.data, s.len), 11);
}
static inline float imu_dispatch_azc_l2g(imu_sig_l2g_t s, float eps) {
    return (float)fxp_azc_computation_l2g(s.data, s.len, eps);
}

#endif // FXP_MODE

// =============================================================================
// Section 3 — _Generic dispatch macros
//
// Each macro selects the correct Section 2 wrapper based on the struct type of
// `sig`.
// Usage:  float val = IMU_GET_RMS(my_signal_struct);
// =============================================================================

#ifdef FXP_MODE

#define IMU_LINE_LENGTH(sig) _Generic((sig),        \
    imu_sig_float_t: imu_dispatch_line_length_float, \
    imu_sig_raw_t:   imu_dispatch_line_length_raw,   \
    imu_sig_l2a_t:   imu_dispatch_line_length_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_line_length_l2g    \
)(sig)

#define IMU_ZCR(sig) _Generic((sig),        \
    imu_sig_float_t: imu_dispatch_zcr_float, \
    imu_sig_raw_t:   imu_dispatch_zcr_raw,   \
    imu_sig_l2a_t:   imu_dispatch_zcr_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_zcr_l2g    \
)(sig)

#define IMU_KURTOSIS(sig) _Generic((sig),        \
    imu_sig_float_t: imu_dispatch_kurtosis_float, \
    imu_sig_raw_t:   imu_dispatch_kurtosis_raw,   \
    imu_sig_l2a_t:   imu_dispatch_kurtosis_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_kurtosis_l2g    \
)(sig)

#define IMU_GET_RMS(sig) _Generic((sig),     \
    imu_sig_float_t: imu_dispatch_rms_float, \
    imu_sig_raw_t:   imu_dispatch_rms_raw,   \
    imu_sig_l2a_t:   imu_dispatch_rms_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_rms_l2g    \
)(sig)

#define IMU_GET_MAX(sig) _Generic((sig),     \
    imu_sig_float_t: imu_dispatch_max_float, \
    imu_sig_raw_t:   imu_dispatch_max_raw,   \
    imu_sig_l2a_t:   imu_dispatch_max_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_max_l2g    \
)(sig)

#define IMU_AZC(sig, eps) _Generic((sig),    \
    imu_sig_float_t: imu_dispatch_azc_float, \
    imu_sig_raw_t:   imu_dispatch_azc_raw,   \
    imu_sig_l2a_t:   imu_dispatch_azc_l2a,   \
    imu_sig_l2g_t:   imu_dispatch_azc_l2g    \
)(sig, eps)

#else // !FXP_MODE — float-only, single arm

#define IMU_LINE_LENGTH(sig) imu_dispatch_line_length_float(sig)
#define IMU_ZCR(sig)         imu_dispatch_zcr_float(sig)
#define IMU_KURTOSIS(sig)    imu_dispatch_kurtosis_float(sig)
#define IMU_GET_RMS(sig)     imu_dispatch_rms_float(sig)
#define IMU_GET_MAX(sig)     imu_dispatch_max_float(sig)
#define IMU_AZC(sig, eps)    imu_dispatch_azc_float(sig, eps)

#endif // FXP_MODE

// =============================================================================
// Section 4 — IMU_SIGNAL_FEATURES: unified feature extraction entry point
// =============================================================================

#define IMU_SIGNAL_FEATURES(features_selector, sig, feats) do {                \
    if ((features_selector)[LINE_LENGTH]) {                                    \
        (feats)[LINE_LENGTH] = IMU_LINE_LENGTH(sig);                           \
    }                                                                          \
    if ((features_selector)[ZERO_CROSSING_RATE_IMU]) {                         \
        (feats)[ZERO_CROSSING_RATE_IMU] = IMU_ZCR(sig);                        \
    }                                                                          \
    if ((features_selector)[KURTOSIS]) {                                       \
        (feats)[KURTOSIS] = IMU_KURTOSIS(sig);                                 \
    }                                                                          \
    if ((features_selector)[ROOT_MEANS_SQUARED_IMU] ||                         \
        (features_selector)[CREST_FACTOR_IMU]) {                               \
        float _rms = IMU_GET_RMS(sig);                                         \
        if ((features_selector)[ROOT_MEANS_SQUARED_IMU])                       \
            (feats)[ROOT_MEANS_SQUARED_IMU] = _rms;                            \
        if ((features_selector)[CREST_FACTOR_IMU]) {                           \
            float _peak = IMU_GET_MAX(sig);                                    \
            (feats)[CREST_FACTOR_IMU] = (_rms > 0.0f) ? _peak / _rms : 0.0f;  \
        }                                                                      \
    }                                                                          \
    if (is_required((features_selector), APPROXIMATE_ZERO_CROSSING,            \
                    APPROXIMATE_ZERO_CROSSING + N_AZC - 1)) {                  \
        for (uint8_t _azc_i = 0; _azc_i < N_AZC; _azc_i++) {                  \
            if ((features_selector)[APPROXIMATE_ZERO_CROSSING + _azc_i] == 1) {\
                float _eps = EPSILON_START + (EPSILON_STEP * _azc_i);          \
                (feats)[APPROXIMATE_ZERO_CROSSING + _azc_i] =                  \
                    IMU_AZC(sig, _eps);                                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
} while(0)
