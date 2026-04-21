#include <stdlib.h>
#include <string.h>

#include <feature_extraction.h>
#include <audio_features.h>
#include <imu_features.h>

#include <audio/audio_pipeline_fxp.h>
#include <core/fxp_convert.h>
#include <core/fxp_core.h>
#include <imu/imu_pipeline.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

typedef struct {
    int32_t b[3];
    int32_t a[3];
    int32_t zi[2];
} fxp_iir2_q30_t;

#define FXP_EEPD_PADLEN 9

/*
 * EEPD band [50, 100] and shared envelope low-pass stage in Q30.
 * This is the only EEPD bin currently selected by the feature selector.
 */
static const fxp_iir2_q30_t k_eepd_band0_q30 = {
    .b = {10439284, 0, -10439284},
    .a = {(1 << 30), -2125785234, 1052863257},
    .zi = {-10439284, -10439284},
};

static const fxp_iir2_q30_t k_eepd_env_q30 = {
    .b = {4128, 8256, 4128},
    .a = {(1 << 30), -2141520527, 1067795215},
    .zi = {1073737696, -1067791087},
};

static int _is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index)
{
    for (uint16_t i = start_index; i <= end_index; i++) {
        if (features_selector[i] == 1) return 1;
    }
    return 0;
}

static inline int32_t _mul_q30(int32_t a_q30, int32_t b_q30)
{
    int64_t p = (int64_t)a_q30 * (int64_t)b_q30;
    if (p >= 0) {
        return fxp_sat_s32_from_s64((p + (1LL << 29)) >> 30);
    }
    return fxp_sat_s32_from_s64(-(((-p) + (1LL << 29)) >> 30));
}

static void _linear_filter_q30(const int32_t *sig_q30, int len,
                               const fxp_iir2_q30_t *filt, const int32_t *zi_q30,
                               int32_t *res_q30)
{
    int32_t sig_1 = 0;
    int32_t y_1 = 0;

    int32_t s_1 = zi_q30[0];
    int32_t s_2 = zi_q30[1];

    res_q30[0] = fxp_sat_s32_from_s64((int64_t)_mul_q30(filt->b[0], sig_q30[0]) + (int64_t)s_1);
    y_1 = res_q30[0];
    sig_1 = sig_q30[0];

    for (int i = 1; i < len; i++) {
        int64_t next_s1 = (int64_t)_mul_q30(filt->b[1], sig_1)
                        - (int64_t)_mul_q30(filt->a[1], y_1)
                        + (int64_t)s_2;
        s_1 = fxp_sat_s32_from_s64(next_s1);

        res_q30[i] = fxp_sat_s32_from_s64((int64_t)_mul_q30(filt->b[0], sig_q30[i]) + (int64_t)s_1);

        int64_t next_s2 = (int64_t)_mul_q30(filt->b[2], sig_1)
                        - (int64_t)_mul_q30(filt->a[2], y_1);
        s_2 = fxp_sat_s32_from_s64(next_s2);

        sig_1 = sig_q30[i];
        y_1 = res_q30[i];
    }
}

static void _filtfilt_q30(const int32_t *sig_q30, int len, const fxp_iir2_q30_t *filt, int32_t *res_q30)
{
    int padded_len = (2 * FXP_EEPD_PADLEN) + len;
    int32_t *pad = (int32_t *)malloc((size_t)padded_len * sizeof(int32_t));
    int32_t *intermediate = (int32_t *)malloc((size_t)padded_len * sizeof(int32_t));
    int32_t *reverse = (int32_t *)malloc((size_t)padded_len * sizeof(int32_t));
    int32_t *res_padded = (int32_t *)malloc((size_t)padded_len * sizeof(int32_t));
    int32_t initial[2];

    if (!pad || !intermediate || !reverse || !res_padded) {
        free(pad);
        free(intermediate);
        free(reverse);
        free(res_padded);
        memset(res_q30, 0, (size_t)len * sizeof(int32_t));
        return;
    }

    int32_t left_end = sig_q30[0];
    int32_t right_end = sig_q30[len - 1];

    for (int i = 0; i < FXP_EEPD_PADLEN; i++) {
        int32_t left_ext = sig_q30[FXP_EEPD_PADLEN - i];
        int32_t right_ext = sig_q30[len - 2 - i];
        pad[i] = fxp_sat_s32_from_s64((2LL * (int64_t)left_end) - (int64_t)left_ext);
        pad[FXP_EEPD_PADLEN + len + i] = fxp_sat_s32_from_s64((2LL * (int64_t)right_end) - (int64_t)right_ext);
    }
    memcpy(&pad[FXP_EEPD_PADLEN], sig_q30, (size_t)len * sizeof(int32_t));

    int32_t x0 = pad[0];
    initial[0] = _mul_q30(filt->zi[0], x0);
    initial[1] = _mul_q30(filt->zi[1], x0);
    _linear_filter_q30(pad, padded_len, filt, initial, intermediate);

    for (int i = 0; i < padded_len; i++) {
        reverse[i] = intermediate[padded_len - 1 - i];
    }

    initial[0] = _mul_q30(filt->zi[0], reverse[0]);
    initial[1] = _mul_q30(filt->zi[1], reverse[0]);
    _linear_filter_q30(reverse, padded_len, filt, initial, res_padded);

    for (int i = 0; i < len; i++) {
        res_q30[i] = res_padded[FXP_EEPD_PADLEN + len - 1 - i];
    }

    free(pad);
    free(intermediate);
    free(reverse);
    free(res_padded);
}

static int16_t _find_peaks_q30(const int32_t *x, int16_t len)
{
    int16_t i = 1;
    int16_t i_max = (int16_t)(len - 1);
    int16_t npeaks = 0;

    while (i < i_max) {
        if (x[i - 1] < x[i]) {
            int16_t i_ahead = (int16_t)(i + 1);
            while (i_ahead < i_max && x[i_ahead] == x[i]) {
                i_ahead++;
            }
            if (x[i_ahead] < x[i]) {
                npeaks++;
                i = i_ahead;
            }
        }
        i++;
    }
    return npeaks;
}

static fxp_q16_t _compute_eepd0_q16(const int16_t *sig_q14, int16_t len)
{
    int32_t *sig_q30 = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    int32_t *band_q30 = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    int32_t *sq_q30 = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    int32_t *env_q30 = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    int32_t *norm_q30 = (int32_t *)malloc((size_t)len * sizeof(int32_t));

    if (!sig_q30 || !band_q30 || !sq_q30 || !env_q30 || !norm_q30) {
        free(sig_q30);
        free(band_q30);
        free(sq_q30);
        free(env_q30);
        free(norm_q30);
        return 0;
    }

    for (int16_t i = 0; i < len; i++) {
        sig_q30[i] = ((int32_t)sig_q14[i]) << 16;
    }

    _filtfilt_q30(sig_q30, len, &k_eepd_band0_q30, band_q30);

    for (int16_t i = 0; i < len; i++) {
        sq_q30[i] = _mul_q30(band_q30[i], band_q30[i]);
    }

    _filtfilt_q30(sq_q30, len, &k_eepd_env_q30, env_q30);

    int32_t max_v = env_q30[0];
    for (int16_t i = 1; i < len; i++) {
        if (env_q30[i] > max_v) max_v = env_q30[i];
    }
    if (max_v <= 0) max_v = 1;

    for (int16_t i = 0; i < len; i++) {
        int64_t num = ((int64_t)env_q30[i]) << 30;
        int64_t q = (num >= 0)
            ? ((num + ((int64_t)max_v >> 1)) / (int64_t)max_v)
            : -(((-num) + ((int64_t)max_v >> 1)) / (int64_t)max_v);
        norm_q30[i] = fxp_sat_s32_from_s64(q);
    }

    int16_t n_peaks = _find_peaks_q30(norm_q30, len);

    free(sig_q30);
    free(band_q30);
    free(sq_q30);
    free(env_q30);
    free(norm_q30);

    return fxp_q16_from_int((int32_t)n_peaks);
}

void audio_features_fxp_q16_from_q14(const int8_t *features_selector,
                                     const int16_t *sig_q14,
                                     int16_t len,
                                     int16_t fs,
                                     fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig_q14 || !feats_q16 || len <= 0 || fs <= 0) return;

    fxp_audio_fft_features_from_q14(features_selector, sig_q14, len, fs, feats_q16);
    fxp_audio_periodogram_features_from_q14(features_selector, sig_q14, len, fs, feats_q16);
    fxp_audio_mel_features_from_q14(features_selector, sig_q14, len, feats_q16);

    if (features_selector[ENERGY_ENVELOPE_PEAK_DETECT]) {
        feats_q16[ENERGY_ENVELOPE_PEAK_DETECT] = _compute_eepd0_q16(sig_q14, len);
    }

    /*
     * Float-only kernels intentionally left untouched:
     *   - SPECTRAL_DECREASE / SPECTRAL_SLOPE / SPECTRAL_SKEW
     *   - SPECTRAL_STD / SPECTRAL_ENTROPY
     *   - EEPD bins 1..(N_EEPD-1)
     * They are not selected by the current feature selector path, so they are
     * not invoked in this FxP runtime flow, but are kept for future work.
     */
}

void audio_features(const int8_t *features_selector,
                    const int16_t *sig_q14,
                    int16_t len,
                    int16_t fs,
                    fxp_q16_t *feats_q16)
{
    audio_features_fxp_q16_from_q14(features_selector, sig_q14, len, fs, feats_q16);
}

static void _imu_features_fxp_q16_from_raw_impl(const int8_t *features_selector,
                                                const q11_5_t raw_fxp[][Num_IMU_signals],
                                                int16_t len,
                                                fxp_q16_t *feats_q16)
{
    uq10_6_t *combo_l2a = (uq10_6_t *)malloc((size_t)len * sizeof(*combo_l2a));
    uq5_11_t *combo_l2g = (uq5_11_t *)malloc((size_t)len * sizeof(*combo_l2g));
    q11_5_t *axis_samples = (q11_5_t *)malloc((size_t)len * sizeof(q11_5_t));

    if (!combo_l2a || !combo_l2g || !axis_samples) {
        free(combo_l2a);
        free(combo_l2g);
        free(axis_samples);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        combo_l2a[i] = fxp_l2_norm_accel_from_raw(raw_fxp[i][0], raw_fxp[i][1], raw_fxp[i][2]);
        combo_l2g[i] = fxp_l2_norm_gyro_from_raw(raw_fxp[i][3], raw_fxp[i][4], raw_fxp[i][5]);
    }

    const int8_t axis_ids[Num_IMU_signals] = {
        ACCELEROMETER_X, ACCELEROMETER_Y, ACCELEROMETER_Z,
        GYROSCOPE_Y, GYROSCOPE_P, GYROSCOPE_R
    };
    const int8_t axis_feat_base[Num_IMU_signals] = {
        ACCEL_X_FEAT, ACCEL_Y_FEAT, ACCEL_Z_FEAT,
        GYRO_Y_FEAT, GYRO_P_FEAT, GYRO_R_FEAT
    };

    for (int s = 0; s < Num_IMU_signals; s++) {
        int8_t base = axis_feat_base[s];
        if (!_is_required(features_selector, base, (uint16_t)(base + Num_imu_feat_families - 1))) continue;

        for (int16_t i = 0; i < len; i++) {
            axis_samples[i] = raw_fxp[i][axis_ids[s]];
        }
        imu_sig_raw_t sig_raw = {.data = axis_samples, .len = len};
        imu_run_feature_table_q16(&features_selector[base], imu_view_from_raw(sig_raw), &feats_q16[base]);
    }

    if (_is_required(features_selector, ACCEL_COMBO, (uint16_t)(ACCEL_COMBO + Num_imu_feat_families - 1))) {
        imu_sig_l2a_t s_l2a = {.data = combo_l2a, .len = len};
        imu_run_feature_table_q16(&features_selector[ACCEL_COMBO], imu_view_from_l2a(s_l2a), &feats_q16[ACCEL_COMBO]);
    }

    if (_is_required(features_selector, GYRO_COMBO, (uint16_t)(GYRO_COMBO + Num_imu_feat_families - 1))) {
        imu_sig_l2g_t s_l2g = {.data = combo_l2g, .len = len};
        imu_run_feature_table_q16(&features_selector[GYRO_COMBO], imu_view_from_l2g(s_l2g), &feats_q16[GYRO_COMBO]);
    }

    /*
     * Remaining IMU float-only families are intentionally left untouched and
     * are not dispatched in the current selected-feature pipeline.
     */

    free(combo_l2a);
    free(combo_l2g);
    free(axis_samples);
}

void audio_features_fxp_q16(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig || !feats_q16 || len <= 0 || fs <= 0) return;

    int16_t *sig_q14 = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    if (!sig_q14) return;

    /* Single allowed float->FxP conversion boundary for audio input carriers. */
    for (int16_t i = 0; i < len; i++) {
        sig_q14[i] = FXP_AUDIO_FROM_FLOAT(sig[i]);
    }

    audio_features_fxp_q16_from_q14(features_selector, sig_q14, len, fs, feats_q16);
    free(sig_q14);
}

void imu_features_fxp_q16_from_raw(const int8_t *features_selector,
                                   const q11_5_t sig_raw[][Num_IMU_signals],
                                   int16_t len,
                                   fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig_raw || !feats_q16 || len <= 0) return;
    _imu_features_fxp_q16_from_raw_impl(features_selector, sig_raw, len, feats_q16);
}

void imu_features(const int8_t *features_selector,
                  const q11_5_t sig_raw[][Num_IMU_signals],
                  int16_t len,
                  fxp_q16_t *feats_q16)
{
    imu_features_fxp_q16_from_raw(features_selector, sig_raw, len, feats_q16);
}

void imu_features_fxp_q16(const int8_t *features_selector, const float sig[][Num_IMU_signals], int16_t len, fxp_q16_t *feats_q16)
{
    if (!features_selector || !sig || !feats_q16 || len <= 0) return;

    q11_5_t(*raw_fxp)[Num_IMU_signals] = (q11_5_t(*)[Num_IMU_signals])malloc((size_t)len * sizeof(*raw_fxp));
    if (!raw_fxp) return;

    for (int16_t i = 0; i < len; i++) {
        /* Single allowed float->FxP conversion boundary for IMU input carriers. */
        raw_fxp[i][0] = FXP_IMU_RAW_FROM_FLOAT(sig[i][0]);
        raw_fxp[i][1] = FXP_IMU_RAW_FROM_FLOAT(sig[i][1]);
        raw_fxp[i][2] = FXP_IMU_RAW_FROM_FLOAT(sig[i][2]);
        raw_fxp[i][3] = FXP_IMU_RAW_FROM_FLOAT(sig[i][3]);
        raw_fxp[i][4] = FXP_IMU_RAW_FROM_FLOAT(sig[i][4]);
        raw_fxp[i][5] = FXP_IMU_RAW_FROM_FLOAT(sig[i][5]);
    }

    _imu_features_fxp_q16_from_raw_impl(features_selector, raw_fxp, len, feats_q16);

    free(raw_fxp);
}

#endif
