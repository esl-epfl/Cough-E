// =============================================================================
// fxp_accuracy_harness.c — Per-window FxP vs float accuracy measurement
//
// Compiled by run_accuracy_analysis.py with:
//   -DIMU_HEADER="path/to/imu_input_*.h"
//   -DFXP_MODE
//
// For every sliding window (50 samples, step 25) it runs the float path and
// the FxP path for every kernel × signal-type combination that produces a
// model input.  Prints one CSV row per (window, kernel, signal_type):
//
//   window,kernel,signal_type,float_val,fxp_val,abs_err,rel_err
//
// No colour output — intended for machine parsing by the Python driver.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <feature_extraction.h>
#include <imu_features.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>
#include <fxp.h>

#ifndef IMU_HEADER
#  error "Compile with -DIMU_HEADER=\"path/to/imu_input_*.h\""
#endif
#include IMU_HEADER   // defines imu_in[IMU_LEN][6], IMU_FS, IMU_LEN

#define WIN_LEN   WINDOW_SAMP_IMU   // 50
#define WIN_STEP  IMU_STEP          // 25

// ── helpers ──────────────────────────────────────────────────────────────────

static float rel_err(float fval, float xval)
{
    float ae = fabsf(xval - fval);
    return (fabsf(fval) > 1e-9f) ? ae / fabsf(fval) : ae;
}

static void emit(int win, const char *kernel, const char *sig,
                 float fval, float xval)
{
    float ae = fabsf(xval - fval);
    float re = rel_err(fval, xval);
    printf("%d,%s,%s,%.9g,%.9g,%.9g,%.9g\n",
           win, kernel, sig, fval, xval, ae, re);
}

// ── AZC epsilon values tested ─────────────────────────────────────────────────
static const float AZC_EPS[] = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
#define N_AZC_EPS  8

// ── main ─────────────────────────────────────────────────────────────────────

int main(void)
{
    // CSV header
    printf("window,kernel,signal_type,float_val,fxp_val,abs_err,rel_err\n");

    int n_wins = (IMU_LEN - WIN_LEN) / WIN_STEP + 1;

    for (int w = 0; w < n_wins; w++) {
        int start = w * WIN_STEP;

        // ── Build float signal arrays for this window ─────────────────────
        float raw[6][WIN_LEN];
        float l2a_f[WIN_LEN];
        float l2g_f[WIN_LEN];

        for (int i = 0; i < WIN_LEN; i++) {
            for (int ax = 0; ax < 6; ax++)
                raw[ax][i] = imu_in[start + i][ax];
            l2a_f[i] = sqrtf(raw[0][i]*raw[0][i] + raw[1][i]*raw[1][i] + raw[2][i]*raw[2][i]);
            l2g_f[i] = sqrtf(raw[3][i]*raw[3][i] + raw[4][i]*raw[4][i] + raw[5][i]*raw[5][i]);
        }

        // ── Build FxP signal arrays ───────────────────────────────────────
        q11_5_t  raw_fxp[6][WIN_LEN];
        uq10_6_t l2a_fxp[WIN_LEN];
        uq5_11_t l2g_fxp[WIN_LEN];

        for (int i = 0; i < WIN_LEN; i++) {
            for (int ax = 0; ax < 6; ax++)
                raw_fxp[ax][i] = FXP_IMU_RAW_FROM_FLOAT(raw[ax][i]);
            l2a_fxp[i] = fxp_l2_norm_accel(raw[0][i], raw[1][i], raw[2][i]);
            l2g_fxp[i] = fxp_l2_norm_gyro( raw[3][i], raw[4][i], raw[5][i]);
        }

        // axis labels matching signal names used in the model
        static const char *ax_names[6] = {
            "ACCEL_X","ACCEL_Y","ACCEL_Z","GYRO_Y","GYRO_P","GYRO_R"
        };

        // ── ZCR (RAW only — unsigned signals always give ZCR=0) ──────────
        for (int ax = 0; ax < 6; ax++) {
            float fv = compute_zrc(raw[ax], WIN_LEN);
            float xv = fxp_compute_zcr_raw(raw_fxp[ax], WIN_LEN);
            emit(w, "compute_zrc", ax_names[ax], fv, xv);
        }

        // ── L2 norm (first sample per window as representative) ──────────
        emit(w, "L2_norm", "COMBO_ACCEL", l2a_f[0],
             FXP_TO_FLOAT(l2a_fxp[0], 6));
        emit(w, "L2_norm", "COMBO_GYRO",  l2g_f[0],
             FXP_TO_FLOAT(l2g_fxp[0], 11));

        // ── RMS ───────────────────────────────────────────────────────────
        for (int ax = 0; ax < 6; ax++) {
            float fv = get_rms(raw[ax], WIN_LEN);
            uq16_16_t xv = fxp_get_rms_raw(raw_fxp[ax], WIN_LEN);
            emit(w, "get_rms", ax_names[ax], fv, FXP_TO_FLOAT(xv, 16));
        }
        {
            float fv = get_rms(l2a_f, WIN_LEN);
            uq13_3_t xv = fxp_get_rms_l2a(l2a_fxp, WIN_LEN);
            emit(w, "get_rms", "COMBO_ACCEL", fv, FXP_TO_FLOAT(xv, 3));
        }
        {
            float fv = get_rms(l2g_f, WIN_LEN);
            uq7_9_t xv = fxp_get_rms_l2g(l2g_fxp, WIN_LEN);
            emit(w, "get_rms", "COMBO_GYRO", fv, FXP_TO_FLOAT(xv, 9));
        }

        // ── Line length ───────────────────────────────────────────────────
        for (int ax = 0; ax < 6; ax++) {
            float fv = get_line_length(raw[ax], WIN_LEN);
            uq9_23_t xv = fxp_get_line_length_raw(raw_fxp[ax], WIN_LEN);
            emit(w, "get_line_length", ax_names[ax], fv, FXP_TO_FLOAT(xv, 23));
        }
        {
            float fv = get_line_length(l2g_f, WIN_LEN);
            uq7_9_t xv = fxp_get_line_length_l2g(l2g_fxp, WIN_LEN);
            emit(w, "get_line_length", "COMBO_GYRO", fv, FXP_TO_FLOAT(xv, 9));
        }

        // ── Kurtosis (RAW only) ───────────────────────────────────────────
        for (int ax = 0; ax < 6; ax++) {
            float fv = get_kurtosis(raw[ax], WIN_LEN);
            q34_30_t xv = fxp_get_kurtosis_raw(raw_fxp[ax], WIN_LEN);
            emit(w, "get_kurtosis", ax_names[ax], fv, FXP_TO_FLOAT(xv, 30));
        }

        // ── Crest factor (L2_G only) ──────────────────────────────────────
        {
            float rms_f  = get_rms(l2g_f, WIN_LEN);
            float peak_f = get_max(l2g_f, WIN_LEN);
            float fv = (rms_f > 1e-9f) ? peak_f / rms_f : 0.0f;
            uq7_9_t  rms_x  = fxp_get_rms_l2g(l2g_fxp, WIN_LEN);
            uq5_11_t peak_x = fxp_get_max_l2g(l2g_fxp, WIN_LEN);
            uq2_14_t cf_x   = (rms_x > 0) ? fxp_cf_l2g_result(peak_x, rms_x) : 0;
            emit(w, "crest_factor", "COMBO_GYRO", fv, FXP_TO_FLOAT(cf_x, 14));
        }

        // ── AZC ───────────────────────────────────────────────────────────
        for (int e = 0; e < N_AZC_EPS; e++) {
            float eps = AZC_EPS[e];
            char sig_eps[32];

            for (int ax = 0; ax < 6; ax++) {
                snprintf(sig_eps, sizeof(sig_eps), "%s", ax_names[ax]);
                float fv = (float)azc_computation(raw[ax], WIN_LEN, eps);
                float xv = (float)fxp_azc_computation_raw(raw_fxp[ax], WIN_LEN, eps);
                // label: kernel includes eps to distinguish the 8 AZC variants
                char kern[32];
                snprintf(kern, sizeof(kern), "azc_%.1f", eps);
                emit(w, kern, sig_eps, fv, xv);
            }
            {
                float fv = (float)azc_computation(l2a_f, WIN_LEN, eps);
                float xv = (float)fxp_azc_computation_l2a(l2a_fxp, WIN_LEN, eps);
                char kern[32]; snprintf(kern, sizeof(kern), "azc_%.1f", eps);
                emit(w, kern, "COMBO_ACCEL", fv, xv);
            }
            {
                float fv = (float)azc_computation(l2g_f, WIN_LEN, eps);
                float xv = (float)fxp_azc_computation_l2g(l2g_fxp, WIN_LEN, eps);
                char kern[32]; snprintf(kern, sizeof(kern), "azc_%.1f", eps);
                emit(w, kern, "COMBO_GYRO", fv, xv);
            }
        }
    }

    return 0;
}
