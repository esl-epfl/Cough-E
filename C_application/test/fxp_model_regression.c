#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <imu_features.h>
#include <imu_model.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>
#include <imu/imu_kernels.h>

#include <input_data/imu_input_55502_w0_9wnds.h>

#define WIN_LEN  WINDOW_SAMP_IMU
#define WIN_STEP IMU_STEP

static const char *k_sig_name[8] = {
    "ACCEL_X", "ACCEL_Y", "ACCEL_Z",
    "GYRO_Y", "GYRO_P", "GYRO_R",
    "COMBO_ACCEL", "COMBO_GYRO"
};

static const char *k_fam_name[Num_imu_feat_families] = {
    "LINE_LENGTH",
    "ZERO_CROSSING_RATE_IMU",
    "KURTOSIS",
    "ROOT_MEANS_SQUARED_IMU",
    "CREST_FACTOR_IMU",
    "AZC_0",
    "AZC_1",
    "AZC_2",
    "AZC_3",
    "AZC_4",
    "AZC_5",
    "AZC_6",
    "AZC_7"
};

static float abs_threshold_for_feature(int fam_idx)
{
    if (fam_idx == LINE_LENGTH) return 0.02f;
    if (fam_idx == ZERO_CROSSING_RATE_IMU) return 0.10f;
    if (fam_idx == KURTOSIS) return 0.20f;
    if (fam_idx == ROOT_MEANS_SQUARED_IMU) return 0.10f;
    if (fam_idx == CREST_FACTOR_IMU) return 0.02f;
    return 6.0f; // AZC is count-based
}

static int evaluate_feature(
    int sig_idx, int fam_idx,
    float raw[6][WIN_LEN], q11_5_t raw_fxp[6][WIN_LEN],
    float *l2a_f, float *l2g_f, uq10_6_t *l2a_fxp, uq5_11_t *l2g_fxp,
    float *float_val, float *fxp_val)
{
    int azc_i = fam_idx - APPROXIMATE_ZERO_CROSSING;

    if (fam_idx == LINE_LENGTH) {
        if (sig_idx < 6) {
            *float_val = get_line_length(raw[sig_idx], WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_line_length_raw(raw_fxp[sig_idx], WIN_LEN),
                                    FXP_FRAC_IMU_LINE_LENGTH_RAW);
            return 1;
        }
        if (sig_idx == 7) {
            *float_val = get_line_length(l2g_f, WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_line_length_l2g(l2g_fxp, WIN_LEN),
                                    FXP_FRAC_IMU_LINE_LENGTH_L2G);
            return 1;
        }
        return 0;
    }

    if (fam_idx == ZERO_CROSSING_RATE_IMU) {
        if (sig_idx < 6) {
            *float_val = compute_zrc(raw[sig_idx], WIN_LEN);
            *fxp_val = fxp_compute_zcr_raw(raw_fxp[sig_idx], WIN_LEN);
            return 1;
        }
        return 0;
    }

    if (fam_idx == KURTOSIS) {
        if (sig_idx < 6) {
            *float_val = get_kurtosis(raw[sig_idx], WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_kurtosis_raw(raw_fxp[sig_idx], WIN_LEN),
                                    FXP_FRAC_IMU_KURTOSIS_RAW);
            return 1;
        }
        return 0;
    }

    if (fam_idx == ROOT_MEANS_SQUARED_IMU) {
        if (sig_idx < 6) {
            *float_val = get_rms(raw[sig_idx], WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_rms_raw(raw_fxp[sig_idx], WIN_LEN),
                                    FXP_FRAC_IMU_RMS_RAW);
            return 1;
        }
        if (sig_idx == 6) {
            *float_val = get_rms(l2a_f, WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_rms_l2a(l2a_fxp, WIN_LEN),
                                    FXP_FRAC_IMU_RMS_L2A);
            return 1;
        }
        if (sig_idx == 7) {
            *float_val = get_rms(l2g_f, WIN_LEN);
            *fxp_val = FXP_TO_FLOAT(fxp_get_rms_l2g(l2g_fxp, WIN_LEN),
                                    FXP_FRAC_IMU_RMS_L2G);
            return 1;
        }
        return 0;
    }

    if (fam_idx == CREST_FACTOR_IMU) {
        if (sig_idx == 7) {
            float rms_f = get_rms(l2g_f, WIN_LEN);
            float peak_f = get_max(l2g_f, WIN_LEN);
            uq7_9_t rms_x = fxp_get_rms_l2g(l2g_fxp, WIN_LEN);
            uq5_11_t peak_x = fxp_get_max_l2g(l2g_fxp, WIN_LEN);
            uq2_14_t cf_x = (rms_x > 0) ? fxp_cf_l2g_result(peak_x, rms_x) : 0;
            *float_val = (rms_f > 0.0f) ? (peak_f / rms_f) : 0.0f;
            *fxp_val = FXP_TO_FLOAT(cf_x, FXP_FRAC_IMU_CREST_L2G);
            return 1;
        }
        return 0;
    }

    if (fam_idx >= APPROXIMATE_ZERO_CROSSING && fam_idx < APPROXIMATE_ZERO_CROSSING + N_AZC) {
        float eps = EPSILON_START + (EPSILON_STEP * azc_i);
        if (sig_idx < 6) {
            *float_val = (float)azc_computation(raw[sig_idx], WIN_LEN, eps);
            *fxp_val = (float)fxp_azc_computation_raw(raw_fxp[sig_idx], WIN_LEN, eps);
            return 1;
        }
        if (sig_idx == 6) {
            *float_val = (float)azc_computation(l2a_f, WIN_LEN, eps);
            *fxp_val = (float)fxp_azc_computation_l2a(l2a_fxp, WIN_LEN, eps);
            return 1;
        }
        if (sig_idx == 7) {
            *float_val = (float)azc_computation(l2g_f, WIN_LEN, eps);
            *fxp_val = (float)fxp_azc_computation_l2g(l2g_fxp, WIN_LEN, eps);
            return 1;
        }
        return 0;
    }

    return 0;
}

int main(void)
{
    int n_wins = (IMU_LEN - WIN_LEN) / WIN_STEP + 1;
    int failures = 0;
    int checks = 0;
    float worst_abs = -1.0f;
    float worst_thr = 0.0f;
    float worst_float = 0.0f;
    float worst_fxp = 0.0f;
    int worst_w = -1;
    int worst_sig = -1;
    int worst_fam = -1;

    for (int w = 0; w < n_wins; w++) {
        int start = w * WIN_STEP;
        float raw[6][WIN_LEN];
        q11_5_t raw_fxp[6][WIN_LEN];
        float l2a_f[WIN_LEN];
        float l2g_f[WIN_LEN];
        uq10_6_t l2a_fxp[WIN_LEN];
        uq5_11_t l2g_fxp[WIN_LEN];

        for (int i = 0; i < WIN_LEN; i++) {
            for (int ax = 0; ax < 6; ax++) {
                raw[ax][i] = imu_in[start + i][ax];
                raw_fxp[ax][i] = FXP_IMU_RAW_FROM_FLOAT(raw[ax][i]);
            }
            l2a_f[i] = sqrtf(raw[0][i] * raw[0][i] + raw[1][i] * raw[1][i] + raw[2][i] * raw[2][i]);
            l2g_f[i] = sqrtf(raw[3][i] * raw[3][i] + raw[4][i] * raw[4][i] + raw[5][i] * raw[5][i]);
            l2a_fxp[i] = fxp_l2_norm_accel_from_raw(raw_fxp[0][i], raw_fxp[1][i], raw_fxp[2][i]);
            l2g_fxp[i] = fxp_l2_norm_gyro_from_raw(raw_fxp[3][i], raw_fxp[4][i], raw_fxp[5][i]);
        }

        for (int sig = 0; sig < 8; sig++) {
            int base = sig * Num_imu_feat_families;
            for (int fam = 0; fam < Num_imu_feat_families; fam++) {
                if (imu_features_selector[base + fam] != 1) continue;

                float fval = 0.0f, xval = 0.0f;
                if (!evaluate_feature(sig, fam, raw, raw_fxp, l2a_f, l2g_f, l2a_fxp, l2g_fxp, &fval, &xval)) {
                    continue;
                }

                checks++;
                float abs_err = fabsf(xval - fval);
                float thr = abs_threshold_for_feature(fam);
                if (abs_err > worst_abs) {
                    worst_abs = abs_err;
                    worst_thr = thr;
                    worst_float = fval;
                    worst_fxp = xval;
                    worst_w = w;
                    worst_sig = sig;
                    worst_fam = fam;
                }
                if (abs_err > thr) {
                    failures++;
                    fprintf(stderr,
                            "FAIL w=%d sig=%s fam=%s float=%.9g fxp=%.9g abs=%.9g thr=%.9g\n",
                            w, k_sig_name[sig], k_fam_name[fam], fval, xval, abs_err, thr);
                }
            }
        }
    }

    if (checks == 0) {
        fprintf(stderr, "FxP IMU discrepancy test: no valid checks were executed.\n");
        return 1;
    }

    fprintf(stderr,
            "FxP IMU discrepancy summary: checks=%d failures=%d worst_abs=%.9g (w=%d sig=%s fam=%s float=%.9g fxp=%.9g thr=%.9g)\n",
            checks, failures, worst_abs, worst_w, k_sig_name[worst_sig],
            k_fam_name[worst_fam], worst_float, worst_fxp, worst_thr);

    if (failures > 0) {
        fprintf(stderr, "FxP IMU discrepancy test: FAIL (%d/%d checks failed).\n", failures, checks);
        return 1;
    }

    printf("FxP IMU discrepancy test: PASS (%d checks)\n", checks);
    return 0;
}
