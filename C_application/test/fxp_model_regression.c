#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <imu_features.h>
#include <imu_model.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>
#include <imu/imu_kernels.h>

#ifndef IMU_HEADER
#define IMU_HEADER <input_data/imu_input_55502_w0_9wnds.h>
#endif
#include IMU_HEADER

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

typedef struct {
    int used;
    int is_count_based;
    int n;
    double sum_sq_err;
    double sum_sq_float;
    double sum_abs_err;
    double sum_abs_float;
    int exact_match_count;
    float max_abs;
    float max_abs_float;
} metric_acc_t;

static int is_count_based_family(int fam_idx)
{
    return (fam_idx >= APPROXIMATE_ZERO_CROSSING);
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
    int checks = 0;
    metric_acc_t stats[8][Num_imu_feat_families] = {0};
    float global_max_abs = 0.0f;
    float global_cont_max_abs = 0.0f;
    float global_count_max_abs = 0.0f;

    double global_cont_sq_err = 0.0;
    double global_cont_sq_float = 0.0;
    int global_cont_n = 0;

    double global_count_abs_err = 0.0;
    double global_count_abs_float = 0.0;
    double global_count_sum_abs = 0.0;
    int global_count_n = 0;

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
                metric_acc_t *m = &stats[sig][fam];
                m->used = 1;
                m->is_count_based = is_count_based_family(fam);
                m->n += 1;
                m->sum_sq_err += (double)abs_err * (double)abs_err;
                m->sum_abs_err += (double)abs_err;
                m->sum_abs_float += fabs((double)fval);
                if (fabsf(xval - fval) < 0.5f) {
                    m->exact_match_count += 1;
                }
                if ((double)abs_err > m->max_abs) {
                    m->max_abs = abs_err;
                }
                if (fabsf(fval) > m->max_abs_float) {
                    m->max_abs_float = fabsf(fval);
                }
                if (abs_err > global_max_abs) {
                    global_max_abs = abs_err;
                }

                if (m->is_count_based) {
                    global_count_n += 1;
                    global_count_abs_err += (double)abs_err;
                    global_count_abs_float += fabs((double)fval);
                    global_count_sum_abs += (double)abs_err;
                    if (abs_err > global_count_max_abs) global_count_max_abs = abs_err;
                } else {
                    global_cont_n += 1;
                    global_cont_sq_err += (double)abs_err * (double)abs_err;
                    global_cont_sq_float += (double)fval * (double)fval;
                    m->sum_sq_float += (double)fval * (double)fval;
                    if (abs_err > global_cont_max_abs) global_cont_max_abs = abs_err;
                }
            }
        }
    }

    if (checks == 0) {
        fprintf(stderr, "FxP IMU discrepancy test: no valid checks were executed.\n");
        return 1;
    }

    printf("\nIMU FxP Error Tables (vs float baseline)\n");
    printf("========================================\n");

    printf("\n[Continuous Features] RMSE and RelRMSE%%\n");
    printf("%-12s %-24s %6s %12s %12s %12s %12s\n",
           "Signal", "Feature", "N", "RMSE", "RelRMSE%", "MaxAbs", "MaxAbs%");
    printf("--------------------------------------------------------------------------------\n");
    for (int sig = 0; sig < 8; sig++) {
        for (int fam = 0; fam < Num_imu_feat_families; fam++) {
            metric_acc_t *m = &stats[sig][fam];
            if (!m->used || m->n == 0 || m->is_count_based) continue;
            double rmse = sqrt(m->sum_sq_err / (double)m->n);
            double baseline_rms = (m->sum_sq_float > 0.0) ? sqrt(m->sum_sq_float / (double)m->n) : 0.0;
            double rel_rmse_pct = (baseline_rms > 0.0) ? (100.0 * rmse / baseline_rms) : 0.0;
            double max_abs_pct = (m->max_abs_float > 0.0f) ? (100.0 * (double)m->max_abs / (double)m->max_abs_float) : 0.0;
            printf("%-12s %-24s %6d %12.6g %12.4f %12.6g %12.4f\n",
                   k_sig_name[sig], k_fam_name[fam], m->n, rmse, rel_rmse_pct, m->max_abs, max_abs_pct);
        }
    }
    if (global_cont_n > 0) {
        double g_rmse = sqrt(global_cont_sq_err / (double)global_cont_n);
        double g_baseline_rms = (global_cont_sq_float > 0.0) ? sqrt(global_cont_sq_float / (double)global_cont_n) : 0.0;
        double g_rel_rmse = (g_baseline_rms > 0.0) ? (100.0 * g_rmse / g_baseline_rms) : 0.0;
        printf("--------------------------------------------------------------------------------\n");
        printf("%-12s %-24s %6d %12.6g %12.4f %12.6g %12s\n",
               "ALL", "CONTINUOUS", global_cont_n, g_rmse, g_rel_rmse, global_cont_max_abs, "-");
    }

    printf("\n[Count-Based Features] Alternative metric = WAPE%%\n");
    printf("WAPE%% = 100 * sum(|FxP - Float|) / sum(|Float|)\n");
    printf("%-12s %-24s %6s %12s %12s %12s %12s %12s\n",
           "Signal", "Feature", "N", "MAE(cnt)", "WAPE%", "MaxAbs", "MaxAbs%", "Exact%");
    printf("------------------------------------------------------------------------------------------------\n");
    for (int sig = 0; sig < 8; sig++) {
        for (int fam = 0; fam < Num_imu_feat_families; fam++) {
            metric_acc_t *m = &stats[sig][fam];
            if (!m->used || m->n == 0 || !m->is_count_based) continue;
            double mae = m->sum_abs_err / (double)m->n;
            double wape = (m->sum_abs_float > 0.0) ? (100.0 * m->sum_abs_err / m->sum_abs_float) : 0.0;
            double max_abs_pct = (m->max_abs_float > 0.0f) ? (100.0 * (double)m->max_abs / (double)m->max_abs_float) : 0.0;
            double exact = 100.0 * (double)m->exact_match_count / (double)m->n;
            printf("%-12s %-24s %6d %12.6g %12.4f %12.6g %12.4f %12.2f\n",
                   k_sig_name[sig], k_fam_name[fam], m->n, mae, wape, m->max_abs, max_abs_pct, exact);
        }
    }
    if (global_count_n > 0) {
        double g_mae = global_count_sum_abs / (double)global_count_n;
        double g_wape = (global_count_abs_float > 0.0) ? (100.0 * global_count_abs_err / global_count_abs_float) : 0.0;
        printf("------------------------------------------------------------------------------------------------\n");
        printf("%-12s %-24s %6d %12.6g %12.4f %12.6g %12s %12s\n",
               "ALL", "COUNT_BASED", global_count_n, g_mae, g_wape, global_count_max_abs, "-", "-");
    }

    printf("\nSummary: checks=%d global_max_abs=%.6g\n", checks, global_max_abs);

    /* Machine-readable lines for full-dataset aggregation scripts. */
    printf("REG_METRICS_CONT,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
           global_cont_n, global_cont_sq_err, global_cont_sq_float, global_cont_max_abs);
    printf("REG_METRICS_COUNT,n=%d,sum_abs_err=%.17g,sum_abs_float=%.17g,max_abs=%.9g\n",
           global_count_n, global_count_abs_err, global_count_abs_float, global_count_max_abs);
    printf("REG_METRICS_META,checks=%d,global_max_abs=%.9g\n", checks, global_max_abs);

    /* Per signal-feature machine-readable metrics for kernel-wise aggregation. */
    for (int sig = 0; sig < 8; sig++) {
        for (int fam = 0; fam < Num_imu_feat_families; fam++) {
            metric_acc_t *m = &stats[sig][fam];
            if (!m->used || m->n == 0) continue;
            if (m->is_count_based) {
                printf("REG_KERNEL_COUNT,signal=%s,feature=%s,n=%d,sum_abs_err=%.17g,sum_abs_float=%.17g,max_abs=%.9g\n",
                       k_sig_name[sig], k_fam_name[fam], m->n, m->sum_abs_err, m->sum_abs_float, m->max_abs);
            } else {
                printf("REG_KERNEL_CONT,signal=%s,feature=%s,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
                       k_sig_name[sig], k_fam_name[fam], m->n, m->sum_sq_err, m->sum_sq_float, m->max_abs);
            }
        }
    }
    return 0;
}
