// =============================================================================
// fxp_test_imu.c — Fixed-point IMU kernel test harness
//
// Compiles with -DFXP_MODE against the real Cough-E source tree.
// Uses the first window (50 samples) from the hardcoded imu_input header.
//
// For each kernel × signal type:
//   1. Run the float version
//   2. Run the FxP version, convert result back to float
//   3. Print: kernel, signal type, float result, FxP result, abs error, rel error
//
// Build: see test/Makefile
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Pull in full Cough-E headers
#include <feature_extraction.h>
#include <imu_features.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>
#include <fxp.h>

// Input data (provides imu_in[IMU_LEN][6] and IMU_FS, IMU_LEN)
#include <input_data/imu_input_55502_w0_9wnds.h>

#define WIN_LEN     WINDOW_SAMP_IMU   // 50 samples
#define WIN_START   0                 // First window

// ── Colour output helpers ─────────────────────────────────────────────────────
#define COL_RESET  "\033[0m"
#define COL_PASS   "\033[32m"   // green
#define COL_WARN   "\033[33m"   // yellow
#define COL_FAIL   "\033[31m"   // red
#define COL_HEAD   "\033[1;36m" // bold cyan

// Relative error threshold for PASS / WARN / FAIL
#define THRESH_PASS  0.01f   // < 1%  : green
#define THRESH_WARN  0.05f   // < 5%  : yellow, else red

static void print_result(const char *kernel, const char *sigtype,
                          float fval, float xval)
{
    float aerr = fabsf(xval - fval);
    float rerr = (fabsf(fval) > 1e-6f) ? aerr / fabsf(fval) : aerr;
    const char *col = (rerr < THRESH_PASS) ? COL_PASS
                    : (rerr < THRESH_WARN) ? COL_WARN : COL_FAIL;
    printf("  %-22s %-6s  float=%10.5f  fxp=%10.5f  aerr=%9.5f  rerr=%s%.4f%%%s\n",
           kernel, sigtype, fval, xval, aerr, col, rerr * 100.0f, COL_RESET);
}

int main(void)
{
    // ── Build float signal arrays for the first window ────────────────────────
    float raw[6][WIN_LEN];       // raw[axis][sample]
    float l2a_f[WIN_LEN];        // float L2_A
    float l2g_f[WIN_LEN];        // float L2_G

    for(int i = 0; i < WIN_LEN; i++){
        for(int ax = 0; ax < 6; ax++)
            raw[ax][i] = imu_in[WIN_START + i][ax];

        // Float L2_A: sqrt(ax^2 + ay^2 + az^2)
        l2a_f[i] = sqrtf(raw[0][i]*raw[0][i] + raw[1][i]*raw[1][i] + raw[2][i]*raw[2][i]);
        // Float L2_G: sqrt(gx^2 + gy^2 + gz^2)  [indices 3,4,5]
        l2g_f[i] = sqrtf(raw[3][i]*raw[3][i] + raw[4][i]*raw[4][i] + raw[5][i]*raw[5][i]);
    }

    // ── Build FxP signal arrays ───────────────────────────────────────────────
    q11_5_t  raw_fxp[6][WIN_LEN];
    uq10_6_t l2a_fxp[WIN_LEN];
    uq5_11_t l2g_fxp[WIN_LEN];

    for(int i = 0; i < WIN_LEN; i++){
        for(int ax = 0; ax < 6; ax++)
            raw_fxp[ax][i] = FXP_IMU_RAW_FROM_FLOAT(raw[ax][i]);
        l2a_fxp[i] = fxp_l2_norm_accel(raw[0][i], raw[1][i], raw[2][i]);
        l2g_fxp[i] = fxp_l2_norm_gyro(raw[3][i], raw[4][i], raw[5][i]);
    }

    printf("\n%s=== FxP IMU Kernel Test  (window %d, len=%d) ===%s\n\n",
           COL_HEAD, WIN_START, WIN_LEN, COL_RESET);

    // =========================================================================
    // L2 norm (per-sample spot check: first sample only)
    // =========================================================================
    printf("%sL2 norm (first sample):%s\n", COL_HEAD, COL_RESET);
    print_result("L2_norm_accel", "L2_A",
                 l2a_f[0], FXP_TO_FLOAT(l2a_fxp[0], 6));
    print_result("L2_norm_gyro",  "L2_G",
                 l2g_f[0], FXP_TO_FLOAT(l2g_fxp[0], 11));

    // =========================================================================
    // RMS
    // =========================================================================
    printf("\n%sRMS:%s\n", COL_HEAD, COL_RESET);
    // Use accel_z (index 2) as representative RAW axis (large signal ~105)
    float rms_raw_f = get_rms(raw[2], WIN_LEN);
    uq16_16_t rms_raw_x = fxp_get_rms_raw(raw_fxp[2], WIN_LEN);
    print_result("get_rms", "RAW(az)", rms_raw_f, FXP_TO_FLOAT(rms_raw_x, 16));

    float rms_l2a_f = get_rms(l2a_f, WIN_LEN);
    uq13_3_t rms_l2a_x = fxp_get_rms_l2a(l2a_fxp, WIN_LEN);
    print_result("get_rms", "L2_A", rms_l2a_f, FXP_TO_FLOAT(rms_l2a_x, 3));

    float rms_l2g_f = get_rms(l2g_f, WIN_LEN);
    uq7_9_t rms_l2g_x = fxp_get_rms_l2g(l2g_fxp, WIN_LEN);
    print_result("get_rms", "L2_G", rms_l2g_f, FXP_TO_FLOAT(rms_l2g_x, 9));

    // =========================================================================
    // Line length
    // =========================================================================
    printf("\n%sLine Length:%s\n", COL_HEAD, COL_RESET);
    // Use accel_x (index 0) as RAW — has largest variation
    float ll_raw_f = get_line_length(raw[0], WIN_LEN);
    uq9_23_t ll_raw_x = fxp_get_line_length_raw(raw_fxp[0], WIN_LEN);
    print_result("get_line_length", "RAW(ax)", ll_raw_f, FXP_TO_FLOAT(ll_raw_x, 23));

    float ll_l2g_f = get_line_length(l2g_f, WIN_LEN);
    uq7_9_t ll_l2g_x = fxp_get_line_length_l2g(l2g_fxp, WIN_LEN);
    print_result("get_line_length", "L2_G", ll_l2g_f, FXP_TO_FLOAT(ll_l2g_x, 9));

    // =========================================================================
    // Kurtosis (RAW only)
    // =========================================================================
    printf("\n%sKurtosis:%s\n", COL_HEAD, COL_RESET);
    float kurt_f = get_kurtosis(raw[0], WIN_LEN);
    q34_30_t kurt_x = fxp_get_kurtosis_raw(raw_fxp[0], WIN_LEN);
    print_result("get_kurtosis", "RAW(ax)", kurt_f, FXP_TO_FLOAT(kurt_x, 30));

    // Also test with accel_z (near-constant ~105 -> kurtosis near -3 Fisher)
    kurt_f = get_kurtosis(raw[2], WIN_LEN);
    kurt_x = fxp_get_kurtosis_raw(raw_fxp[2], WIN_LEN);
    print_result("get_kurtosis", "RAW(az)", kurt_f, FXP_TO_FLOAT(kurt_x, 30));

    // =========================================================================
    // Crest factor (L2_G only)
    // =========================================================================
    printf("\n%sCrest Factor (L2_G):%s\n", COL_HEAD, COL_RESET);
    float peak_f  = get_max(l2g_f, WIN_LEN);
    float cf_f    = peak_f / rms_l2g_f;
    uq5_11_t peak_x = fxp_get_max_l2g(l2g_fxp, WIN_LEN);
    uq2_14_t cf_x   = fxp_cf_l2g_result(peak_x, rms_l2g_x);
    print_result("crest_factor", "L2_G", cf_f, FXP_TO_FLOAT(cf_x, 14));

    // =========================================================================
    // AZC (one epsilon value per signal type to keep output concise)
    // =========================================================================
    printf("\n%sAZC (eps=0.3):%s\n", COL_HEAD, COL_RESET);
    float eps = 0.3f;

    float azc_raw_f = (float)azc_computation(raw[0], WIN_LEN, eps);
    float azc_raw_x = (float)fxp_azc_computation_raw(raw_fxp[0], WIN_LEN, eps);
    print_result("azc_computation", "RAW(ax)", azc_raw_f, azc_raw_x);

    float azc_l2a_f = (float)azc_computation(l2a_f, WIN_LEN, eps);
    float azc_l2a_x = (float)fxp_azc_computation_l2a(l2a_fxp, WIN_LEN, eps);
    print_result("azc_computation", "L2_A", azc_l2a_f, azc_l2a_x);

    float azc_l2g_f = (float)azc_computation(l2g_f, WIN_LEN, eps);
    float azc_l2g_x = (float)fxp_azc_computation_l2g(l2g_fxp, WIN_LEN, eps);
    print_result("azc_computation", "L2_G", azc_l2g_f, azc_l2g_x);

    // Also test a larger epsilon (0.7) to exercise a different DP branch
    eps = 0.7f;
    printf("\n%sAZC (eps=0.7):%s\n", COL_HEAD, COL_RESET);
    azc_raw_f = (float)azc_computation(raw[0], WIN_LEN, eps);
    azc_raw_x = (float)fxp_azc_computation_raw(raw_fxp[0], WIN_LEN, eps);
    print_result("azc_computation", "RAW(ax)", azc_raw_f, azc_raw_x);

    azc_l2g_f = (float)azc_computation(l2g_f, WIN_LEN, eps);
    azc_l2g_x = (float)fxp_azc_computation_l2g(l2g_fxp, WIN_LEN, eps);
    print_result("azc_computation", "L2_G", azc_l2g_f, azc_l2g_x);

    printf("\n%sColour key: %sPASS(<1%%)%s  %sWARN(<5%%)%s  %sFAIL(>=5%%)%s\n\n",
           COL_HEAD, COL_PASS, COL_HEAD, COL_WARN, COL_HEAD, COL_FAIL, COL_RESET);

    return 0;
}
