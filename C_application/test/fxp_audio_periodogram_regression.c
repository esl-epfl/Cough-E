#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <audio_features.h>
#include <frequency_features.h>
#include <welch_psd.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)
#include <audio/audio_periodogram_block.h>
#endif

#define FS_AUDIO 8000
#define SIG_LEN 6400
#define PSD_LEN ((NPERSEG / 2) + 1)
#define N_SIGNALS 5
#define PI_F 3.14159265358979323846f

typedef struct {
    const char *name;
    int n;
    double sum_sq_err;
    double sum_sq_float;
    float max_abs;
} metric_acc_t;

static const int k_feat_idx[2 + N_PSD] = {
    SPECTRAL_FLATNESS,
    DOMINANT_FREQUENCY,
    POWER_SPECTRAL_DENSITY,
    POWER_SPECTRAL_DENSITY + 1,
    POWER_SPECTRAL_DENSITY + 2,
};

static const char *k_feat_name[2 + N_PSD] = {
    "SPECTRAL_FLATNESS",
    "DOMINANT_FREQUENCY",
    "PSD_BAND_1",
    "PSD_BAND_2",
    "PSD_BAND_3",
};

static float _rand_uniform(unsigned *state)
{
    *state = (*state * 1103515245u) + 12345u;
    return ((float)((*state >> 8) & 0x00FFFFFFu) / 8388608.0f) - 1.0f;
}

static void build_signal(int signal_id, float *x)
{
    unsigned rng = 0xB7B7A5u + (unsigned)signal_id * 761u;
    for (int i = 0; i < SIG_LEN; i++) {
        float t = (float)i / (float)SIG_LEN;
        switch (signal_id) {
            case 0:
                x[i] = 0.80f * sinf(2.0f * PI_F * 12.0f * t);
                break;
            case 1:
                x[i] = 0.55f * sinf(2.0f * PI_F * 6.0f * t)
                     + 0.30f * sinf(2.0f * PI_F * 31.0f * t + 0.3f);
                break;
            case 2: {
                float env = (t < 0.5f) ? (2.0f * t) : (2.0f * (1.0f - t));
                x[i] = env * sinf(2.0f * PI_F * 20.0f * t);
                break;
            }
            case 3: {
                float f = 4.0f + (90.0f - 4.0f) * t;
                x[i] = 0.75f * sinf(2.0f * PI_F * f * t);
                break;
            }
            default:
                x[i] = 0.80f * _rand_uniform(&rng);
                break;
        }
    }
}

static void _acc_update(metric_acc_t *m, float fxp_val, float ref_val)
{
    float abs_err = fabsf(fxp_val - ref_val);
    m->n += 1;
    m->sum_sq_err += (double)abs_err * (double)abs_err;
    m->sum_sq_float += (double)ref_val * (double)ref_val;
    if (abs_err > m->max_abs) m->max_abs = abs_err;
}

static void _print_machine_rows(metric_acc_t *rows, int n_rows, int checks, float global_max_abs)
{
    int n_total = 0;
    double sum_sq_err_total = 0.0;
    double sum_sq_float_total = 0.0;
    float max_total = 0.0f;

    for (int i = 0; i < n_rows; i++) {
        n_total += rows[i].n;
        sum_sq_err_total += rows[i].sum_sq_err;
        sum_sq_float_total += rows[i].sum_sq_float;
        if (rows[i].max_abs > max_total) max_total = rows[i].max_abs;
    }

    printf("AUDIO_PSD_REG_METRICS_CONT,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
           n_total, sum_sq_err_total, sum_sq_float_total, max_total);
    printf("AUDIO_PSD_REG_METRICS_META,checks=%d,global_max_abs=%.9g\n", checks, global_max_abs);
    for (int i = 0; i < n_rows; i++) {
        printf("AUDIO_PSD_REG_KERNEL_CONT,feature=%s,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
               rows[i].name, rows[i].n, rows[i].sum_sq_err, rows[i].sum_sq_float, rows[i].max_abs);
    }
}

static int run_isolated_suite(void)
{
    const int n_rows = 2 + N_PSD;
    metric_acc_t rows[2 + N_PSD];
    for (int i = 0; i < n_rows; i++) {
        rows[i].name = k_feat_name[i];
        rows[i].n = 0;
        rows[i].sum_sq_err = 0.0;
        rows[i].sum_sq_float = 0.0;
        rows[i].max_abs = 0.0f;
    }

    float *x = (float *)malloc(SIG_LEN * sizeof(float));
    float *psd = (float *)malloc(PSD_LEN * sizeof(float));
    float *freqs = (float *)malloc(PSD_LEN * sizeof(float));
    if (!x || !psd || !freqs) {
        fprintf(stderr, "Allocation failed.\n");
        free(x);
        free(psd);
        free(freqs);
        return 1;
    }

    int8_t selector[Number_AUDIO_Features];
    memset(selector, 0, sizeof(selector));
    selector[SPECTRAL_FLATNESS] = 1;
    selector[DOMINANT_FREQUENCY] = 1;
    for (int i = 0; i < N_PSD; i++) selector[POWER_SPECTRAL_DENSITY + i] = 1;

    int8_t psd_selector[N_PSD] = {1, 1, 1};
    float global_max_abs = 0.0f;
    int checks = 0;

    for (int s = 0; s < N_SIGNALS; s++) {
        build_signal(s, x);
        compute_periodogram(x, SIG_LEN, FS_AUDIO, psd, freqs);

        float ref[2 + N_PSD];
        ref[0] = compute_flatness(psd, PSD_LEN);
        ref[1] = get_domiant_freq(psd, freqs, PSD_LEN);

        float band_powers_ref[N_PSD] = {0};
        normalized_bandpowers(psd, freqs, PSD_LEN, psd_selector, band_powers_ref);
        for (int i = 0; i < N_PSD; i++) ref[2 + i] = band_powers_ref[i];

        float feats[Number_AUDIO_Features];
        memset(feats, 0, sizeof(feats));
        fxp_audio_periodogram_features_from_signal(selector, x, SIG_LEN, FS_AUDIO, feats);

        for (int i = 0; i < n_rows; i++) {
            float fxp_val = feats[k_feat_idx[i]];
            _acc_update(&rows[i], fxp_val, ref[i]);
            float abs_err = fabsf(fxp_val - ref[i]);
            if (abs_err > global_max_abs) global_max_abs = abs_err;
            checks++;
        }
    }

    printf("\nPeriodogram Audio FxP Kernel Regression (isolated suite)\n");
    printf("=========================================================\n");
    printf("%-22s %8s %14s %14s\n", "Kernel", "N", "RMSE", "RelRMSE(%)");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < n_rows; i++) {
        double rmse = (rows[i].n > 0) ? sqrt(rows[i].sum_sq_err / (double)rows[i].n) : 0.0;
        double baseline_rms = (rows[i].n > 0 && rows[i].sum_sq_float > 0.0)
                            ? sqrt(rows[i].sum_sq_float / (double)rows[i].n) : 0.0;
        double rel = (baseline_rms > 0.0) ? (100.0 * rmse / baseline_rms) : 0.0;
        printf("%-22s %8d %14.6f %14.6f\n", rows[i].name, rows[i].n, rmse, rel);
    }

    _print_machine_rows(rows, n_rows, checks, global_max_abs);

    free(x);
    free(psd);
    free(freqs);
    return 0;
}

#ifdef AUDIO_HEADER
#include AUDIO_HEADER

static int run_dataset_recording(void)
{
    if (AUDIO_LEN < WINDOW_SAMP_AUDIO) {
        fprintf(stderr, "AUDIO_LEN (%d) smaller than WINDOW_SAMP_AUDIO (%d).\n", AUDIO_LEN, WINDOW_SAMP_AUDIO);
        return 1;
    }

    const int n_rows = 2 + N_PSD;
    metric_acc_t rows[2 + N_PSD];
    for (int i = 0; i < n_rows; i++) {
        rows[i].name = k_feat_name[i];
        rows[i].n = 0;
        rows[i].sum_sq_err = 0.0;
        rows[i].sum_sq_float = 0.0;
        rows[i].max_abs = 0.0f;
    }

    int8_t selector[Number_AUDIO_Features];
    memset(selector, 0, sizeof(selector));
    selector[SPECTRAL_FLATNESS] = 1;
    selector[DOMINANT_FREQUENCY] = 1;
    for (int i = 0; i < N_PSD; i++) selector[POWER_SPECTRAL_DENSITY + i] = 1;

    int8_t psd_selector[N_PSD] = {1, 1, 1};

    float *psd = (float *)malloc(PSD_LEN * sizeof(float));
    float *freqs = (float *)malloc(PSD_LEN * sizeof(float));
    if (!psd || !freqs) {
        fprintf(stderr, "Allocation failed.\n");
        free(psd);
        free(freqs);
        return 1;
    }

    int n_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    float global_max_abs = 0.0f;
    int checks = 0;

#ifdef AUDIO_PSD_DEBUG_FLATNESS
    int n_flat_dumped = 0;
    const int MAX_FLAT_DUMPS = 8;
    const float FLAT_DUMP_THRESH = 0.15f;
#endif

    for (int w = 0; w < n_wins; w++) {
        int start = w * AUDIO_STEP;
        const float *sig = &audio_in.air[start];

        compute_periodogram(sig, WINDOW_SAMP_AUDIO, AUDIO_FS, psd, freqs);

        float ref[2 + N_PSD];
        ref[0] = compute_flatness(psd, PSD_LEN);
        ref[1] = get_domiant_freq(psd, freqs, PSD_LEN);
        float band_powers_ref[N_PSD] = {0};
        normalized_bandpowers(psd, freqs, PSD_LEN, psd_selector, band_powers_ref);
        for (int i = 0; i < N_PSD; i++) ref[2 + i] = band_powers_ref[i];

        float feats[Number_AUDIO_Features];
        memset(feats, 0, sizeof(feats));
        fxp_audio_periodogram_features_from_signal(selector, sig, WINDOW_SAMP_AUDIO, AUDIO_FS, feats);

#ifdef AUDIO_PSD_DEBUG_FLATNESS
        {
            float flat_err = fabsf(feats[SPECTRAL_FLATNESS] - ref[0]);
            if (flat_err > FLAT_DUMP_THRESH && n_flat_dumped < MAX_FLAT_DUMPS) {
                n_flat_dumped++;
                /* Float reference internals */
                float amean_ref = 0.0f;
                float sum_logs_ref = 0.0f;
                float min_bin = psd[0];
                float max_bin = psd[0];
                int n_subnorm = 0;
                int n_zero = 0;
                for (int i = 0; i < PSD_LEN; i++) {
                    amean_ref += psd[i];
                    sum_logs_ref += logf(psd[i]);
                    if (psd[i] < min_bin) min_bin = psd[i];
                    if (psd[i] > max_bin) max_bin = psd[i];
                    if (psd[i] == 0.0f) n_zero++;
                    else if (psd[i] < 1e-20f) n_subnorm++;
                }
                amean_ref /= (float)PSD_LEN;
                float mean_log_ref = sum_logs_ref / (float)PSD_LEN;
                float gmean_ref = expf(mean_log_ref);

                printf("FLAT_DBG,win=%d,flat_ref=%.9g,flat_fxp=%.9g,abs_err=%.9g,"
                       "amean=%.9g,gmean=%.9g,mean_log=%.6f,"
                       "min_bin=%.9g,max_bin=%.9g,dyn_range_dB=%.2f,n_zero=%d,n_subnorm=%d\n",
                       w, ref[0], feats[SPECTRAL_FLATNESS], flat_err,
                       amean_ref, gmean_ref, mean_log_ref,
                       min_bin, max_bin,
                       (min_bin > 0.0f && max_bin > 0.0f) ? 10.0f * log10f(max_bin / min_bin) : -1.0f,
                       n_zero, n_subnorm);
            }
        }
#endif

        for (int i = 0; i < n_rows; i++) {
            float fxp_val = feats[k_feat_idx[i]];
            _acc_update(&rows[i], fxp_val, ref[i]);
            float abs_err = fabsf(fxp_val - ref[i]);
            if (abs_err > global_max_abs) global_max_abs = abs_err;
            checks++;
        }
    }

    _print_machine_rows(rows, n_rows, checks, global_max_abs);
    free(psd);
    free(freqs);
    return 0;
}
#endif

int main(void)
{
#if !defined(FXP_MODE) || !defined(FIXED_POINT)
    fprintf(stderr, "This harness requires FXP_MODE and FIXED_POINT.\n");
    return 1;
#else
#ifdef AUDIO_HEADER
    return run_dataset_recording();
#else
    return run_isolated_suite();
#endif
#endif
}
