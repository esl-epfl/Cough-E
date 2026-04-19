#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <audio_features.h>
#include <frequency_features.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)
#include <audio/audio_fft_block.h>
#endif

#define FS_AUDIO 8000
#define NFFT 6400
#define FFT_LEN ((NFFT / 2) + 1)
#define N_SIGNALS 5
#define PI_F 3.14159265358979323846f

typedef struct {
    const char *name;
    int n;
    double sum_sq_err;
    double sum_sq_float;
    float max_abs;
} metric_acc_t;

static const int k_feat_idx[4] = {
    SPECTRAL_ROLLOFF,
    SPECTRAL_CENTROID,
    SPECTRAL_SPREAD,
    SPECTRAL_KURTOSIS,
};

static const char *k_feat_name[4] = {
    "SPECTRAL_ROLLOFF",
    "SPECTRAL_CENTROID",
    "SPECTRAL_SPREAD",
    "SPECTRAL_KURTOSIS",
};

static float _rand_uniform(unsigned *state)
{
    *state = (*state * 1103515245u) + 12345u;
    return ((float)((*state >> 8) & 0x00FFFFFFu) / 8388608.0f) - 1.0f;
}

static void build_signal(int signal_id, float *x)
{
    unsigned rng = 0xA5A5A5u + (unsigned)signal_id * 977u;
    for (int i = 0; i < NFFT; i++) {
        float t = (float)i / (float)NFFT;
        switch (signal_id) {
            case 0:
                x[i] = (i == 0) ? 0.85f : 0.0f; /* impulse */
                break;
            case 1:
                x[i] = 0.85f * sinf(2.0f * PI_F * 7.0f * t);
                break;
            case 2:
                x[i] = 0.55f * sinf(2.0f * PI_F * 5.0f * t)
                     + 0.30f * sinf(2.0f * PI_F * 37.0f * t + 0.3f);
                break;
            case 3: {
                float f = 3.0f + (120.0f - 3.0f) * t;
                x[i] = 0.85f * sinf(2.0f * PI_F * f * t);
                break;
            }
            default:
                x[i] = 0.85f * _rand_uniform(&rng);
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

static void _print_machine_rows(metric_acc_t *rows, int checks, float global_max_abs)
{
    int n_total = 0;
    double sum_sq_err_total = 0.0;
    double sum_sq_float_total = 0.0;
    float max_total = 0.0f;

    for (int i = 0; i < 4; i++) {
        n_total += rows[i].n;
        sum_sq_err_total += rows[i].sum_sq_err;
        sum_sq_float_total += rows[i].sum_sq_float;
        if (rows[i].max_abs > max_total) max_total = rows[i].max_abs;
    }

    printf("AUDIO_REG_METRICS_CONT,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
           n_total, sum_sq_err_total, sum_sq_float_total, max_total);
    printf("AUDIO_REG_METRICS_META,checks=%d,global_max_abs=%.9g\n", checks, global_max_abs);
    for (int i = 0; i < 4; i++) {
        printf("AUDIO_REG_KERNEL_CONT,feature=%s,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
               rows[i].name, rows[i].n, rows[i].sum_sq_err, rows[i].sum_sq_float, rows[i].max_abs);
    }
}

static int run_isolated_suite(void)
{
    metric_acc_t rows[4];
    for (int i = 0; i < 4; i++) {
        rows[i].name = k_feat_name[i];
        rows[i].n = 0;
        rows[i].sum_sq_err = 0.0;
        rows[i].sum_sq_float = 0.0;
        rows[i].max_abs = 0.0f;
    }

    float *x = (float *)malloc(NFFT * sizeof(float));
    float *mags = (float *)malloc(FFT_LEN * sizeof(float));
    float *freqs = (float *)malloc(FFT_LEN * sizeof(float));
    if (!x || !mags || !freqs) {
        fprintf(stderr, "Allocation failed.\n");
        free(x);
        free(mags);
        free(freqs);
        return 1;
    }

    int8_t selector[Number_AUDIO_Features];
    memset(selector, 0, sizeof(selector));
    for (int i = 0; i < 4; i++) selector[k_feat_idx[i]] = 1;

    float global_max_abs = 0.0f;
    int checks = 0;
    for (int s = 0; s < N_SIGNALS; s++) {
        build_signal(s, x);

        float sum_mags = 0.0f;
        compute_rfft(x, NFFT, FS_AUDIO, mags, freqs, &sum_mags);

        float ref[4];
        ref[0] = compute_rolloff(mags, freqs, FFT_LEN, sum_mags);
        ref[1] = compute_centroid(mags, freqs, FFT_LEN, sum_mags);
        ref[2] = compute_spread(mags, freqs, FFT_LEN, sum_mags, ref[1]);
        ref[3] = compute_kurt(mags, freqs, FFT_LEN, sum_mags, ref[1], ref[2]);

        float feats[Number_AUDIO_Features];
        memset(feats, 0, sizeof(feats));
        fxp_audio_fft_features_hybrid(selector, mags, FFT_LEN, FS_AUDIO, NFFT, feats);

        for (int i = 0; i < 4; i++) {
            float fxp_val = feats[k_feat_idx[i]];
            _acc_update(&rows[i], fxp_val, ref[i]);
            float abs_err = fabsf(fxp_val - ref[i]);
            if (abs_err > global_max_abs) global_max_abs = abs_err;
            checks++;
        }
    }

    printf("\nFFT Audio FxP Kernel Regression (isolated suite)\n");
    printf("================================================\n");
    printf("%-22s %8s %14s %14s\n", "Kernel", "N", "RMSE", "RelRMSE(%)");
    printf("-----------------------------------------------------------\n");
    for (int i = 0; i < 4; i++) {
        double rmse = (rows[i].n > 0) ? sqrt(rows[i].sum_sq_err / (double)rows[i].n) : 0.0;
        double baseline_rms = (rows[i].n > 0 && rows[i].sum_sq_float > 0.0)
                            ? sqrt(rows[i].sum_sq_float / (double)rows[i].n) : 0.0;
        double rel = (baseline_rms > 0.0) ? (100.0 * rmse / baseline_rms) : 0.0;
        printf("%-22s %8d %14.6f %14.6f\n", rows[i].name, rows[i].n, rmse, rel);
    }

    _print_machine_rows(rows, checks, global_max_abs);

    free(x);
    free(mags);
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

    metric_acc_t rows[4];
    for (int i = 0; i < 4; i++) {
        rows[i].name = k_feat_name[i];
        rows[i].n = 0;
        rows[i].sum_sq_err = 0.0;
        rows[i].sum_sq_float = 0.0;
        rows[i].max_abs = 0.0f;
    }

    int8_t selector[Number_AUDIO_Features];
    memset(selector, 0, sizeof(selector));
    for (int i = 0; i < 4; i++) selector[k_feat_idx[i]] = 1;

    float *mags = (float *)malloc(FFT_LEN * sizeof(float));
    float *freqs = (float *)malloc(FFT_LEN * sizeof(float));
    if (!mags || !freqs) {
        fprintf(stderr, "Allocation failed.\n");
        free(mags);
        free(freqs);
        return 1;
    }

    int n_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    float global_max_abs = 0.0f;
    int checks = 0;

    for (int w = 0; w < n_wins; w++) {
        int start = w * AUDIO_STEP;
        const float *sig = &audio_in.air[start];

        float sum_mags = 0.0f;
        compute_rfft(sig, WINDOW_SAMP_AUDIO, AUDIO_FS, mags, freqs, &sum_mags);
        if (sum_mags <= 0.0f) continue;

        float ref[4];
        ref[0] = compute_rolloff(mags, freqs, FFT_LEN, sum_mags);
        ref[1] = compute_centroid(mags, freqs, FFT_LEN, sum_mags);
        ref[2] = compute_spread(mags, freqs, FFT_LEN, sum_mags, ref[1]);
        ref[3] = compute_kurt(mags, freqs, FFT_LEN, sum_mags, ref[1], ref[2]);

        float feats[Number_AUDIO_Features];
        memset(feats, 0, sizeof(feats));
        fxp_audio_fft_features_hybrid(selector, mags, FFT_LEN, AUDIO_FS, WINDOW_SAMP_AUDIO, feats);

        for (int i = 0; i < 4; i++) {
            float fxp_val = feats[k_feat_idx[i]];
            _acc_update(&rows[i], fxp_val, ref[i]);
            float abs_err = fabsf(fxp_val - ref[i]);
            if (abs_err > global_max_abs) global_max_abs = abs_err;
            checks++;
        }
    }

    _print_machine_rows(rows, checks, global_max_abs);
    free(mags);
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

