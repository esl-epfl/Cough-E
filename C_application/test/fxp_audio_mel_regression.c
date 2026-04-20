#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <audio_features.h>
#include <frequency_features.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)
#include <audio/audio_mel_block.h>
#endif

#define SIG_LEN 6400
#define N_SIGNALS 5
#define PI_F 3.14159265358979323846f

typedef struct {
    const char *name;
    int n;
    double sum_sq_err;
    double sum_sq_float;
    float max_abs;
} metric_acc_t;

static const int k_feat_base[4] = {
    MEL_FREQUENCY_CEPSTRAL_COEFFICIENT,
    MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC,
    MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC),
    MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC),
};

static const char *k_feat_name[4] = {
    "MEL_MEAN",
    "MEL_STD",
    "MEL_MAX",
    "MEL_ENTROPY",
};

static float _rand_uniform(unsigned *state)
{
    *state = (*state * 1103515245u) + 12345u;
    return ((float)((*state >> 8) & 0x00FFFFFFu) / 8388608.0f) - 1.0f;
}

static void build_signal(int signal_id, float *x)
{
    unsigned rng = 0xC3D2A1u + (unsigned)signal_id * 613u;
    for (int i = 0; i < SIG_LEN; i++) {
        float t = (float)i / (float)SIG_LEN;
        switch (signal_id) {
            case 0:
                x[i] = 0.85f * sinf(2.0f * PI_F * 11.0f * t);
                break;
            case 1:
                x[i] = 0.45f * sinf(2.0f * PI_F * 4.0f * t)
                     + 0.28f * sinf(2.0f * PI_F * 29.0f * t + 0.25f);
                break;
            case 2: {
                float f = 5.0f + (140.0f - 5.0f) * t;
                x[i] = 0.75f * sinf(2.0f * PI_F * f * t);
                break;
            }
            case 3: {
                float env = (t < 0.5f) ? (2.0f * t) : (2.0f * (1.0f - t));
                x[i] = env * (0.85f * sinf(2.0f * PI_F * 18.0f * t));
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

    printf("AUDIO_MEL_REG_METRICS_CONT,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
           n_total, sum_sq_err_total, sum_sq_float_total, max_total);
    printf("AUDIO_MEL_REG_METRICS_META,checks=%d,global_max_abs=%.9g\n", checks, global_max_abs);
    for (int i = 0; i < 4; i++) {
        printf("AUDIO_MEL_REG_KERNEL_CONT,feature=%s,n=%d,sum_sq_err=%.17g,sum_sq_float=%.17g,max_abs=%.9g\n",
               rows[i].name, rows[i].n, rows[i].sum_sq_err, rows[i].sum_sq_float, rows[i].max_abs);
    }
}

static void _build_selector(int8_t *selector)
{
    memset(selector, 0, Number_AUDIO_Features * sizeof(int8_t));
    for (int i = 0; i < N_MFCC; i++) {
        selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = 1;
        selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = 1;
        selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] = 1;
        selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i] = 1;
    }
}

static void _evaluate_signal(const float *sig, int16_t len,
                             metric_acc_t *rows, int *checks, float *global_max_abs)
{
    uint8_t idx_needed[N_MFCC];
    for (uint8_t i = 0; i < N_MFCC; i++) idx_needed[i] = i;

    float ref_mean[N_MFCC];
    float ref_std[N_MFCC];
    float ref_max[N_MFCC];
    float ref_entropy[N_MFCC];

    get_mel_spectrogram_features(sig, len, idx_needed, N_MFCC,
                                 ref_mean, ref_std, ref_max, ref_entropy);

    int8_t selector[Number_AUDIO_Features];
    _build_selector(selector);

    float feats[Number_AUDIO_Features];
    memset(feats, 0, sizeof(feats));
    fxp_audio_mel_features_from_signal(selector, sig, len, feats);

    for (int i = 0; i < N_MFCC; i++) {
        const float ref_vals[4] = {
            ref_mean[i],
            ref_std[i],
            ref_max[i],
            ref_entropy[i],
        };

        for (int k = 0; k < 4; k++) {
            float fxp_val = feats[k_feat_base[k] + i];
            float ref_val = ref_vals[k];
            if (!isfinite(fxp_val) || !isfinite(ref_val)) {
                continue;
            }
            _acc_update(&rows[k], fxp_val, ref_val);
            {
                float abs_err = fabsf(fxp_val - ref_val);
                if (abs_err > *global_max_abs) *global_max_abs = abs_err;
            }
            *checks += 1;
        }
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

    float *x = (float *)malloc(SIG_LEN * sizeof(float));
    if (!x) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    float global_max_abs = 0.0f;
    int checks = 0;
    for (int s = 0; s < N_SIGNALS; s++) {
        build_signal(s, x);
        _evaluate_signal(x, SIG_LEN, rows, &checks, &global_max_abs);
    }

    printf("\nMel Audio FxP Kernel Regression (isolated suite)\n");
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

    int n_wins = ((AUDIO_LEN - WINDOW_SAMP_AUDIO) / AUDIO_STEP) + 1;
    float global_max_abs = 0.0f;
    int checks = 0;

    for (int w = 0; w < n_wins; w++) {
        int start = w * AUDIO_STEP;
        const float *sig = &audio_in.air[start];
        _evaluate_signal(sig, WINDOW_SAMP_AUDIO, rows, &checks, &global_max_abs);
    }

    _print_machine_rows(rows, checks, global_max_abs);
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
