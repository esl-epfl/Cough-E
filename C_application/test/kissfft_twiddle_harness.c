#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kiss_fftr.h"

#if defined(FIXED_POINT) && (FIXED_POINT == 32)
#define KISS_SCALAR_SCALE 2147483647.0
#elif defined(FIXED_POINT)
#define KISS_SCALAR_SCALE 32767.0
#endif

static kiss_fft_scalar scalar_from_float(float x)
{
#if defined(FIXED_POINT)
    double y = (double)x * KISS_SCALAR_SCALE;
    if (y > KISS_SCALAR_SCALE) y = KISS_SCALAR_SCALE;
    if (y < -KISS_SCALAR_SCALE) y = -KISS_SCALAR_SCALE;
    long long q = llround(y);
    return (kiss_fft_scalar)q;
#else
    return x;
#endif
}

static double scalar_to_double(kiss_fft_scalar x)
{
#if defined(FIXED_POINT)
    return (double)x / KISS_SCALAR_SCALE;
#else
    return (double)x;
#endif
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <nfft> <input_txt>\n", argv[0]);
        return 2;
    }

    int nfft = atoi(argv[1]);
    if (nfft <= 0 || (nfft % 2) != 0) {
        fprintf(stderr, "nfft must be a positive even number\n");
        return 2;
    }

    FILE *fp = fopen(argv[2], "r");
    if (!fp) {
        perror("fopen");
        return 1;
    }

    kiss_fft_scalar *time_data = (kiss_fft_scalar *)malloc((size_t)nfft * sizeof(*time_data));
    kiss_fft_cpx *freq_data = (kiss_fft_cpx *)malloc((size_t)(nfft / 2 + 1) * sizeof(*freq_data));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);

    if (!time_data || !freq_data || !cfg) {
        fprintf(stderr, "Allocation failed (nfft=%d)\n", nfft);
        fclose(fp);
        free(time_data);
        free(freq_data);
        free(cfg);
        return 1;
    }

    for (int i = 0; i < nfft; i++) {
        float x = 0.0f;
        if (fscanf(fp, "%f", &x) != 1) {
            fprintf(stderr, "Input file has fewer than nfft=%d samples\n", nfft);
            fclose(fp);
            free(time_data);
            free(freq_data);
            free(cfg);
            return 1;
        }
        time_data[i] = scalar_from_float(x);
    }
    fclose(fp);

    kiss_fftr(cfg, time_data, freq_data);

    for (int k = 0; k <= nfft / 2; k++) {
        double re = scalar_to_double(freq_data[k].r);
        double im = scalar_to_double(freq_data[k].i);
        printf("BIN,%d,%.17g,%.17g\n", k, re, im);
    }

    free(time_data);
    free(freq_data);
    free(cfg);
    return 0;
}

