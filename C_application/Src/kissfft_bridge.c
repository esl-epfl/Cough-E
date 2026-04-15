#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <kissfft_bridge.h>

#ifdef FIXED_POINT

#if (FIXED_POINT == 32)
#define KISSFFT_BRIDGE_SCALAR_MAX 2147483647.0f
#else
#define KISSFFT_BRIDGE_SCALAR_MAX 32767.0f
#endif

typedef struct {
    int16_t nfft;
    float gain;
    int8_t valid;
} kissfft_gain_cache_t;

#define KISSFFT_GAIN_CACHE_SIZE 8
static kissfft_gain_cache_t _gain_cache[KISSFFT_GAIN_CACHE_SIZE];

static inline kiss_fft_scalar _float_to_scalar(float x)
{
    float y = x * KISSFFT_BRIDGE_SCALAR_MAX;
    if (y > KISSFFT_BRIDGE_SCALAR_MAX) y = KISSFFT_BRIDGE_SCALAR_MAX;
    if (y < -KISSFFT_BRIDGE_SCALAR_MAX) y = -KISSFFT_BRIDGE_SCALAR_MAX;
    return (kiss_fft_scalar)lroundf(y);
}

static inline float _scalar_to_float(kiss_fft_scalar x)
{
    return ((float)x) / KISSFFT_BRIDGE_SCALAR_MAX;
}

static float _compute_gain_for_nfft(int16_t nfft)
{
    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, 0, 0);
    if (!cfg) return 1.0f;

    kiss_fft_scalar *x = (kiss_fft_scalar *)calloc((size_t)nfft, sizeof(kiss_fft_scalar));
    kiss_fft_cpx *y = (kiss_fft_cpx *)malloc(((size_t)nfft / 2U + 1U) * sizeof(kiss_fft_cpx));
    if (!x || !y) {
        free(x);
        free(y);
        free(cfg);
        return 1.0f;
    }

    x[0] = _float_to_scalar(1.0f); /* impulse of amplitude 1.0 */
    kiss_fftr(cfg, x, y);

    float ref = _scalar_to_float(y[1].r);
    if (fabsf(ref) < 1e-12f) {
        ref = _scalar_to_float(y[0].r);
    }
    if (fabsf(ref) < 1e-12f) {
        ref = 1.0f;
    }

    free(x);
    free(y);
    free(cfg);
    return 1.0f / ref;
}

static float _get_gain_for_nfft(int16_t nfft)
{
    for (int i = 0; i < KISSFFT_GAIN_CACHE_SIZE; i++) {
        if (_gain_cache[i].valid && _gain_cache[i].nfft == nfft) {
            return _gain_cache[i].gain;
        }
    }

    float g = _compute_gain_for_nfft(nfft);

    for (int i = 0; i < KISSFFT_GAIN_CACHE_SIZE; i++) {
        if (!_gain_cache[i].valid) {
            _gain_cache[i].nfft = nfft;
            _gain_cache[i].gain = g;
            _gain_cache[i].valid = 1;
            return g;
        }
    }

    /* Simple fallback when cache is full. */
    _gain_cache[0].nfft = nfft;
    _gain_cache[0].gain = g;
    _gain_cache[0].valid = 1;
    return g;
}

void kissfft_bridge_convert_input(const float *in, int16_t len, kiss_fft_scalar *out, float *signal_scale_out)
{
    float max_abs = 0.0f;
    for (int16_t i = 0; i < len; i++) {
        float a = fabsf(in[i]);
        if (a > max_abs) max_abs = a;
    }

    float signal_scale = (max_abs > 1.0f) ? max_abs : 1.0f;
    float inv = 1.0f / signal_scale;
    for (int16_t i = 0; i < len; i++) {
        out[i] = _float_to_scalar(in[i] * inv);
    }

    *signal_scale_out = signal_scale;
}

void kissfft_bridge_spectrum_to_float(const kiss_fft_cpx *in, int16_t nfft, float signal_scale, float *re, float *im)
{
    float gain = _get_gain_for_nfft(nfft) * signal_scale;
    int16_t n_bins = (int16_t)(nfft / 2) + 1;
    for (int16_t i = 0; i < n_bins; i++) {
        re[i] = _scalar_to_float(in[i].r) * gain;
        im[i] = _scalar_to_float(in[i].i) * gain;
    }
}

#endif

