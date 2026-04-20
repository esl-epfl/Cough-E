#include <math.h>

#include <welch_psd.h>

#include <audio/audio_periodogram_bridge.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

static float _fxp_audio_psd_hann_energy(void)
{
    static int initialized = 0;
    static float sum_sq = 0.0f;
    if (!initialized) {
        for (int16_t i = 0; i < NPERSEG; i++) {
            sum_sq += hann_window[i] * hann_window[i];
        }
        initialized = 1;
    }
    return sum_sq;
}

void fxp_audio_psd_bridge_from_float(const float *psd,
                                     int16_t psd_len,
                                     int16_t fs,
                                     int16_t sig_len,
                                     uq18_14_t *proxy_q14,
                                     uq12_20_t *freqs_q20)
{
    if (!psd || !proxy_q14 || !freqs_q20 || psd_len <= 0 || fs <= 0 || sig_len <= 0) return;

    int16_t hop = (NPERSEG - NOVERLAP);
    int16_t steps = (hop > 0 && sig_len > NOVERLAP) ? (int16_t)((sig_len - NOVERLAP) / hop) : 0;
    if (steps <= 0) steps = 1;

    float hann_energy = _fxp_audio_psd_hann_energy();
    if (hann_energy <= 0.0f) hann_energy = 1.0f;
    float scale = 1.0f / ((float)fs * hann_energy);

    for (int16_t i = 0; i < psd_len; i++) {
        float denom = scale;
        float proxy = 0.0f;
        if (denom > 0.0f && psd[i] > 0.0f) {
            proxy = (psd[i] * (float)steps) / denom;
        }
        uint32_t proxy_u32 = FXP_FROM_FLOAT_U(proxy, FXP_FRAC_AUDIO_PSD_PROXY);
        proxy_q14[i] = fxp_sat_u32_from_u64((uint64_t)proxy_u32);

        uint64_t freq_q20 = (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)NPERSEG;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }
}

float fxp_audio_psd_bridge_domfreq_to_float(uq12_20_t dom_freq_q20)
{
    return FXP_TO_FLOAT(dom_freq_q20, FXP_FRAC_AUDIO_FFT_FREQUENCIES);
}

float fxp_audio_psd_bridge_flatness_to_float(uq0_16_t flatness_q16)
{
    return FXP_TO_FLOAT(flatness_q16, FXP_FRAC_AUDIO_PSD_FLATNESS);
}

float fxp_audio_psd_bridge_bandpower_to_float(uq0_16_t bandpower_q16)
{
    return FXP_TO_FLOAT(bandpower_q16, FXP_FRAC_AUDIO_PSD_BANDPOWER);
}

#endif
