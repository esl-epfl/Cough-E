#include <audio/audio_fft_bridge.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_fft_bridge_from_float(const float *mags, int16_t fft_len, int16_t fs, int16_t nfft,
                                     uq12_20_t *mags_q20, uq12_20_t *freqs_q20, uq15_17_t *sum_mags_q17)
{
    if (!mags || !mags_q20 || !freqs_q20 || !sum_mags_q17 || fft_len <= 0 || nfft <= 0) {
        if (sum_mags_q17) *sum_mags_q17 = 0;
        return;
    }

    uint64_t sum_q17 = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        uint32_t mag = FXP_FROM_FLOAT_U(mags[i], FXP_FRAC_AUDIO_FFT_MAGNITUDES);
        mags_q20[i] = fxp_sat_u32_from_u64((uint64_t)mag);
        sum_q17 += (uint64_t)(mags_q20[i] >> 3);

        uint64_t freq_q20 = (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)nfft;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    *sum_mags_q17 = fxp_sat_u32_from_u64(sum_q17);
}

float fxp_audio_fft_bridge_rolloff_to_float(uq12_20_t rolloff_q20)
{
    return FXP_TO_FLOAT(rolloff_q20, FXP_FRAC_AUDIO_FFT_FREQUENCIES);
}

float fxp_audio_fft_bridge_centroid_to_float(uq10_21_t centroid_q21)
{
    return FXP_TO_FLOAT(centroid_q21, FXP_FRAC_AUDIO_FFT_CENTROID);
}

float fxp_audio_fft_bridge_spread_to_float(uq11_5_t spread_q5)
{
    return FXP_TO_FLOAT(spread_q5, FXP_FRAC_AUDIO_FFT_SPREAD);
}

float fxp_audio_fft_bridge_kurtosis_to_float(uq7_15_t kurt_q15)
{
    return FXP_TO_FLOAT(kurt_q15, FXP_FRAC_AUDIO_FFT_KURTOSIS);
}

#endif
