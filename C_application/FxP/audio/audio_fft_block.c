#include <stdlib.h>

#include <audio_features.h>

#include <audio/audio_fft_block.h>
#include <audio/audio_fft_bridge.h>
#include <audio/audio_fft_kernels.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_fft_features_hybrid(const int8_t *features_selector,
                                   const float *magnitudes,
                                   int16_t fft_len,
                                   int16_t fs,
                                   int16_t nfft,
                                   float *feats)
{
    if (!features_selector || !magnitudes || !feats || fft_len <= 0 || nfft <= 0) return;

    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];

    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    uq12_20_t *mags_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    if (!mags_q20 || !freqs_q20) {
        free(mags_q20);
        free(freqs_q20);
        return;
    }

    uq15_17_t sum_mags_q17 = 0;
    fxp_audio_fft_bridge_from_float(magnitudes, fft_len, fs, nfft, mags_q20, freqs_q20, &sum_mags_q17);

    fxp_audio_fft_view_t view = {
        .mags_q20 = mags_q20,
        .freqs_q20 = freqs_q20,
        .len = fft_len,
        .sum_mags_q17 = sum_mags_q17,
    };

    if (need_rolloff) {
        uq12_20_t rolloff_q20 = fxp_audio_fft_rolloff_q20(&view);
        feats[SPECTRAL_ROLLOFF] = fxp_audio_fft_bridge_rolloff_to_float(rolloff_q20);
    }

    uq10_21_t centroid_q21 = 0;
    if (need_centroid) {
        centroid_q21 = fxp_audio_fft_centroid_q21(&view);
        if (features_selector[SPECTRAL_CENTROID]) {
            feats[SPECTRAL_CENTROID] = fxp_audio_fft_bridge_centroid_to_float(centroid_q21);
        }
    }

    uq11_5_t spread_q5 = 0;
    if (need_spread) {
        spread_q5 = fxp_audio_fft_spread_q5(&view, centroid_q21);
        if (features_selector[SPECTRAL_SPREAD]) {
            feats[SPECTRAL_SPREAD] = fxp_audio_fft_bridge_spread_to_float(spread_q5);
        }
    }

    if (need_kurt) {
        uq7_15_t kurt_q15 = fxp_audio_fft_kurtosis_q15(&view, centroid_q21, spread_q5);
        feats[SPECTRAL_KURTOSIS] = fxp_audio_fft_bridge_kurtosis_to_float(kurt_q15);
    }

    free(mags_q20);
    free(freqs_q20);
}

#endif
