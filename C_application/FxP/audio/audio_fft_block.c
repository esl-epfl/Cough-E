#include <stdlib.h>
#include <math.h>

#include <audio_features.h>
#include <kiss_fftr.h>

#include <audio/audio_fft_block.h>
#include <audio/audio_fft_bridge.h>
#include <audio/audio_fft_kernels.h>
#include <core/fxp_convert.h>
#include <core/fxp_sat.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

static void _fxp_audio_fft_write_features(const int8_t *features_selector,
                                          const fxp_audio_fft_view_t *view,
                                          float *feats)
{
    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];

    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    if (need_rolloff) {
        uq12_20_t rolloff_q20 = fxp_audio_fft_rolloff_q20(view);
        feats[SPECTRAL_ROLLOFF] = fxp_audio_fft_bridge_rolloff_to_float(rolloff_q20);
    }

    uq10_21_t centroid_q21 = 0;
    if (need_centroid) {
        centroid_q21 = fxp_audio_fft_centroid_q21(view);
        if (features_selector[SPECTRAL_CENTROID]) {
            feats[SPECTRAL_CENTROID] = fxp_audio_fft_bridge_centroid_to_float(centroid_q21);
        }
    }

    uq11_5_t spread_q5 = 0;
    if (need_spread) {
        spread_q5 = fxp_audio_fft_spread_q5(view, centroid_q21);
        if (features_selector[SPECTRAL_SPREAD]) {
            feats[SPECTRAL_SPREAD] = fxp_audio_fft_bridge_spread_to_float(spread_q5);
        }
    }

    if (need_kurt) {
        uq7_15_t kurt_q15 = fxp_audio_fft_kurtosis_q15(view, centroid_q21, spread_q5);
        feats[SPECTRAL_KURTOSIS] = fxp_audio_fft_bridge_kurtosis_to_float(kurt_q15);
    }
}

void fxp_audio_fft_features_from_signal(const int8_t *features_selector,
                                        const float *sig,
                                        int16_t len,
                                        int16_t fs,
                                        float *feats)
{
    if (!features_selector || !sig || !feats || len <= 0 || fs <= 0) return;

    int need_rolloff = features_selector[SPECTRAL_ROLLOFF];
    int need_centroid = features_selector[SPECTRAL_CENTROID]
                     || features_selector[SPECTRAL_SPREAD]
                     || features_selector[SPECTRAL_KURTOSIS];
    int need_spread = features_selector[SPECTRAL_SPREAD]
                   || features_selector[SPECTRAL_KURTOSIS];
    int need_kurt = features_selector[SPECTRAL_KURTOSIS];
    if (!need_rolloff && !need_centroid && !need_spread && !need_kurt) return;

    int16_t fft_len = (int16_t)((len / 2) + 1);
    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    if (!cfg) return;

    kiss_fft_scalar *sig_q = (kiss_fft_scalar *)malloc((size_t)len * sizeof(kiss_fft_scalar));
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *)malloc((size_t)fft_len * sizeof(kiss_fft_cpx));
    uq12_20_t *mags_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uq12_20_t *freqs_q20 = (uq12_20_t *)malloc((size_t)fft_len * sizeof(uq12_20_t));
    uint32_t *mag_raw = (uint32_t *)malloc((size_t)fft_len * sizeof(uint32_t));

    if (!sig_q || !cx_out || !mags_q20 || !freqs_q20 || !mag_raw) {
        free(sig_q);
        free(cx_out);
        free(mags_q20);
        free(freqs_q20);
        free(mag_raw);
        free(cfg);
        return;
    }

    float max_abs = 0.0f;
    for (int16_t i = 0; i < len; i++) {
        float a = fabsf(sig[i]);
        if (a > max_abs) max_abs = a;
    }
    float norm_gain = (max_abs > 0.0f) ? (1.0f / max_abs) : 1.0f;

    for (int16_t i = 0; i < len; i++) {
        sig_q[i] = (kiss_fft_scalar)FXP_AUDIO_FROM_FLOAT(sig[i] * norm_gain);
    }

    kiss_fftr(cfg, sig_q, cx_out);

    uint32_t max_mag = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        int64_t re = (int64_t)cx_out[i].r;
        int64_t im = (int64_t)cx_out[i].i;
        uint64_t re_sq = (uint64_t)(re * re);
        uint64_t im_sq = (uint64_t)(im * im);
        uint64_t mag = fxp_sqrt64(re_sq + im_sq);
        uint32_t mag_u32 = fxp_sat_u32_from_u64(mag);
        mag_raw[i] = mag_u32;
        if (mag_u32 > max_mag) max_mag = mag_u32;
    }
    if (max_mag == 0U) max_mag = 1U;

    uint64_t sum_q17 = 0;
    for (int16_t i = 0; i < fft_len; i++) {
        uint64_t scaled_q20 =
            (((uint64_t)mag_raw[i] << FXP_FRAC_AUDIO_FFT_MAGNITUDES) + ((uint64_t)max_mag >> 1U)) / (uint64_t)max_mag;
        mags_q20[i] = fxp_sat_u32_from_u64(scaled_q20);
        sum_q17 += (uint64_t)(mags_q20[i] >> 3);

        uint64_t freq_q20 =
            (((uint64_t)i * (uint64_t)fs) << FXP_FRAC_AUDIO_FFT_FREQUENCIES) / (uint64_t)len;
        freqs_q20[i] = fxp_sat_u32_from_u64(freq_q20);
    }

    fxp_audio_fft_view_t view = {
        .mags_q20 = mags_q20,
        .freqs_q20 = freqs_q20,
        .len = fft_len,
        .sum_mags_q17 = fxp_sat_u32_from_u64(sum_q17),
    };

    _fxp_audio_fft_write_features(features_selector, &view, feats);

    free(sig_q);
    free(cx_out);
    free(mags_q20);
    free(freqs_q20);
    free(mag_raw);
    free(cfg);
}

void fxp_audio_fft_features_hybrid(const int8_t *features_selector,
                                   const float *magnitudes,
                                   int16_t fft_len,
                                   int16_t fs,
                                   int16_t nfft,
                                   float *feats)
{
    if (!features_selector || !magnitudes || !feats || fft_len <= 0 || nfft <= 0) return;

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

    _fxp_audio_fft_write_features(features_selector, &view, feats);

    free(mags_q20);
    free(freqs_q20);
}

#endif
