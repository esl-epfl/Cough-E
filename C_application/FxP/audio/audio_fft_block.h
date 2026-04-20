#pragma once

#include <inttypes.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_fft_features_from_signal(const int8_t *features_selector,
                                        const float *sig,
                                        int16_t len,
                                        int16_t fs,
                                        float *feats);

void fxp_audio_fft_features_hybrid(const int8_t *features_selector,
                                   const float *magnitudes,
                                   int16_t fft_len,
                                   int16_t fs,
                                   int16_t nfft,
                                   float *feats);

#endif
