#pragma once

#include <inttypes.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_periodogram_features_from_signal(const int8_t *features_selector,
                                                const float *sig,
                                                int16_t sig_len,
                                                int16_t fs,
                                                float *feats);

void fxp_audio_periodogram_features_hybrid(const int8_t *features_selector,
                                           const float *psd,
                                           int16_t psd_len,
                                           int16_t fs,
                                           int16_t sig_len,
                                           float *feats);

#endif
