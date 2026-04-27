#pragma once

#include <inttypes.h>

#include <audio_features.h>
#include <core/fxp_core.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void audio_fft_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t len,
                        int16_t fs,
                        fxp_q16_t *feats_q16);

void audio_psd_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t sig_len,
                        int16_t fs,
                        fxp_q16_t *feats_q16);

void audio_mel_features(const int8_t *features_selector,
                        const int16_t *sig_q14,
                        int16_t len,
                        fxp_q16_t *feats_q16);

#endif
