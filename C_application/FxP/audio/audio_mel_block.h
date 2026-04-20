#pragma once

#include <inttypes.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_mel_features_from_signal(const int8_t *features_selector,
                                        const float *sig,
                                        int16_t len,
                                        float *feats);

#endif
