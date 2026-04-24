#pragma once

#include <inttypes.h>

#include <core/cough_backend.h>
#include <audio_features.h>

#include <core/fxp_core.h>
#include <core/fxp_convert.h>
#include <core/fxp_qformats.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

typedef struct {
    const uq12_20_t *mags_q20;
    const uq12_20_t *freqs_q20;
    int16_t len;
    uq15_17_t sum_mags_q17;
} fxp_audio_fft_view_t;

typedef struct {
    const uq18_14_t *proxy_q14;
    const int32_t *log_proxy_q11;
    const uq12_20_t *freqs_q20;
    int16_t len;
} fxp_audio_psd_view_t;

void fxp_audio_fft_features_from_q14(const int8_t *features_selector,
                                     const int16_t *sig_q14,
                                     int16_t len,
                                     int16_t fs,
                                     fxp_q16_t *feats_q16);

void fxp_audio_periodogram_features_from_q14(const int8_t *features_selector,
                                             const int16_t *sig_q14,
                                             int16_t sig_len,
                                             int16_t fs,
                                             fxp_q16_t *feats_q16);

void fxp_audio_mel_features_from_q14(const int8_t *features_selector,
                                     const int16_t *sig_q14,
                                     int16_t len,
                                     fxp_q16_t *feats_q16);

#endif
