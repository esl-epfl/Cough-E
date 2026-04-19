#pragma once

#include <inttypes.h>

#include <core/fxp_convert.h>
#include <core/fxp_math.h>
#include <core/fxp_qformats.h>
#include <core/fxp_types.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

typedef struct {
    const uq12_20_t *mags_q20;
    const uq12_20_t *freqs_q20;
    int16_t len;
    uq15_17_t sum_mags_q17;
} fxp_audio_fft_view_t;

uq12_20_t fxp_audio_fft_rolloff_q20(const fxp_audio_fft_view_t *view);
uq10_21_t fxp_audio_fft_centroid_q21(const fxp_audio_fft_view_t *view);
uq11_5_t fxp_audio_fft_spread_q5(const fxp_audio_fft_view_t *view, uq10_21_t centroid_q21);
uq7_15_t fxp_audio_fft_kurtosis_q15(const fxp_audio_fft_view_t *view, uq10_21_t centroid_q21, uq11_5_t spread_q5);

#endif
