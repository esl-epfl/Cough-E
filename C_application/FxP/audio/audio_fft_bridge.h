#pragma once

#include <inttypes.h>

#include <core/fxp_convert.h>
#include <core/fxp_qformats.h>
#include <core/fxp_sat.h>
#include <core/fxp_types.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_fft_bridge_from_float(const float *mags, int16_t fft_len, int16_t fs, int16_t nfft,
                                     uq12_20_t *mags_q20, uq12_20_t *freqs_q20, uq15_17_t *sum_mags_q17);

float fxp_audio_fft_bridge_rolloff_to_float(uq12_20_t rolloff_q20);
float fxp_audio_fft_bridge_centroid_to_float(uq10_21_t centroid_q21);
float fxp_audio_fft_bridge_spread_to_float(uq11_5_t spread_q5);
float fxp_audio_fft_bridge_kurtosis_to_float(uq7_15_t kurt_q15);

#endif
