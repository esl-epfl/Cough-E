#pragma once

#include <inttypes.h>

#include <core/fxp_convert.h>
#include <core/fxp_qformats.h>
#include <core/fxp_sat.h>
#include <core/fxp_types.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void fxp_audio_psd_bridge_from_float(const float *psd,
                                     int16_t psd_len,
                                     int16_t fs,
                                     int16_t sig_len,
                                     uq18_14_t *proxy_q14,
                                     uq12_20_t *freqs_q20);

float fxp_audio_psd_bridge_domfreq_to_float(uq12_20_t dom_freq_q20);
float fxp_audio_psd_bridge_flatness_to_float(uq0_16_t flatness_q16);
float fxp_audio_psd_bridge_bandpower_to_float(uq0_16_t bandpower_q16);

#endif
