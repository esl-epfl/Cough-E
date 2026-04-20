#pragma once

#include <inttypes.h>

#include <audio_features.h>

#include <core/fxp_convert.h>
#include <core/fxp_math.h>
#include <core/fxp_qformats.h>
#include <core/fxp_sat.h>
#include <core/fxp_types.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

typedef struct {
    const uq18_14_t *proxy_q14;   /* normalized linear PSD proxy, mean ~ 1.0 */
    const int32_t   *log_proxy_q11; /* ln(proxy[i]) in Q11, higher dynamic range than q5_11_t */
    const uq12_20_t *freqs_q20;
    int16_t len;
} fxp_audio_psd_view_t;

uq12_20_t fxp_audio_psd_dominant_freq_q20(const fxp_audio_psd_view_t *view);
uq0_16_t fxp_audio_psd_flatness_q16(const fxp_audio_psd_view_t *view);
void fxp_audio_psd_bandpowers_q16(const fxp_audio_psd_view_t *view,
                                  const int8_t *psd_selector,
                                  uq0_16_t *band_powers_q16);

#endif
