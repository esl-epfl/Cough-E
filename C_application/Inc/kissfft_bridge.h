#ifndef _KISSFFT_BRIDGE_H_
#define _KISSFFT_BRIDGE_H_

#include <inttypes.h>
#include <kiss_fftr.h>

/*
 * Bridge utilities to run KissFFT in FIXED_POINT mode while keeping float
 * boundaries in the surrounding audio pipeline.
 */

#ifdef FIXED_POINT

/*
 * Converts float input into kiss_fft_scalar with frame-wise normalisation.
 * - signal_scale_out is >= 1.0 and captures the pre-normalisation peak.
 * - The output can be passed directly to kiss_fftr().
 */
void kissfft_bridge_convert_input(const float *in, int16_t len, kiss_fft_scalar *out, float *signal_scale_out);

/*
 * Converts KissFFT complex output back to float and applies gain compensation
 * so the result is comparable with float-mode kiss_fftr.
 */
void kissfft_bridge_spectrum_to_float(const kiss_fft_cpx *in, int16_t nfft, float signal_scale, float *re, float *im);

#endif

#endif

