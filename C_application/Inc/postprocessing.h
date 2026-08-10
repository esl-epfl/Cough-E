#ifndef _POSTPROCESSING_H_
#define _POSTPROCESSING_H_

#include <inttypes.h>
#include <types.h>
#include <helpers.h>

// Maximum number of peaks expected between one output and the next one
#define MAX_PEAKS_EXPECTED      50

// Downsample frequency of the postprocessing module
#define FS_DOWNSAMPLE           2000

// Set a maximum number of peaks to be found in a single segment
// This serves to inizialize a buffer where to store their locations
#define MAX_PEAKS_SEG           50

// Constants related to cough physiology
#ifndef FXP_MODE
#define COUGH_END_TOLERANCE     0.01f
#define COUGH_BURST_MIN_DUR                 0.03f
#define COUGH_BURST_MAX_DUR                 0.05f
#define COUGH_EXP_MIN_DUR                   0.2f
#define COUGH_EXP_MAX_DUR                   0.5f
#define COMPRESSIVE_PHASE_DUR               0.2f
#define COUGH_LEN_IN_SERIES_DECREASE_FACTOR 0.8f
#else
/* Integer equivalents used by FxP postprocessing runtime. */
#define COUGH_END_TOLERANCE_MS              10U
#define COUGH_BURST_MIN_MS                  30U
#define COUGH_BURST_MAX_MS                  50U
#define COUGH_EXP_MIN_MS                    200U
#define COUGH_EXP_MAX_MS                    500U
#define COMPRESSIVE_PHASE_MS                200U
#define COUGH_LEN_IN_SERIES_DECREASE_FACTOR_FXP 26214U /* 0.8 in signed 1.15 fixed-point */
#endif

#ifdef FXP_MODE
typedef int16_t postproc_sample_t;
typedef int16_t postproc_peak_t;
#else
typedef audio_data_t postproc_sample_t;
typedef audio_data_t postproc_peak_t;
#endif


uint16_t _clean_cough_segments(uint16_t *starts_idxs, uint16_t *ends_idxs, uint16_t *peaks_locs, postproc_peak_t *peaks, uint16_t n_peaks, uint16_t fs);
void _get_cough_peaks(const postproc_sample_t *seg, int16_t len, int16_t fs, uint16_t *starts, uint16_t *ends, uint16_t *peaks_locs, postproc_peak_t *peaks_amps, uint16_t *new_added);


#endif
