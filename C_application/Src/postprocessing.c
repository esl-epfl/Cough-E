#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <postprocessing.h>
#include <range_analysis.h>

#ifdef FXP_MODE
#include <core/fxp_core.h>
#endif

/**
 * Checks if a specific element is contained in the given array.
 * If yes, it returns 1, 0 otherwise.
 */
uint8_t _contains(uint16_t *arr, uint16_t len, uint16_t elem)
{
    for (uint16_t i = 0; i < len; i++) {
        if (arr[i] == elem) {
            return 1;
        }
    }
    return 0;
}

#ifdef FXP_MODE

/*
 * Downsamples and normalizes (zero-mean and max-abs scaling) in fixed-point.
 * Output is signed 1.15 fixed-point.
 */
static int16_t *_downsample(const int16_t *sig, int16_t len, int16_t fs, int16_t *new_len)
{
    int16_t scale_factor = (int16_t)(fs / FS_DOWNSAMPLE);
    if (scale_factor <= 0) scale_factor = 1;

    *new_len = (int16_t)(len / scale_factor);
    if (*new_len <= 0) {
        *new_len = 1;
    }

    int16_t *res = (int16_t *)malloc((size_t)(*new_len) * sizeof(int16_t));
    if (!res) return NULL;

    int64_t mean_acc = 0;
    for (int16_t i = 0; i < *new_len; i++) {
        int16_t v = sig[i * scale_factor];
        res[i] = v;
        mean_acc += (int64_t)v;
    }

    int64_t mean_bias = (int64_t)(*new_len / 2);
    if (mean_acc < 0) mean_bias = -mean_bias;
    int32_t mean_sample = (int32_t)((mean_acc + mean_bias) / (int64_t)(*new_len));

    int32_t max_abs = 0;
    for (int16_t i = 0; i < *new_len; i++) {
        int32_t centered = (int32_t)res[i] - mean_sample;
        int32_t abs_v = (centered >= 0) ? centered : -centered;
        if (abs_v > max_abs) max_abs = abs_v;
    }

    if (max_abs <= 0) {
        memset(res, 0, (size_t)(*new_len) * sizeof(int16_t));
        return res;
    }

    for (int16_t i = 0; i < *new_len; i++) {
        int32_t centered = (int32_t)res[i] - mean_sample;
        int64_t scaled = ((int64_t)centered) << 15;
        int64_t scaled_bias = (int64_t)(max_abs / 2);
        if (scaled < 0) scaled_bias = -scaled_bias;
        int32_t normalized = (int32_t)((scaled + scaled_bias) / (int64_t)max_abs);
        res[i] = (int16_t)normalized;
    }

    return res;
}

void _get_cough_peaks(const int16_t *seg,
                      int16_t len,
                      int16_t fs,
                      uint16_t *starts,
                      uint16_t *ends,
                      uint16_t *peaks_locs,
                      int16_t *peaks_amps,
                      uint16_t *new_added)
{
    int16_t downsample_len = 0;
    int16_t *downsample_seg = _downsample(seg, len, fs, &downsample_len);
    if (!downsample_seg || downsample_len <= 0) {
        free(downsample_seg);
        *new_added = 0;
        return;
    }

    uint64_t square_sum = 0;
    uint32_t peak_square = 0;
    int16_t fallback_peak_idx = 0;

    /* downsample_seg is signed 1.15 fixed-point; squared values are scaled by 30 bits. */
    for (int16_t i = 0; i < downsample_len; i++) {
        int32_t v = downsample_seg[i];
        uint32_t square = (uint32_t)((int64_t)v * (int64_t)v);
        square_sum += (uint64_t)square;
        if (square > peak_square) {
            peak_square = square;
            fallback_peak_idx = i;
        }
    }

    uint64_t mean_square = (square_sum + ((uint64_t)downsample_len >> 1U)) / (uint64_t)downsample_len;
    uint32_t rms_level = (uint32_t)fxp_sqrt64(mean_square);
    uint32_t low_threshold = rms_level << 15;
    uint32_t high_threshold = (peak_square + (3U * low_threshold) + 2U) >> 2;

    int16_t cough_start = 0;
    int16_t cough_end = 0;
    uint16_t below_th_counter = 0;
    uint16_t tolerance = (uint16_t)(((uint32_t)COUGH_END_TOLERANCE_MS * (uint32_t)FS_DOWNSAMPLE + 500U) / 1000U);

    int16_t scale_freq = (int16_t)(fs / FS_DOWNSAMPLE);
    if (scale_freq <= 0) scale_freq = 1;

    int8_t cough_in_progress = 0;
    int16_t segment_peak_idx = 0;
    int16_t segment_peak_amp = 0;
    uint16_t peaks_found = 0;

    for (int16_t i = 0; i < downsample_len; i++) {
        int32_t sample = downsample_seg[i];
        uint32_t sample_square = (uint32_t)((int64_t)sample * (int64_t)sample);

        if (cough_in_progress) {
            if (sample > segment_peak_amp) {
                segment_peak_amp = (int16_t)sample;
                segment_peak_idx = i;
            }

            if (i == (downsample_len - 1)) {
                cough_end = i;
                cough_in_progress = 0;

                starts[peaks_found] = (uint16_t)((uint32_t)cough_start * (uint32_t)scale_freq);
                ends[peaks_found] = (uint16_t)((uint32_t)cough_end * (uint32_t)scale_freq);

                peaks_locs[peaks_found] = (uint16_t)((uint32_t)segment_peak_idx * (uint32_t)scale_freq);
                peaks_amps[peaks_found] = segment_peak_amp;

                peaks_found++;
            } else if (sample_square < low_threshold) {
                below_th_counter++;

                if (below_th_counter > tolerance) {
                    cough_end = i;
                    cough_in_progress = 0;

                    starts[peaks_found] = (uint16_t)((uint32_t)cough_start * (uint32_t)scale_freq);
                    ends[peaks_found] = (uint16_t)((uint32_t)cough_end * (uint32_t)scale_freq);

                    peaks_locs[peaks_found] = (uint16_t)((uint32_t)segment_peak_idx * (uint32_t)scale_freq);
                    peaks_amps[peaks_found] = segment_peak_amp;

                    peaks_found++;
                }
            } else {
                below_th_counter = 0;
            }
        } else {
            if (sample_square > high_threshold) {
                cough_start = i;
                segment_peak_idx = i;
                segment_peak_amp = (int16_t)sample;
                cough_in_progress = 1;
                below_th_counter = 0;
            }
        }
    }

    if (peaks_found == 0) {
        int16_t peak_ds_idx = fallback_peak_idx;
        peaks_locs[0] = (uint16_t)((uint32_t)peak_ds_idx * (uint32_t)scale_freq);
        peaks_amps[0] = downsample_seg[peak_ds_idx];

        if (peaks_locs[0] > 0U) {
            starts[0] = peaks_locs[0] - 1U;
        } else {
            starts[0] = 0U;
        }
        ends[0] = peaks_locs[0] + 1U;
    }

    free(downsample_seg);

    *new_added = peaks_found;
}

static void _sort_segments_by_peak(uint16_t *starts,
                                   uint16_t *ends,
                                   uint16_t *peaks_locs,
                                   int16_t *peaks,
                                   uint16_t n)
{
    for (uint16_t i = 1; i < n; i++) {
        uint16_t key_loc = peaks_locs[i];
        uint16_t key_start = starts[i];
        uint16_t key_end = ends[i];
        int16_t key_peak = peaks[i];

        uint16_t j = i;
        while (j > 0 && peaks_locs[j - 1] > key_loc) {
            peaks_locs[j] = peaks_locs[j - 1];
            starts[j] = starts[j - 1];
            ends[j] = ends[j - 1];
            peaks[j] = peaks[j - 1];
            j--;
        }

        peaks_locs[j] = key_loc;
        starts[j] = key_start;
        ends[j] = key_end;
        peaks[j] = key_peak;
    }
}

uint16_t _clean_cough_segments(uint16_t *starts_idxs,
                               uint16_t *ends_idxs,
                               uint16_t *peaks_locs,
                               int16_t *peaks,
                               uint16_t n_peaks,
                               uint16_t fs)
{
    if (n_peaks == 0) return 0;

    uint32_t min_dist_ms = (uint32_t)COUGH_BURST_MIN_MS + (uint32_t)COUGH_EXP_MIN_MS;
    uint32_t min_before_ms = (uint32_t)COUGH_BURST_MIN_MS / 2U;
    uint32_t min_after_ms = min_before_ms + (uint32_t)COUGH_EXP_MIN_MS;
    uint32_t max_burst_dist_ms = (uint32_t)COUGH_EXP_MAX_MS + (uint32_t)COUGH_BURST_MIN_MS;

    uint16_t min_dist_btwn_cough_peaks = (uint16_t)((min_dist_ms * (uint32_t)fs + 500U) / 1000U);
    uint16_t min_time_before_peak = (uint16_t)((min_before_ms * (uint32_t)fs + 500U) / 1000U);
    uint16_t min_time_after_peak = (uint16_t)((min_after_ms * (uint32_t)fs + 500U) / 1000U);
    uint16_t max_dist_btwn_cough_peaks_in_burst = (uint16_t)((max_burst_dist_ms * (uint32_t)fs + 500U) / 1000U);
    uint16_t cough_burst_min_samp = (uint16_t)(((uint32_t)COUGH_BURST_MIN_MS * (uint32_t)fs + 500U) / 1000U);
    uint16_t cough_burst_max_samp = (uint16_t)(((uint32_t)COUGH_BURST_MAX_MS * (uint32_t)fs + 500U) / 1000U);
    uint16_t compressive_phase_samp = (uint16_t)(((uint32_t)COMPRESSIVE_PHASE_MS * (uint32_t)fs + 500U) / 1000U);

    _sort_segments_by_peak(starts_idxs, ends_idxs, peaks_locs, peaks, n_peaks);

    uint8_t *merged = (uint8_t *)calloc((size_t)n_peaks, sizeof(uint8_t));
    uint16_t *locs_final = (uint16_t *)malloc((size_t)n_peaks * sizeof(uint16_t));
    if (!merged || !locs_final) {
        free(merged);
        free(locs_final);
        return 0;
    }

    uint16_t n_peaks_final = 0;

    for (uint16_t i = 0; i < n_peaks; i++) {
        if (merged[i]) continue;

        uint16_t tmp_start = starts_idxs[i];
        uint16_t tmp_end = ends_idxs[i];
        uint16_t tmp_loc = peaks_locs[i];
        int16_t tmp_peak = peaks[i];

        for (uint16_t j = (uint16_t)(i + 1U); j < n_peaks; j++) {
            uint16_t dist_samples = (uint16_t)(peaks_locs[j] - peaks_locs[i]);
            if (dist_samples < min_dist_btwn_cough_peaks) {
                merged[j] = 1U;

                tmp_start = (tmp_start < starts_idxs[j]) ? tmp_start : starts_idxs[j];
                tmp_end = (tmp_end < ends_idxs[j]) ? tmp_end : ends_idxs[j];

                if (peaks[j] > tmp_peak) {
                    tmp_peak = peaks[j];
                    tmp_loc = peaks_locs[j];
                }
            }
        }

        starts_idxs[n_peaks_final] = tmp_start;
        ends_idxs[n_peaks_final] = tmp_end;
        locs_final[n_peaks_final] = tmp_loc;
        n_peaks_final++;
    }

    free(merged);

    uint16_t avg_cough_end_samples = min_time_after_peak;
    if (n_peaks_final > 1U) {
        uint32_t burst_sum = 0;
        uint16_t burst_count = 0;

        for (uint16_t i = 0; i < (uint16_t)(n_peaks_final - 1U); i++) {
            uint16_t d = (uint16_t)(locs_final[i + 1U] - locs_final[i]);
            if (d <= max_dist_btwn_cough_peaks_in_burst) {
                burst_sum += d;
                burst_count++;
            }
        }

        if (burst_count > 0U) {
            uint16_t mean_d = (uint16_t)((burst_sum + (burst_count / 2U)) / burst_count);
            if (mean_d > cough_burst_max_samp) {
                avg_cough_end_samples = (uint16_t)(mean_d - cough_burst_max_samp);
            } else {
                avg_cough_end_samples = 0U;
            }
        }
    }

    uint16_t cough_series_count = 0;

    for (uint16_t i = 0; i < n_peaks_final; i++) {
        uint16_t time_start_peak = (uint16_t)(locs_final[i] - starts_idxs[i]);
        if (time_start_peak < min_time_before_peak) {
            if (locs_final[i] > min_time_before_peak) {
                starts_idxs[i] = (uint16_t)(locs_final[i] - min_time_before_peak);
            } else {
                starts_idxs[i] = 0U;
            }
        }

        uint16_t time_to_next_peak;
        if (i < (uint16_t)(n_peaks_final - 1U)) {
            time_to_next_peak = (uint16_t)(locs_final[i + 1U] - locs_final[i]);
        } else {
            time_to_next_peak = UINT16_MAX;
        }

        if (time_to_next_peak > max_dist_btwn_cough_peaks_in_burst) {
            uint32_t series_multiplier = (1U << 15);
            if (cough_series_count > 0U) {
                series_multiplier = COUGH_LEN_IN_SERIES_DECREASE_FACTOR_FXP;
                for (uint16_t j = 0; j < (uint16_t)(cough_series_count - 1U); j++) {
                    series_multiplier = (uint32_t)((((uint64_t)series_multiplier * (uint64_t)series_multiplier) + (1U << 14)) >> 15);
                }
            }

            uint16_t extra = (uint16_t)((((uint64_t)series_multiplier * (uint64_t)avg_cough_end_samples) + (1U << 14)) >> 15);
            ends_idxs[i] = (uint16_t)((uint32_t)locs_final[i] + (uint32_t)extra);
            cough_series_count = 0U;
        } else {
            cough_series_count++;

            if (i < (uint16_t)(n_peaks_final - 1U)) {
                uint16_t delta = (time_to_next_peak > compressive_phase_samp)
                    ? (uint16_t)(time_to_next_peak - compressive_phase_samp)
                    : 0U;

                if (delta < min_dist_btwn_cough_peaks) {
                    ends_idxs[i] = (starts_idxs[i + 1U] > cough_burst_min_samp)
                        ? (uint16_t)(starts_idxs[i + 1U] - cough_burst_min_samp)
                        : 0U;
                } else {
                    uint32_t sub = (uint32_t)compressive_phase_samp + (uint32_t)cough_burst_min_samp;
                    ends_idxs[i] = (locs_final[i + 1U] > sub)
                        ? (uint16_t)(locs_final[i + 1U] - sub)
                        : 0U;
                }
            }
        }
    }

    free(locs_final);
    return n_peaks_final;
}

#else

/**
 * Downsamples a signal arrat according to the `FS_DOWNSAMPLE` parameter.
 * The resulting signal is first downsampled and normalized by subtracting the mean
 * and dividing by the maximum absolute value.
 */
static float *_downsample(const float *sig, int16_t len, int16_t fs, int16_t *new_len)
{
    RA_LOG_ARRAY("POSTPROC", "_downsample", "sig_input", sig, len);

    int8_t scale_factor = fs / FS_DOWNSAMPLE;
    *new_len = len / scale_factor;

    float *res = (float *)malloc(*new_len * sizeof(float));
    float mean = 0.0f;

    for (int16_t i = 0; i < *new_len; i++) {
        res[i] = sig[i * scale_factor];
        mean += res[i];
    }

    mean = mean / *new_len;
    RA_LOG_SCALAR("POSTPROC", "_downsample", "mean", mean);

    sub_constant(res, *new_len, mean, res);
    RA_LOG_ARRAY("POSTPROC", "_downsample", "zero_mean", res, *new_len);

    float max_abs = vect_max_abs_value(res, *new_len);
    RA_LOG_SCALAR("POSTPROC", "_downsample", "max_abs", max_abs);
    vect_div_const(res, *new_len, max_abs, res);
    RA_LOG_ARRAY("POSTPROC", "_downsample", "result", res, *new_len);

    return res;
}

void _get_cough_peaks(const float *seg,
                      int16_t len,
                      int16_t fs,
                      uint16_t *starts,
                      uint16_t *ends,
                      uint16_t *peaks_locs,
                      float *peaks_amps,
                      uint16_t *new_added)
{
    RA_LOG_ARRAY("POSTPROC", "_get_cough_peaks", "seg_input", seg, len);

    int16_t downsample_len = 0.0;
    float *downsample_seg = _downsample(seg, len, fs, &downsample_len);

    float *seg_squared = (float *)malloc(downsample_len * sizeof(float));
    vect_mult(downsample_seg, downsample_seg, downsample_len, seg_squared);
    RA_LOG_ARRAY("POSTPROC", "_get_cough_peaks", "seg_squared", seg_squared, downsample_len);

    float peak = vect_max_value(seg_squared, downsample_len);
    RA_LOG_SCALAR("POSTPROC", "_get_cough_peaks", "peak", peak);

    float th_low = sqrtf(vect_mean(seg_squared, downsample_len));
    float th_high = 0.25f * peak + 0.75f * th_low;
    RA_LOG_SCALAR("POSTPROC", "_get_cough_peaks", "th_low", th_low);
    RA_LOG_SCALAR("POSTPROC", "_get_cough_peaks", "th_high", th_high);

    int16_t cough_start = 0;
    int16_t cough_end = 0;
    int8_t below_th_counter = 0;
    int16_t tolerance = COUGH_END_TOLERANCE * FS_DOWNSAMPLE;

    int16_t scale_freq = (int16_t)(fs / FS_DOWNSAMPLE);

    int8_t cough_in_progress = 0;
    int8_t peaks_found = 0;

    for (int16_t i = 0; i < downsample_len; i++) {
        if (cough_in_progress) {
            if (i == (downsample_len - 1)) {
                cough_end = i;
                cough_in_progress = 0;

                starts[peaks_found] = cough_start * scale_freq;
                ends[peaks_found] = cough_end * scale_freq;

                peaks_locs[peaks_found] = (vect_max_index(&downsample_seg[cough_start], (cough_end - cough_start) + 1) + cough_start) * scale_freq;
                peaks_amps[peaks_found] = downsample_seg[peaks_locs[peaks_found] / scale_freq];

                peaks_found++;

            } else if (seg_squared[i] < th_low) {
                below_th_counter++;

                if (below_th_counter > tolerance) {
                    if (i < downsample_len)
                        cough_end = i;
                    else
                        cough_end = (downsample_len - 1);
                    cough_in_progress = 0;

                    starts[peaks_found] = cough_start * scale_freq;
                    ends[peaks_found] = cough_end * scale_freq;

                    peaks_locs[peaks_found] = (vect_max_index(&downsample_seg[cough_start], (cough_end - cough_start + 1)) + cough_start) * scale_freq;
                    peaks_amps[peaks_found] = downsample_seg[peaks_locs[peaks_found] / scale_freq];

                    peaks_found++;
                }
            } else {
                below_th_counter = 0;
            }
        } else {
            if (seg_squared[i] > th_high) {
                if (i >= 0) {
                    cough_start = i;
                } else {
                    cough_start = 0;
                }

                cough_in_progress = 1;
                below_th_counter = 0;
            }
        }
    }

    if (peaks_found == 0) {
        peaks_locs[peaks_found] = vect_max_index(seg_squared, downsample_len) * scale_freq;
        peaks_amps[peaks_found] = seg_squared[peaks_locs[peaks_found] / scale_freq];
        starts[peaks_found] = peaks_locs[peaks_found] - 1;
        ends[peaks_found] = peaks_locs[peaks_found] + 1;
    }

    free(downsample_seg);
    free(seg_squared);

    *new_added = peaks_found;

    if (peaks_found > 0) {
        RA_LOG_ARRAY("POSTPROC", "_get_cough_peaks", "peaks_amps", peaks_amps, peaks_found);
    }
}

uint16_t _clean_cough_segments(uint16_t *starts_idxs,
                               uint16_t *ends_idxs,
                               uint16_t *peaks_locs,
                               float *peaks,
                               uint16_t n_peaks,
                               uint16_t fs)
{
    float min_dist_btwn_cough_peaks = COUGH_BURST_MIN_DUR + COUGH_EXP_MIN_DUR;
    float min_time_before_peak = COUGH_BURST_MIN_DUR / 2;
    float min_time_after_peak = COUGH_BURST_MIN_DUR / 2 + COUGH_EXP_MIN_DUR;
    float max_dist_btwn_cough_peaks_in_burst = COUGH_EXP_MAX_DUR + COUGH_BURST_MIN_DUR;

    uint16_t *sorted_idxs = (uint16_t *)malloc(n_peaks * sizeof(uint16_t));
    argsort(peaks_locs, n_peaks, sorted_idxs);

    order_by_idxs(starts_idxs, n_peaks, sorted_idxs, UINT16_T_SORT);
    order_by_idxs(ends_idxs, n_peaks, sorted_idxs, UINT16_T_SORT);
    order_by_idxs(peaks_locs, n_peaks, sorted_idxs, UINT16_T_SORT);
    order_by_idxs(peaks, n_peaks, sorted_idxs, FLOAT_SORT);

    free(sorted_idxs);

    uint16_t idxs_merged[n_peaks];
    uint16_t n_peaks_merged = 0;

    uint16_t locs_final[n_peaks];
    uint16_t n_peaks_final = 0;

    float dist = 0;
    uint16_t tmp_start = 0;
    uint16_t tmp_end = 0;
    uint16_t tmp_loc = 0;

    for (uint16_t i = 0; i < n_peaks; i++) {
        if (!_contains(idxs_merged, n_peaks_merged, i)) {
            tmp_start = starts_idxs[i];
            tmp_end = ends_idxs[i];
            tmp_loc = peaks_locs[i];

            for (uint16_t j = (i + 1); j < n_peaks; j++) {
                dist = (peaks_locs[j] - peaks_locs[i]) / (float)fs;
                if (dist < min_dist_btwn_cough_peaks) {
                    idxs_merged[n_peaks_merged] = j;
                    n_peaks_merged++;

                    tmp_start = min(tmp_start, starts_idxs[j]);
                    tmp_end = min(tmp_end, ends_idxs[j]);

                    if (peaks[i] < peaks[j]) {
                        tmp_loc = peaks_locs[j];
                    }
                }
            }

            starts_idxs[n_peaks_final] = tmp_start;
            ends_idxs[n_peaks_final] = tmp_end;
            locs_final[n_peaks_final] = tmp_loc;

            n_peaks_final++;
        }
    }

    float *cough_distances = (float *)malloc((uint32_t)((n_peaks_final - 1) * sizeof(float)));

    for (uint16_t i = 0; i < (n_peaks_final - 1); i++) {
        cough_distances[i] = (locs_final[i + 1] - locs_final[i]) / (float)fs;
    }

    float *cough_burst_distances = (float *)malloc((n_peaks_final - 1) * sizeof(float));
    uint16_t n_busts_dists = 0;
    for (uint16_t i = 0; i < (n_peaks_final - 1); i++) {
        if (cough_distances[i] <= max_dist_btwn_cough_peaks_in_burst) {
            cough_burst_distances[n_busts_dists] = cough_distances[i];
            n_busts_dists++;
        }
    }

    free(cough_distances);

    float avg_cough_end_times = min_time_after_peak;

    if (n_busts_dists > 0) {
        avg_cough_end_times = vect_mean(cough_burst_distances, n_busts_dists) - COUGH_BURST_MAX_DUR;
    }

    free(cough_burst_distances);

    float time_start_peak = 0.0;
    float time_to_next_peak = 0.0;
    uint16_t cough_series_count = 0;
    float series_multiplier = 0.0;

    for (uint16_t i = 0; i < n_peaks_final; i++) {
        time_start_peak = (locs_final[i] - starts_idxs[i]) / (float)fs;

        if (time_start_peak < min_time_before_peak) {
            starts_idxs[i] = locs_final[i] - (uint16_t)(min_time_before_peak * fs);
        }

        if (i < (n_peaks_final - 1)) {
            time_to_next_peak = (float)(locs_final[i + 1] - locs_final[i]) / fs;
        } else {
            time_to_next_peak = 100;
        }

        if (time_to_next_peak > max_dist_btwn_cough_peaks_in_burst) {
            if (cough_series_count > 0) {
                series_multiplier = COUGH_LEN_IN_SERIES_DECREASE_FACTOR;
                for (uint16_t j = 0; j < (cough_series_count - 1); j++) {
                    series_multiplier *= series_multiplier;
                }
            } else {
                series_multiplier = 1;
            }

            ends_idxs[i] = locs_final[i] + (uint16_t)(series_multiplier * avg_cough_end_times * fs);
            cough_series_count = 0;
        } else {
            cough_series_count++;

            if (i < (n_peaks_final - 1)) {
                if ((time_to_next_peak - COMPRESSIVE_PHASE_DUR) < min_dist_btwn_cough_peaks) {
                    ends_idxs[i] = starts_idxs[i + 1] - (uint16_t)(COUGH_BURST_MIN_DUR * fs);
                } else {
                    ends_idxs[i] = locs_final[i + 1] - (uint16_t)(COMPRESSIVE_PHASE_DUR * fs) - (uint16_t)(COUGH_BURST_MIN_DUR * fs);
                }
            }
        }
    }

    return n_peaks_final;
}

#endif
