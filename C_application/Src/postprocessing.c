#include <stdio.h>
#include <stdlib.h>

#include <postprocessing.h>

/**
 * Checks if a specific element is contained in the given array.
 * If yes, it returns 1, 0 otherwise.
 * 
 * @param *arr  :   pointer to the input array
 * @param len   :   length of the input array
 * @param elem  :   element to find inside the array
 * 
*/
uint8_t _contains(uint16_t *arr, uint16_t len, uint16_t elem){

    for(uint16_t i=0; i<len; i++){
        if(arr[i] == elem){
            return 1;       // return true
        }
    }

    return 0;
}


/**
 * Downsamples a signal arrat according to the `FS_DOWNSAMPLE` parameter.
 * The resulting signal is first downsampled and normalized by subtracting the mean
 * and dividing by the maximum absolute value.
 * 
 * @param *sig      :   pointer to the input signal to be downsampled
 * @param len       :   length of the input signal
 * @param fs        :   original sampling frequency of the input signal 
 * @param *new_len  :   pointer where to store the new length of the downampled signal
 * 
 * @return res      :   pointer to the resulting downsampled and normalized siagnal
 * 
*/
float* _downsample(const float* sig, int16_t len, int16_t fs, int16_t *new_len){

    // Compute the new length of the downsamples signal
    int8_t scale_factor = fs / FS_DOWNSAMPLE;
    *new_len = len / scale_factor;

    float* res = (float*)malloc(*new_len * sizeof(float));
    float mean = 0.0;

    // Downsample and accumulate the mean
    for(int16_t i=0; i<*new_len; i++){
        res[i] = sig[i*scale_factor];
        mean += res[i];
    }

    mean = mean / *new_len;

    // Subtract the mean
    sub_constant(res, *new_len, mean, res);

    // Divide by the maximum absolute value
    float max_abs = vect_max_abs_value(res, *new_len);
    vect_div_const(res, *new_len, max_abs, res);

    return res;
}


void _get_cough_peaks(const float* seg, int16_t len, int16_t fs, uint16_t *starts, uint16_t *ends, uint16_t *peaks_locs, float *peaks_amps, uint16_t *new_added){
    
    // Downsample //
    int16_t downsample_len = 0.0;
    float* downsample_seg = _downsample(seg, len, fs, &downsample_len);

    // Hysteresys thresholds //

    // Stores the squared values of the downsampled signal
    float *seg_squared = (float*)malloc(downsample_len * sizeof(float));
    vect_mult(downsample_seg, downsample_seg, downsample_len, seg_squared);
    
    float peak = vect_max_value(seg_squared, downsample_len);

    // Thresholds 
    float th_low = sqrtf(vect_mean(seg_squared, downsample_len));   // RMS
    float th_high = 0.25 * peak + 0.75 * th_low;

    int16_t cough_start = 0;
    int16_t cough_end = 0;
    int8_t below_th_counter = 0;
    int16_t tolerance = COUGH_END_TOLERANCE * FS_DOWNSAMPLE;

    int16_t scale_freq = (int16_t)(fs / FS_DOWNSAMPLE);

    int8_t cough_in_progress = 0;   // Initialized as false
    int8_t peaks_found = 0;

    for(int16_t i=0; i<downsample_len; i++){
        if (cough_in_progress){
            
            if(i == (downsample_len-1)){
                cough_end = i;
                cough_in_progress = 0;      // Set to false

                // Update segment idxs, peaks_locs, and peaks
                starts[peaks_found] = cough_start * scale_freq;
                ends[peaks_found] = cough_end * scale_freq;

                peaks_locs[peaks_found] = (vect_max_index(&downsample_seg[cough_start], (cough_end - cough_start)+1) + cough_start) * scale_freq;
                peaks_amps[peaks_found] = downsample_seg[peaks_locs[peaks_found] / scale_freq];

                peaks_found++;

            } else if(seg_squared[i] < th_low){
                
                below_th_counter++;

                if (below_th_counter > tolerance){
                    if(i < downsample_len)
                        cough_end = i;
                    else
                        cough_end = (downsample_len - 1);
                    cough_in_progress = 0;  // Set to false

                    // Update segment idxs, peaks_locs, and peaks
                    starts[peaks_found] = cough_start * scale_freq;
                    ends[peaks_found] = cough_end * scale_freq;

                    peaks_locs[peaks_found] = (vect_max_index(&downsample_seg[cough_start], (cough_end - cough_start + 1)) + cough_start) * scale_freq;
                    peaks_amps[peaks_found] = downsample_seg[peaks_locs[peaks_found] / scale_freq];
                

                    peaks_found++;
                }
            } else {
                below_th_counter = 0;
            }
        }
        else {
            if(seg_squared[i] > th_high){
                if(i >= 0){
                    cough_start = i;
                } else {
                    cough_start = 0;
                }

                cough_in_progress = 1;
                below_th_counter = 0;
            }
        }
    }


    if (peaks_found == 0){
        peaks_locs[peaks_found] = vect_max_index(seg_squared, downsample_len) * scale_freq;
        peaks_amps[peaks_found] = seg_squared[peaks_locs[peaks_found] / scale_freq];
        starts[peaks_found] = peaks_locs[peaks_found] - 1;
        ends[peaks_found] = peaks_locs[peaks_found] + 1;
    }

    free(downsample_seg);
    free(seg_squared);

    *new_added = peaks_found;

}


uint16_t _clean_cough_segments(uint16_t *starts_idxs, uint16_t *ends_idxs, uint16_t *peaks_locs, float *peaks, uint16_t n_peaks, uint16_t fs){

    // Set min/max cough lengths based on cough physiology constants
    float min_dist_btwn_cough_peaks = COUGH_BURST_MIN_DUR + COUGH_EXP_MIN_DUR;
    float min_time_before_peak = COUGH_BURST_MIN_DUR / 2;
    float min_time_after_peak = COUGH_BURST_MIN_DUR / 2 + COUGH_EXP_MIN_DUR;
    float max_dist_btwn_cough_peaks_in_burst = COUGH_EXP_MAX_DUR + COUGH_BURST_MIN_DUR;

    // Sort based on peak locations
    uint16_t *sorted_idxs = (uint16_t*)malloc(n_peaks * sizeof(uint16_t));
    argsort(peaks_locs, n_peaks, sorted_idxs);

    order_by_idxs(starts_idxs, n_peaks, sorted_idxs, UINT16_T_SORT);
    order_by_idxs(ends_idxs, n_peaks, sorted_idxs, UINT16_T_SORT);    
    order_by_idxs(peaks_locs, n_peaks, sorted_idxs, UINT16_T_SORT);
    order_by_idxs(peaks, n_peaks, sorted_idxs, FLOAT_SORT);

    free(sorted_idxs);

    // Merge regions whose peaks are close together
    uint16_t idxs_merged[n_peaks];
    uint16_t n_peaks_merged = 0;
    
    uint16_t locs_final[n_peaks];
    uint16_t n_peaks_final = 0;

    float dist = 0;
    uint16_t tmp_start = 0;
    uint16_t tmp_end = 0;
    uint16_t tmp_loc = 0;


    for(uint16_t i=0; i<n_peaks; i++){

        // Check that the current peak has not been merged yet, otherwise skip the iteration
        if (!_contains(idxs_merged, n_peaks_merged, i)){
            tmp_start = starts_idxs[i];
            tmp_end = ends_idxs[i];
            tmp_loc = peaks_locs[i];
            // tmp_amp = peaks[i];

            for(uint16_t j=(i+1); j<n_peaks; j++){
                dist = (peaks_locs[j] - peaks_locs[i]) / (float)fs;    // distance between 2 peaks
                if(dist < min_dist_btwn_cough_peaks){
                    idxs_merged[n_peaks_merged] = j;
                    n_peaks_merged++;

                    // Update tmp_start and tmp_end
                    tmp_start = min(tmp_start, starts_idxs[j]);
                    tmp_end = min(tmp_end, ends_idxs[j]);

                    if(peaks[i] < peaks[j]){
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

    // Determine cough duration for coughs that aren't in a burst

    // Casting to (uint32_t) to avoid the following warning: `argument 1 value ‘18446744073709551612’ exceeds maximum object size 9223372036854775807`
    // It seems to be a GCC bug
    float *cough_distances = (float*)malloc((uint32_t)((n_peaks_final-1) * sizeof(float)));

    for(uint16_t i=0; i<(n_peaks_final-1); i++){
        cough_distances[i] = (locs_final[i+1] - locs_final[i]) / (float)fs;
    }

    float *cough_burst_distances = (float*)malloc((n_peaks_final-1) * sizeof(float));
    uint16_t n_busts_dists = 0;
    for(uint16_t i=0; i<(n_peaks_final-1); i++){
        if(cough_distances[i] <= max_dist_btwn_cough_peaks_in_burst){
            cough_burst_distances[n_busts_dists] = cough_distances[i];
            n_busts_dists++;
        }
    }

    free(cough_distances);

    float avg_cough_end_times = min_time_after_peak;

    if(n_busts_dists > 0){
        avg_cough_end_times = vect_mean(cough_burst_distances, n_busts_dists) - COUGH_BURST_MAX_DUR;
    }

    free(cough_burst_distances);

    // Refine cough segment starts and ends

    float time_start_peak = 0.0;
    float time_to_next_peak = 0.0;
    uint16_t cough_series_count = 0;
    float series_multiplier = 0.0;

    for(uint16_t i=0; i<n_peaks_final; i++){
        time_start_peak = (locs_final[i] - starts_idxs[i]) / (float)fs;

        // If the cough burst is too short, modify its start time
        if(time_start_peak < min_time_before_peak){
            starts_idxs[i] = locs_final[i] - (uint16_t)(min_time_before_peak * fs);
        }

        // Measure distance between first peak and next to determine if cough is in a series
        if(i < (n_peaks_final - 1)){
            time_to_next_peak = (float)(locs_final[i+1] - locs_final[i]) / fs;
        } else {
            time_to_next_peak = 100;
        }

        // If it's not in a series or at the end of a series, assign the average cough duration
        if(time_to_next_peak > max_dist_btwn_cough_peaks_in_burst){

            if(cough_series_count > 0){
                series_multiplier = COUGH_LEN_IN_SERIES_DECREASE_FACTOR;
                for(uint16_t j=0; j<(cough_series_count-1); j++){
                    series_multiplier *= series_multiplier;
                }
            } else {
                series_multiplier = 1;
            }

            ends_idxs[i] = locs_final[i] + (uint16_t)(series_multiplier * avg_cough_end_times * fs);
            cough_series_count = 0;
        }
        // Otherwise, end the cough before the start of the next cough
        else {
            cough_series_count++;

            if(i < (n_peaks_final - 1)){
                if((time_to_next_peak - COMPRESSIVE_PHASE_DUR) < min_dist_btwn_cough_peaks){ //Coughs in a series without recompressing lungs
                    ends_idxs[i] = starts_idxs[i+1] - (uint16_t)(COUGH_BURST_MIN_DUR * fs);
                } else {
                    // Coughs in a series with recompressing lungs
                    ends_idxs[i] = locs_final[i+1] - (uint16_t)(COMPRESSIVE_PHASE_DUR * fs) - (uint16_t)(COUGH_BURST_MIN_DUR * fs);
                }
            }
        }
    }

    return n_peaks_final;
}
