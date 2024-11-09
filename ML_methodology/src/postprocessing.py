"""Functions related to cough signal post-processing; detection of peaks following the classifier output"""

import numpy as np

# Constants related to cough physiology
cough_burst_min_dur = 0.03
cough_burst_max_dur = 0.05
cough_exp_min_dur = 0.2
cough_exp_max_dur = 0.5
compressive_phase_dur = 0.2
cough_length_in_series_decrease_factor = 0.8

def get_cough_peaks(x, fs, fs_downsample, tolerance_multiplier):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""

    # Downsample the segment for computational efficiency
    downsample_seg = x[np.arange(0,len(x),int(fs/fs_downsample))]
    downsample_seg = downsample_seg - np.mean(downsample_seg)
    downsample_seg = downsample_seg/np.max(np.abs(downsample_seg))

    #Define hysteresis thresholds
    x_squared = downsample_seg**2
    rms = np.sqrt(np.mean(x_squared))
    peak = np.max(x_squared)
    seg_th_l = rms
    seg_th_h = 0.25*peak + 0.75*rms

    #Segment coughs
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(tolerance_multiplier*fs_downsample)
    below_th_counter = 0
    
    segment_indices = []
    peak_locs = []
    peaks = []
    
    for i, sample in enumerate(x_squared):
        if cough_in_progress:
            if i == (len(downsample_seg)-1):
                cough_end=i
                cough_in_progress = False
                segment_indices.append(np.array([cough_start,cough_end])*int(fs/fs_downsample))
                peak_locs.append((np.argmax(downsample_seg[cough_start:cough_end+1])+cough_start)*int(fs/fs_downsample))
                peaks.append(np.max(downsample_seg[cough_start:cough_end+1]))
                #print(cough_start, cough_end)
            elif sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i if (i < len(downsample_seg)) else len(downsample_seg)-1
                    cough_in_progress = False
                    segment_indices.append(np.array([cough_start,cough_end])*int(fs/fs_downsample))
                    peak_locs.append((np.argmax(downsample_seg[cough_start:cough_end+1])+cough_start)*int(fs/fs_downsample))
                    peaks.append(np.max(downsample_seg[cough_start:cough_end+1]))
                    #print(cough_start, cough_end)
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i if (i >=0) else 0
                cough_in_progress = True
                below_th_counter = 0
    
    if len(peaks)==0:
        peaks.append(np.max(x_squared))
        peak_loc = np.argmax(x_squared)*int(fs/fs_downsample)
        peak_locs.append(peak_loc)
        segment_indices.append(np.array([peak_loc-1, peak_loc+1]))

    return segment_indices, peak_locs, peaks

def clean_cough_segments(segment_indices_list, peak_locs_list, peak_amp_list, fs_audio):
    
    # Set min/max cough lengths based on cough physiology constants
    min_dist_btwn_cough_peaks = cough_burst_min_dur + cough_exp_min_dur
    min_time_before_peak = cough_burst_min_dur/2
    min_time_after_peak = cough_burst_min_dur/2 + cough_exp_min_dur
    max_dist_btwn_cough_peaks_in_burst = cough_exp_max_dur + cough_burst_min_dur
    max_cough_dur = cough_burst_max_dur + cough_exp_max_dur

    
    # Sort based on peak locations:
    peak_sorted_inds = np.argsort(peak_locs_list)
    peak_locs_list = np.array(peak_locs_list)[peak_sorted_inds]
    segment_indices_list = np.array(segment_indices_list)[peak_sorted_inds]
    peak_amp_list = np.array(peak_amp_list)[peak_sorted_inds]

    # Merge region whose peaks are close together
    segment_indices_final = []
    peak_locs_final = []
    peak_amp_final = []
    indices_merged = []
    for i,pk_loc in enumerate(peak_locs_list):
        if not np.in1d(i,indices_merged)[0]:
            segment_indices = segment_indices_list[i]
            peak_loc = peak_locs_list[i]
            peak_amp = peak_amp_list[i]
            for j in np.arange(i+1,len(peak_locs_list)):
                pk_dist = (peak_locs_list[j] - peak_locs_list[i])/fs_audio
                if pk_dist < min_dist_btwn_cough_peaks:
                    indices_merged.append(j)
                    segment_indices = np.array([np.min([segment_indices[0],segment_indices_list[j,0]]),np.max([segment_indices[1],segment_indices_list[j,1]])])
                    if peak_amp < peak_amp_list[j]:
                        peak_loc = peak_locs_list[j]
                        peak_amp = peak_amp_list[j]
            segment_indices_final.append(segment_indices)
            peak_locs_final.append(peak_loc)
            peak_amp_final.append(peak_amp)
    segment_indices_final = np.array(segment_indices_final)
    peak_locs_final = np.array(peak_locs_final)
    peak_amp_final = np.array(peak_amp_final)

    # Determine cough duration for coughs that aren't in a burst
    cough_distances = (peak_locs_final[1:] - peak_locs_final[:-1])/fs_audio
    cough_burst_distances = cough_distances[cough_distances<=max_dist_btwn_cough_peaks_in_burst]
    avg_cough_end_time = np.mean(cough_burst_distances) - cough_burst_max_dur if not len(cough_burst_distances)==0 else min_time_after_peak

    # Refine cough segment starts and ends
    i = 0
    cough_series_count = 0
    for (s,e), p in zip(segment_indices_final, peak_locs_final):
        time_start_peak = (p-s)/fs_audio
        # If the cough burst is too short, modify its start time
        if time_start_peak < min_time_before_peak:
            segment_indices_final[i,0] = p-int(min_time_before_peak*fs_audio)

        # Measure distance between first peak and next to determine if cough is in a series
        if i<len(peak_locs_final)-1:
            time_to_next_peak = (peak_locs_final[i+1]-p)/fs_audio
        else:
            time_to_next_peak = 100
        
        # If it's not in a series or at the end of a series, assign the average cough duration
        if time_to_next_peak > max_dist_btwn_cough_peaks_in_burst:
            series_multiplier = cough_length_in_series_decrease_factor**cough_series_count #if cough_series_count>0 else 1
            segment_indices_final[i,1] = p+int(series_multiplier*avg_cough_end_time*fs_audio)
            cough_series_count = 0
        # Otherwise, end the cough before the start of the next cough
        else:
            cough_series_count += 1
            if i<len(peak_locs_final)-1:
                if (time_to_next_peak-compressive_phase_dur<min_dist_btwn_cough_peaks): # Coughs in a series without recompressing lungs
                    segment_indices_final[i,1] = segment_indices_final[i+1,0] - int(cough_burst_min_dur*fs_audio)
                else: # Coughs in a series with recompressing lungs
                    segment_indices_final[i,1] = peak_locs_final[i+1] - int(compressive_phase_dur*fs_audio) - int(cough_burst_min_dur*fs_audio)
        i += 1
    
    return segment_indices_final, peak_locs_final