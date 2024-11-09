import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys
import IPython.display as ipd
from enum import Enum
sys.path.append(os.path.abspath('src'))
from .helpers import *
import json


def segmentation_with_cough_tolerance(data_folder, subj_id, experiment_conditions, window_len, overlap, fs_audio,  overlap_threshold=0.7, tolerance_before=0.2, tolerance_after = 0.05):
    """
    For each subject, extract windows of all of the sounds for each movement, noise condition, and trial specified in the selected experimental conditions.
    Inputs:
    - data_folder (string): folder where input data is stored
    - subj_id (string): ID of the subject 
    - experiment_conditions (dictionary): dict of selected experiment conditinons for which to run the tests (i.e. sound, movement, noise, trial)
    - window_len (float): length of signal window in seconds for performing classification
    - overlap (int): percent (in range (0-100)) overlap from one window to the next
    - fs_audio (int): sampling frequency of the audio signals; used for downsampling
    - overlap_threshold (float between 0 and 1): percent of signal overlapping with cough above which to classify a segment as a cough
    Outputs: 
    - audio_data: NxMx2 data matrix where 
        - N = Number of segments
        - M = int(window_len * fs_audio)
        - first index = outer microphone, second index = body-facing microphone
    - imu_data: NxLx6 data matrix where
        - L = int(window_len * 100)
        - third dimension specifies IMU signal (accel x,y,z, IMU y,p,r)
    - labels: Nx1 vector of labels
        - 1 = cough
        - 0 = background of cough recording
        - -1: laugh
        - -2: breathing
        - -3: throat clearing
        - -4: talking
    - <trial/mov/noise>_log: Nx1 arrays of experimental conditions of each segment
    """
    # Set up result vectors
    window_len_audio = int(window_len*fs_audio)
    window_len_imu = int(window_len*FS_IMU)
    audio_data = np.zeros((1,window_len_audio,2))
    imu_data = np.zeros((1,window_len_imu,6))
    labels_data = np.zeros(1)
    
    # Set up segmentation
    step_samp = int(np.rint(window_len_audio*(1 - overlap/100)))
    n_overlaps = int(window_len_audio/step_samp -1)

    step_samp_imu = int(np.rint(window_len_imu*(1 - overlap/100)))
    
    # Extract signal windows for each noise condition
    trial_log = np.array([])
    mov_log = np.array([])
    noise_log = np.array([])
    for trial in experiment_conditions.trials:
        for mov in experiment_conditions.movements:
            for noise in experiment_conditions.noises:
                for sound in experiment_conditions.sounds:
                    path = data_folder + subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                    if os.path.exists(path):
                        if (len(os.listdir(path)) > 0):
                            # Extract signal windows
                            air, skin = load_audio(data_folder, subj_id, fs_audio, trial, mov, noise, sound)
                            imu = load_imu(data_folder, subj_id, trial, mov, noise, sound)
                            imu_stack = np.stack((imu.x,imu.y,imu.z,imu.Y,imu.P,imu.R),axis=1)

                            n_runs = min(int(len(air)/step_samp), int(len(imu.x)/step_samp_imu)) - n_overlaps
                            if n_runs<=0:
                                print('problem: {0}'.format(path))
                            else:

                                if ((sound == Sound.COUGH) & (os.path.isfile(path + '/ground_truth.json'))):
                                    fn = path + '/ground_truth.json'
                                    with open(fn, 'rb') as f:
                                        loaded_dict = json.load(f)
                                    starts = loaded_dict["start_times"]
                                    ends = loaded_dict["end_times"]
                                    starts = np.array(starts) - tolerance_before
                                    ends = np.array(ends) + tolerance_after

                                    # Get indices of coughs (wrt IMU signal indexing)
                                    cough_mask = np.zeros(imu_stack.shape[0])
                                    individual_cough_indices = []
                                    for s, e in zip(starts, ends):
                                        #find indices of coughs
                                        start_idx = int(np.rint(s*FS_IMU))
                                        end_idx = int(np.rint(e*FS_IMU))
                                        cough_mask[start_idx:end_idx+1] = 1
                                        individual_cough_indices.append(np.arange(start_idx,end_idx+1))
                                        cough_indices = np.where(cough_mask==1)

                                # Segment signals
                                n_new_samples = 0
                                for j in range(n_runs):
                                    seg_out = air[j*step_samp:j*step_samp+window_len_audio]
                                    seg_in = skin[j*step_samp:j*step_samp+window_len_audio]
                                    audio_stack = np.stack((seg_out,seg_in),axis=1).reshape(1,window_len_audio,2)
                                    seg_imu = imu_stack[j*step_samp_imu:j*step_samp_imu+window_len_imu,:].reshape(1,-1,6)

                                    if ((sound == Sound.COUGH) & (os.path.isfile(path + '/ground_truth.json'))):
                                        # Measure how much of the window is cough samples
                                        sample_range = np.arange(j*step_samp_imu,j*step_samp_imu+window_len_imu)
                                        perc_overlap_with_cough = sum(np.in1d(sample_range,cough_indices))/len(sample_range)

                                        # Measure how much of the cough sample is included in the sample
                                        percents_of_coughs_in_sample = np.zeros(len(individual_cough_indices))
                                        for k,cough_ndx in enumerate(individual_cough_indices):
                                            perc_of_cough_in_sample = sum(np.in1d(cough_ndx,sample_range))/len(cough_ndx)
                                            percents_of_coughs_in_sample[k] = perc_of_cough_in_sample
                                        perc_overlap_cough_sample = np.max(percents_of_coughs_in_sample)

                                        if ((perc_overlap_with_cough >= overlap_threshold) | (perc_overlap_cough_sample >= overlap_threshold)):
                                            labels_data = np.concatenate((labels_data,[1]))
                                            audio_data = np.concatenate((audio_data,audio_stack), axis=0)
                                            imu_data = np.concatenate((imu_data,seg_imu),axis=0)
                                            n_new_samples += 1

                                        elif ((perc_overlap_with_cough == 0) & (perc_overlap_cough_sample == 0)):
                                            labels_data = np.concatenate((labels_data,[0]))
                                            audio_data = np.concatenate((audio_data,audio_stack), axis=0)
                                            imu_data = np.concatenate((imu_data,seg_imu),axis=0)
                                            n_new_samples += 1
                                    else:
                                        audio_data = np.concatenate((audio_data,audio_stack), axis=0)
                                        imu_data = np.concatenate((imu_data,seg_imu),axis=0)
                                        if (sound == Sound.LAUGH):
                                            labels_data = np.concatenate((labels_data,[-1]))
                                        elif (sound == Sound.BREATH):
                                            labels_data = np.concatenate((labels_data,[-2]))
                                        elif (sound == Sound.THROAT):
                                            labels_data = np.concatenate((labels_data,[-3]))
                                        elif (sound == Sound.TALK):
                                            labels_data = np.concatenate((labels_data,[-4]))
                                        n_new_samples += 1
                                
                                # Keep track of experimental segments
                                trial_log = np.concatenate((trial_log,np.tile(trial,n_new_samples)))
                                mov_log = np.concatenate((mov_log,np.tile(mov,n_new_samples)))
                                noise_log = np.concatenate((noise_log,np.tile(noise,n_new_samples)))
                                

    audio_data = np.delete(audio_data,0,axis=0)
    imu_data = np.delete(imu_data,0,axis=0)
    labels_data = np.delete(labels_data,0)
    return audio_data, imu_data, labels_data, trial_log, mov_log, noise_log

