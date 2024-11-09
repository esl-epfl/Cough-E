"""
Extract features

Author: Lara Orlandic
Contact: lara.orlandic@epfl.ch

Perform feature extraction on specified signals of the cough counting dataset and save them to intermediate files for further processing.
"""

from tty import CFLAG
import hydra
import os, sys
from src import *


@hydra.main(version_base=None, config_path="config", config_name="config")


def run_config(cfg):

    features_folder = "intermediates/{0}/".format(cfg.ext_feat_folder)
    subj_ids = os.listdir(cfg.data_folder)

    # Set up feature extractors
    names_audio,counts_audio = generate_feature_name_vec(cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands)
    n_audio_feat = len(names_audio)
    names_imu,counts_imu = generate_feature_name_vec_imu(cfg.feature_extraction.DP_epsilon)
    n_imu_feat = len(names_imu) 

    if not os.path.exists(features_folder):
        print("Setting up folder {0}".format(cfg.ext_feat_folder))
        os.mkdir(features_folder)
    
    feat_select_vec = np.ones(n_audio_feat)
    audio_feat_extr = AudioFeatures(feat_select_vec==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)

    feat_select_vec_imu = np.ones(n_imu_feat)
    imu_feat_extr = IMUFeatures(feat_select_vec_imu==1,counts_imu, names_imu, DP_epsilon=cfg.feature_extraction.DP_epsilon)
    
    print("Extracting {0} audio features and {1} IMU features".format(n_audio_feat if (cfg.signals.out_mic_sel | cfg.signals.in_mic_sel) else 0, n_imu_feat if (cfg.signals.imu_sel) else 0))

    if cfg.feature_extraction.ext_train_feats:
        
        for s in subj_ids:
            print("Extracting features for subject {0}".format(s))
            # Set up folder for subject
            out_subj_folder = features_folder + s + "/"
            if not os.path.exists(out_subj_folder):
                os.mkdir(out_subj_folder)
            audio, imu, label, trial_log, mov_log, noise_log = segmentation_with_cough_tolerance(cfg.data_folder, s, cfg.experiment_conditions.conditions_train, cfg.signals.window_len, cfg.signals.overlap_train, cfg.signals.fs_audio, overlap_threshold=0.7, tolerance_before=0.2, tolerance_after = 0.05)
            if cfg.signals.out_mic_sel:
                X_audio_out = np.zeros((audio.shape[0],n_audio_feat))
            if cfg.signals.in_mic_sel:
                X_audio_in = np.zeros((audio.shape[0],n_audio_feat))
            if cfg.signals.imu_sel: 
                X_imu = np.zeros((audio.shape[0],n_imu_feat))
            count = 0
            for audio_sigs, imu_sigs in zip(audio,imu):
                # Extract inner mic audio features
                if cfg.signals.in_mic_sel:
                    feats_audio_in, _ = audio_feat_extr.compute_features(audio_sigs[:,1],cfg.signals.fs_audio)
                    X_audio_in[count,:] = feats_audio_in
                # Extract outer mic audio features
                if cfg.signals.out_mic_sel:
                    feats_audio_out, _ = audio_feat_extr.compute_features(audio_sigs[:,0],cfg.signals.fs_audio)
                    X_audio_out[count,:] = feats_audio_out
                # Extract IMU features for each signal
                if cfg.signals.imu_sel:
                    feats_imu = []
                    for i, signal in enumerate(IMU_Signal):
                        if i<6:
                            sig = imu_sigs[:,i]
                        elif i==6:
                            sig = np.linalg.norm((imu_sigs[:,0],imu_sigs[:,1],imu_sigs[:,2]), axis=0)
                        elif i==7:
                            sig = np.linalg.norm((imu_sigs[:,3],imu_sigs[:,4],imu_sigs[:,5]), axis=0)
                        feats, _ = imu_feat_extr.compute_features(sig,FS_IMU,signal)
                        for feat in feats:
                            feats_imu.append(feat)
                    feats_imu = np.array(feats_imu)
                    X_imu[count,:] = feats_imu
                count += 1
            # Save data to numpy arrays
            if cfg.signals.in_mic_sel:
                np.save(out_subj_folder + "audio_in_features.npy", X_audio_in)
                np.save(out_subj_folder + "audio_in_names.npy", names_audio)
            if cfg.signals.out_mic_sel:
                np.save(out_subj_folder + "audio_out_features.npy", X_audio_out)
                np.save(out_subj_folder + "audio_out_names.npy", names_audio)
            if cfg.signals.imu_sel:
                print("\t> N features:\t{}".format(X_imu.shape))
                np.save(out_subj_folder + "imu_features.npy", X_imu)
                np.save(out_subj_folder + "imu_names.npy", names_imu)
            np.save(out_subj_folder + "labels.npy", label)
            np.save(out_subj_folder + "trial_log.npy", trial_log)
            np.save(out_subj_folder + "mov_log.npy", mov_log)
            np.save(out_subj_folder + "noise_log.npy", noise_log)

    if cfg.feature_extraction.ext_test_feats:
        subj_ids = os.listdir(cfg.test_data_folder)
        print("Extracting testing features")
        for s in subj_ids:
            print("Extracting features for subject {0}".format(s))
            # Set up folder for subject
            out_subj_folder = features_folder + s + "/"
            if not os.path.exists(out_subj_folder):
                os.mkdir(out_subj_folder)
            audio, imu, label, trial_log, mov_log, noise_log = segmentation_with_cough_tolerance(cfg.test_data_folder, s, cfg.experiment_conditions.conditions_test, cfg.signals.window_len, cfg.signals.overlap_train, cfg.signals.fs_audio, overlap_threshold=0.7, tolerance_before=0.2, tolerance_after = 0.05)
            if cfg.signals.out_mic_sel:
                X_audio_out = np.zeros((audio.shape[0],n_audio_feat))
            if cfg.signals.in_mic_sel:
                X_audio_in = np.zeros((audio.shape[0],n_audio_feat))
            if cfg.signals.imu_sel: 
                X_imu = np.zeros((audio.shape[0],n_imu_feat))
            count = 0
            for audio_sigs, imu_sigs in zip(audio,imu):
                # Extract inner mic audio features
                if cfg.signals.in_mic_sel:
                    feats_audio_in, _ = audio_feat_extr.compute_features(audio_sigs[:,1],cfg.signals.fs_audio)
                    X_audio_in[count,:] = feats_audio_in
                # Extract outer mic audio features
                if cfg.signals.out_mic_sel:
                    feats_audio_out, _ = audio_feat_extr.compute_features(audio_sigs[:,0],cfg.signals.fs_audio)
                    X_audio_out[count,:] = feats_audio_out
                # Extract IMU features for each signal
                if cfg.signals.imu_sel:
                    feats_imu = []
                    for i, signal in enumerate(IMU_Signal):
                        if i<6:
                            sig = imu_sigs[:,i]
                        elif i==6:
                            sig = np.linalg.norm((imu_sigs[:,0],imu_sigs[:,1],imu_sigs[:,2]), axis=0)
                        elif i==7:
                            sig = np.linalg.norm((imu_sigs[:,3],imu_sigs[:,4],imu_sigs[:,5]), axis=0)
                        feats, _ = imu_feat_extr.compute_features(sig,FS_IMU,signal)
                        for feat in feats:
                            feats_imu.append(feat)
                    feats_imu = np.array(feats_imu)
                    X_imu[count,:] = feats_imu
                count += 1
            # Save data to numpy arrays
            if cfg.signals.in_mic_sel:
                np.save(out_subj_folder + "audio_in_features.npy", X_audio_in)
                np.save(out_subj_folder + "audio_in_names.npy", names_audio)
            if cfg.signals.out_mic_sel:
                np.save(out_subj_folder + "audio_out_features.npy", X_audio_out)
                np.save(out_subj_folder + "audio_out_names.npy", names_audio)
            if cfg.signals.imu_sel:
                print("\t> N features:\t{}".format(X_imu.shape))
                np.save(out_subj_folder + "imu_features.npy", X_imu)
                np.save(out_subj_folder + "imu_names.npy", names_imu)
            np.save(out_subj_folder + "labels.npy", label)
            np.save(out_subj_folder + "trial_log.npy", trial_log)
            np.save(out_subj_folder + "mov_log.npy", mov_log)
            np.save(out_subj_folder + "noise_log.npy", noise_log)

if __name__ == "__main__":
    run_config()


