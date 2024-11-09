"""
Extract coughvid features

Author: Lara Orlandic
Contact: lara.orlandic@epfl.ch

Perform feature extraction on the coughvid dataset and save them to intermediate files to use for semi-supervised learning.
"""

from tty import CFLAG
import hydra
import os, sys
from src import *


@hydra.main(version_base=None, config_path="config", config_name="config")

def run_config(cfg):
    # Set up output folder to store extracted feature files
    out_folder = "intermediates/coughvid/{0}/".format(cfg.ext_feat_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Load coughvid data for pre-training
    coughvid_folder = cfg.coughvid_folder
    df = pd.read_csv(cfg.coughvid_metadata)
    coughvid_files = df.loc[~df.gender.isna() & (df.cough_detected>0.8)].uuid.to_numpy()
    print("There are {0} COUGHVID recordings".format(len(coughvid_files)))

    # Set up feature extraction
    names_audio,counts_audio = generate_feature_name_vec(cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands)
    n_audio_features = len(names_audio)
    feat_select_vec = np.ones(n_audio_features)
    audio_feat_extr = AudioFeatures(feat_select_vec==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)

    # Save feature names
    np.save(out_folder + "audio_in_names.npy", names_audio)

    # Set up signal segmentation
    window_samp = int(cfg.signals.window_len*cfg.signals.fs_audio)
    step_samp = int(np.rint(window_samp*(1 - cfg.signals.overlap_train/100)))
    n_overlaps = int(window_samp/step_samp -1)

    # Extract features for each recording
    for i, fn in enumerate(coughvid_files):
        
        if i%1000==0:
            print("Processing file {0}/{1}".format(i,len(coughvid_files)))
        
        try:
            # Feature vector for each window in recording
            cvid_signal, _ = librosa.load(coughvid_folder + fn + ".wav", sr=cfg.signals.fs_audio)
            extra_features = np.array([0])
            subject_gender = df.loc[df.uuid == fn].gender
            if subject_gender.item()=="female":
                extra_features[0] = 1

            n_runs = int(len(cvid_signal)/step_samp) - n_overlaps
            feat_arrs = []

            if n_runs>0:

                # Run model across signal with a given overlap and save predictions
                for j in range(n_runs):
                    try:
                        seg_out = cvid_signal[j*step_samp:j*step_samp+window_samp]
                        feats, _ = audio_feat_extr.compute_features(seg_out,cfg.signals.fs_audio)
                        feats = np.concatenate((feats, extra_features))
                        feat_arr = np.array(feats).reshape(1, -1)
                        feat_arrs.append(feat_arr)
                    except:
                        continue

            feat_arrs = np.array(feat_arrs).squeeze(axis=1)

            if len(feat_arrs)>0:
                # Set up folder for recording
                out_subj_folder = out_folder + fn + "/"
                if not os.path.exists(out_subj_folder):
                    os.mkdir(out_subj_folder)

                # Save data to numpy arrays
                np.save(out_subj_folder + "feature_array.npy", feat_arrs)
        except:
            continue

    print("DONE!")


if __name__ == "__main__":
    run_config()