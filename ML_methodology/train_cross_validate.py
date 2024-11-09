from tty import CFLAG
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import subprocess
from src import *
from sklearn.model_selection import GroupKFold
import pickle
from timescoring.annotations import Annotation
from timescoring import scoring


@hydra.main(version_base=None, config_path="config", config_name="config")


def run_config(cfg):

    # Set up file I/O
    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_folder}")
    print(cfg.ext_feat_folder)

    features_folder = "intermediates/{0}/".format(cfg.ext_feat_folder)
    coughvid_features_folder = "intermediates/coughvid/{0}/".format(cfg.ext_feat_folder)

    subj_ids = os.listdir(cfg.data_folder)
    print("Loaded data from {0} subjects".format(len(subj_ids)))

    # Set up feature extractors
    names_audio,counts_audio = generate_feature_name_vec(cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands)
    n_audio_feat = len(names_audio)
    names_imu,counts_imu = generate_feature_name_vec_imu(cfg.feature_extraction.DP_epsilon)
    n_imu_feat = len(names_imu) 

    # FEATURE EXTRACTION
    if cfg.ml.steps.extract_features:
        print("********Extracting features********")
        subprocess.run(["python", "extract_features.py"])

        if cfg.ml.coughvid_ssl:
            print("********Extracting COUGHVID features********")
            subprocess.run(["python", "extract_coughvid_features.py"])

    if cfg.ml.coughvid_ssl:
        cvid_dirs = os.listdir(coughvid_features_folder)
        cvid_dirs = np.delete(cvid_dirs,np.where(np.in1d(cvid_dirs,"audio_in_names.npy")))

    # Load extracted features
    assert os.path.exists(features_folder), "Error: Folder {0} does not exist. Feature extraction needs to be run".format(features_folder) 
    if cfg.ml.coughvid_ssl:
        assert os.path.exists(coughvid_features_folder), "Error: Folder {0} does not exist. COUGHVID feature extraction needs to be run".format(coughvid_features_folder) 

    X, y_t, subjects, feat_names,  trial_logs, mov_logs, noise_logs = get_data_matrices(subj_ids, cfg.data_folder, features_folder, cfg.experiment_conditions.conditions_train, n_audio_feat,n_imu_feat,cfg.signals.in_mic_sel,cfg.signals.out_mic_sel,cfg.signals.imu_sel, cfg.signals.biodata_sel)
    y = (y_t>0)

    # Set up nested CV with an inner and outer loop
    np.random.seed(0)
    scoring_method = "average_precision"
    cv_outer = GroupKFold(n_splits=cfg.ml.n_splits_outer)
    cv_inner = GroupKFold(n_splits=cfg.ml.n_splits_inner)

    # Run over all CV splits
    for j, (train_index, test_index) in enumerate(cv_outer.split(X, y, subjects)):
        test_subj_id = subj_ids[int(np.unique(subjects[test_index])[0])]
        print("CV iteration {0}".format(j))

        # Set up output directory for each fold and keep track of which subject is used for testing
        out_folder_fold = output_folder + "/fold_" + str(j)  + "/"
        if not os.path.exists(out_folder_fold):
            os.makedirs(out_folder_fold)
        with open(out_folder_fold + "test_subject.txt", 'a') as file:
            file.write(test_subj_id)

        X_train = X[train_index,:]
        y_train = y[train_index]
        subjects_train = subjects[train_index]

        # MODEL SELECTION WITH NESTED CROSS-VALIDATION
        if cfg.ml.steps.select_model:
            print("Comparing ML models")

            # Compare accuracy of different ML models with nested cross-validation
            means = np.zeros(len(cfg.ml.model_names))
            stds = np.zeros(len(cfg.ml.model_names))
            for i, mn in enumerate(cfg.ml.model_names):
                mean_score, std_score, _ = train_model_untuned(X_train, y_train, subjects_train, cv_inner, mn, scoring_method, scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
                print("Score for {0}: {1}+/-{2}".format(mn, mean_score, std_score))
                means[i] = mean_score
                stds[i] = std_score
            
            # Save results in output directory
            df_results = pd.DataFrame()
            df_results["model"] = cfg.ml.model_names
            df_results["mean"] = means
            df_results["std"] = stds
            df_results.to_csv(out_folder_fold + "model_comparison_results.csv")

            chosen_model = np.array(cfg.ml.model_names)[np.argmax(means)]
        else: # Use a pre-defined model from a config file
            chosen_model = cfg.ml.model_name

        if cfg.ml.tune_model_hyps:
            # Tune chosen model's hyperparameters
            tuned_result , model_hyps = tuneHyperparameters(X_train, y_train, subjects_train, cv_inner, [chosen_model], scoring_method)
            print("Score for {0} post tuning: {1}".format(chosen_model, tuned_result[0]))
            
            # Train chosen ML model
            model = generate_pipeline(chosen_model, model_hyps[0], scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
        else:
            model = generate_pipeline(chosen_model, scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)

        # RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION
        if cfg.ml.steps.rfecv:
            print("Running RFECV")

            # Select optimal number of features
            rfe = RFECV(estimator=model, step=cfg.ml.step_size, cv=cv_inner, scoring=scoring_method)
            rfe.fit(X_train, y_train, groups=subjects_train)

            n_features_opt = rfe.n_features_
            print("Optimal number of features: {0}".format(n_features_opt))
            top_n = rfe.support_

            # Reduce number of features to a fixed value to trade accuracy for energy efficiency
            if cfg.ml.overwrite_n_features_opt:
                n_features_opt = cfg.ml.n_features_opt

            # Observe effects of feature reduction on performance
            feature_numbers = np.arange(X_train.shape[1],0,-cfg.ml.step_size)[::-1]
            if feature_numbers[0] != 1:
                feature_numbers = np.concatenate(([1], feature_numbers))

            # Save results to file
            mean_scores = rfe.cv_results_["mean_test_score"]
            std_scores = rfe.cv_results_["std_test_score"]
            df_results = pd.DataFrame()
            df_results["number_of_features"] = feature_numbers
            df_results["mean"] = mean_scores
            df_results["std"] = std_scores
            df_results.to_csv(out_folder_fold + "rfecv_results.csv")

            # Save plot to file
            # plt.figure()
            # plt.errorbar(feature_numbers, rfe.cv_results_["mean_test_score"], yerr=rfe.cv_results_["std_test_score"])
            # plt.xlabel("Number of features selected")
            # plt.ylabel("Cross validation" +  scoring_method + "score")
            # plt.savefig(out_folder_fold + "rfecv_plot.png")
        
        else:
            n_features_opt = cfg.ml.n_features_opt


        # FEATURE SELECTION
        if cfg.ml.steps.feature_selection: 
            print("Selecting most significant features")

            # Don't use the optimal features found in RFECV; run another cross-validation loop
            if ~(cfg.ml.steps.rfecv) | (cfg.ml.overwrite_n_features_opt):

                feature_importances = np.zeros((cfg.ml.n_splits_inner,len(feat_names)))

                for i, (train_index_inner, test_index_inner) in enumerate(cv_inner.split(X_train, y_train, subjects_train)):
                    # Split testing and training for inner loop
                    X_train_inner = X_train[train_index_inner,:]
                    y_train_inner = y_train[train_index_inner]
                    
                    # Train model
                    model = generate_pipeline(chosen_model, scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
                    model.fit(X_train_inner, y_train_inner)
                    
                    feature_importances[i,:] = model['classifier'].feature_importances_
                
                # Get the top features (average over folds of feature importances)
                feat_importance_sums = np.sum(feature_importances,axis=1).reshape(-1,1)
                feat_imp_norm = feature_importances / feat_importance_sums
                feat_imp_means = np.mean(feat_imp_norm,axis=0)
                nth_val = np.sort(feat_imp_means)[::-1][n_features_opt-1]
                top_n = feat_imp_means>=nth_val

            # Keep track of which features from each sensor are selected
            feat_select = {}
            feat_ndx = 0
            if cfg.signals.in_mic_sel:
                feat_select["in_mic"] = top_n[feat_ndx:n_audio_feat+feat_ndx]
                feat_ndx += n_audio_feat
            if cfg.signals.out_mic_sel:
                feat_select["out_mic"] = top_n[feat_ndx:n_audio_feat+feat_ndx]
                feat_ndx += n_audio_feat
            if cfg.signals.imu_sel:
                feat_select["imu"] = top_n[feat_ndx:n_imu_feat+feat_ndx]
                feat_ndx += n_imu_feat
            if cfg.signals.biodata_sel:
                feat_select["biodata"] = top_n[feat_ndx:]

            # Save selected features
            pickle.dump(feat_select, open(out_folder_fold + 'feat_select.pkl', 'wb'))
        # else:
        #     # TODO: Load feature selection from somewhere
        #     continue
    
        # TRAINING
        if cfg.ml.steps.train:
            print("Training model")

            # Reset feature extractors with reduced feature sets
            if cfg.ml.steps.feature_selection:
                required_biodata = feat_select["biodata"]
                # If using COUGHVID for pre-training, remove BMI feature
                if cfg.ml.coughvid_ssl & cfg.signals.out_mic_sel:
                    required_biodata[1] = False
                feat_select_audio_in = feat_select["in_mic"] if cfg.signals.in_mic_sel else []
                feat_select_audio_out = feat_select["out_mic"] if cfg.signals.out_mic_sel else []
                feat_select_imu = feat_select["imu"] if cfg.signals.imu_sel else []
                top_n = np.concatenate((feat_select_audio_in,feat_select_audio_out,feat_select_imu,required_biodata))
                audio_feat_extr_out = None if not cfg.signals.out_mic_sel else AudioFeatures(feat_select_audio_out==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
                audio_feat_extr_in = None if not cfg.signals.in_mic_sel else AudioFeatures(feat_select_audio_in==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
                imu_feat_extr = None if not cfg.signals.imu_sel else IMUFeatures(feat_select_imu==1,counts_imu,names_imu,DP_epsilon=cfg.feature_extraction.DP_epsilon)

                X_train = X_train[:,top_n==1]
            else: # Use all features
                feat_select_audio_in = feat_select["in_mic"] if cfg.signals.in_mic_sel else []
                feat_select_audio_out = np.ones(n_audio_feat)
                audio_feat_extr_out = None if not cfg.signals.out_mic_sel else AudioFeatures(feat_select_audio_out==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
                audio_feat_extr_in = None if not cfg.signals.in_mic_sel else AudioFeatures(feat_select_audio_in==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
                feat_select_imu = np.ones(n_imu_feat)
                imu_feat_extr = None if not cfg.signals.imu_sel else IMUFeatures(feat_select_imu==1,counts_imu, names_imu, DP_epsilon=cfg.feature_extraction.DP_epsilon)
                required_biodata = [True, True]
                if cfg.ml.coughvid_ssl & cfg.signals.out_mic_sel:
                    required_biodata[1] = False
            
            # Train model
            if cfg.ml.tune_model_hyps:
                _ , model_hyps = tuneHyperparameters(X_train, y_train, subjects_train, cv_inner, [chosen_model], scoring_method)
                model = generate_pipeline(chosen_model, model_hyps[0], scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
            else:
                model = generate_pipeline(chosen_model, scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
            model.fit(X_train, y_train)
            
            # COUGHVID pre-training
            if cfg.ml.coughvid_ssl & cfg.signals.out_mic_sel:
                model_prob_thresh = 0.7
                feat_select_cvid = np.concatenate((feat_select_audio_out, np.array([required_biodata[0]])))
                # Determine how many coughs we should add to balance the sample
                n_coughs_to_add = int(len(y_train)/2) - sum(y_train)
                # Randomize order of coughvid files
                np.random.shuffle(cvid_dirs)

                # Add features to training set using SSL
                features_to_add = np.zeros((1,X_train.shape[1]))
                n_coughs_added = 0
                sig_idx = 0
                print("Starting COUGHVID SSL")
                # Add features to training data from COUGHVID until you balance the training dataset or run out of COUGHVID data
                while ((n_coughs_added < n_coughs_to_add) & (sig_idx < len(cvid_dirs))):

                    fn = cvid_dirs[sig_idx]
                    feat_array = np.load(coughvid_features_folder + fn + "/feature_array.npy")
                    features_reduced = feat_array[:,feat_select_cvid]
                    feat_arr = get_features_ssl(features_reduced, model, model_prob_thresh)
                    if len(feat_arr) > 0:
                        features_to_add = np.concatenate((features_to_add,feat_arr), axis=0)
                        n_coughs_added += feat_arr.shape[0]
                    sig_idx += 1
                
                features_to_add = np.delete(features_to_add,0,axis=0)
                y_to_add = np.ones(features_to_add.shape[0])
                print("Adding {0} feature vectors to training data".format(len(y_to_add)))
                
                X_ssl = np.concatenate((X_train, features_to_add))
                y_ssl = np.concatenate((y_train, y_to_add))
                
                model = generate_pipeline(chosen_model, scaling=cfg.ml.feature_scaling, smote=cfg.ml.smote)
                model.fit(X_ssl, y_ssl)

            # Save model
            pickle.dump(model, open(out_folder_fold + 'model.pkl', 'wb'))
        else:
            model_folder = cfg.ml.model_feat_folder + "fold_{0}/".format(j)
            model = pickle.load(open(model_folder + "model.pkl", "rb"))
            feat_select = pickle.load(open(model_folder + "feat_select.pkl", "rb"))
            feat_select_audio_in = feat_select["in_mic"] if cfg.signals.in_mic_sel else []
            feat_select_audio_out = feat_select["out_mic"] if cfg.signals.out_mic_sel else []
            feat_select_imu = feat_select["imu"] if cfg.signals.imu_sel else []
            required_biodata = feat_select["biodata"]
            # If using COUGHVID for pre-training, remove BMI feature
            if cfg.ml.coughvid_ssl & cfg.signals.out_mic_sel:
                required_biodata[1] = False
            audio_feat_extr_out = None if not cfg.signals.out_mic_sel else AudioFeatures(feat_select_audio_out==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
            audio_feat_extr_in = None if not cfg.signals.in_mic_sel else AudioFeatures(feat_select_audio_in==1,counts_audio, names_audio, cfg.feature_extraction.n_mfcc, cfg.feature_extraction.psd_bands, cfg.feature_extraction.eepd_bands, cfg.feature_extraction.compute_mfcc)
            imu_feat_extr = None if not cfg.signals.imu_sel else IMUFeatures(feat_select_imu==1,counts_imu,names_imu,DP_epsilon=cfg.feature_extraction.DP_epsilon)
        
        # TESTING
        if cfg.ml.steps.test:
            print("Testing model")

            # Parameters for event-based scoring
            param = scoring.EventScoring.Parameters(cfg.scoring.tolerance_start, cfg.scoring.tolerance_end, cfg.scoring.min_cough_duration/cfg.signals.window_len, cfg.scoring.max_event_duration, cfg.scoring.min_duration_btwn_events)

            # Edge-AI testing emulation
            thresholds_to_test = np.arange(0,1.01,0.01)
            d = []
            iscough = False
            for trial in cfg.experiment_conditions.conditions_test.trials:
                for mov in cfg.experiment_conditions.conditions_test.movements:
                    for noise in cfg.experiment_conditions.conditions_test.noises:
                        for sound in cfg.experiment_conditions.conditions_test.sounds:
                            path = cfg.data_folder + test_subj_id + '/trial_' + trial + '/mov_' + mov + '/background_noise_' + noise + '/' + sound
                            if os.path.exists(path):
                                if (len(os.listdir(path)) > 0) & os.path.isfile(path + '/ground_truth.json'):
                                    iscough = True
                                    fn = path + '/ground_truth.json'
                                    with open(fn, 'rb') as f:
                                        loaded_dict = json.load(f)
                                elif (len(os.listdir(path)) > 0):
                                    iscough = False
                                else:
                                    continue
                                if sum(required_biodata)>0:
                                    fn = cfg.data_folder + test_subj_id + '/biodata.json'
                                    with open(fn, 'rb') as f:
                                        bio = json.load(f)
                                    X_biodata = np.zeros(2)
                                    if bio["Gender"] == "Female":
                                        X_biodata[0] = 1
                                    X_biodata[1] = bio["BMI"]
                                    extra_features = X_biodata[required_biodata]
                                else:
                                    extra_features = None
                                air, skin = load_audio(cfg.data_folder, test_subj_id,  cfg.signals.fs_audio, trial, mov, noise, sound)
                                imu = load_imu(cfg.data_folder, test_subj_id, trial, mov, noise, sound)
                                #final_pred, _ = run_classifier(air, skin, imu, extra_features, model, cfg.signals.window_len, cfg.signals.overlap_test,feat_extr_audio_in = audio_feat_extr_in, feat_extr_audio_out = audio_feat_extr_out, feat_extr_imu = imu_feat_extr)
                                if (cfg.signals.out_mic_sel) | (cfg.signals.in_mic_sel):
                                    segment_indices_list, peak_locs_list, peak_amp_list, model_confidences =  run_classifier_with_postprocessing(air, skin, imu, extra_features, cfg.signals.fs_audio, cfg.postprocessing.fs_downsample, cfg.postprocessing.cough_end_tolerance,model, cfg.signals.window_len, cfg.signals.overlap_test,feat_extr_audio_in = audio_feat_extr_in, feat_extr_audio_out = audio_feat_extr_out, feat_extr_imu = imu_feat_extr)
                                else:
                                    final_pred, _ = run_classifier(air, skin, imu, extra_features, model, cfg.signals.window_len, cfg.signals.overlap_test,feat_extr_audio_in = audio_feat_extr_in, feat_extr_audio_out = audio_feat_extr_out, feat_extr_imu = imu_feat_extr)
                                    model_confidences = 0
                                if (model_confidences is not None):
                                    #performance metrics for each threshold
                                    for i, thresh in enumerate(thresholds_to_test):
                                        if (cfg.signals.out_mic_sel) | (cfg.signals.in_mic_sel): # Audio data included --> use postprocessing
                                            segment_indices_final, _ = clean_cough_segments(segment_indices_list[model_confidences>=thresh], peak_locs_list[model_confidences>=thresh], peak_amp_list[model_confidences>=thresh], cfg.signals.fs_audio)
                                            if len(segment_indices_final)>0:
                                                pred_region =  get_ground_truth_regions(imu.x,segment_indices_final[:,0]/cfg.signals.fs_audio,segment_indices_final[:,1]/cfg.signals.fs_audio)
                                            else:
                                                pred_region = np.zeros(imu.x.shape)
                                            if iscough:
                                                gt =  get_ground_truth_regions(imu.x,loaded_dict["start_times"],loaded_dict["end_times"])
                                            else:
                                                gt = np.zeros(imu.x.shape)
                                        else: # only IMU data
                                            if final_pred is not None:
                                                pred_region = final_pred>=thresh
                                                if iscough:
                                                    gt =  get_ground_truth_regions(imu.x,loaded_dict["start_times"],loaded_dict["end_times"])
                                                    gt = gt[:len(final_pred)]
                                                else:
                                                    gt = np.zeros(len(final_pred))
                                        labels = Annotation(gt, FS_IMU)
                                        pred = Annotation(pred_region ,FS_IMU)
                                        scores_dur = scoring.SampleScoring(labels, pred, FS_IMU)
                                        scores_evt = scoring.EventScoring(labels, pred, param)
                                        d.append({"subject": test_subj_id, "trial": trial, "movement": mov, "noise":noise, "sound":sound, "threshold":thresh, "true_dur":scores_dur.refTrue, "pred_dur":scores_dur.tp + scores_dur.fp, "tp_dur":scores_dur.tp, "fp_dur": scores_dur.fp, "se_dur":scores_dur.sensitivity, "ppv_dur":scores_dur.precision, "f1_dur": scores_dur.f1, "tp_evt":scores_evt.tp, "fp_evt":scores_evt.fp, "fn_evt":scores_evt.refTrue - scores_evt.tp, "se_evt":scores_evt.sensitivity, "ppv_evt": scores_evt.precision, "f1_evt": scores_evt.f1})
            df = pd.DataFrame(d)
            df.to_csv(out_folder_fold + "results.csv")
    
    print("DONE!!")


if __name__ == "__main__":
    run_config()





