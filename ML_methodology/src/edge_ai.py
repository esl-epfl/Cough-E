"""Functions relating to edge-AI testing implementation"""

from .helpers import *
from .feature_tree import *
from .postprocessing import *
import json
from sklearn.metrics import roc_auc_score, confusion_matrix


def get_features_ssl(feature_array, classifier, thresh=0.5):
    """
    Run classifier through a signal and return feaure vectors of signal windows for which the classifier outputs a probability greater than the threshold 
    Inputs:
    - feature_array: LxN matrix of feature vectors extracted from the COUGHVID dataset
    - classifier: model used for prediction
    - thresh: classifier probability threshold above which we add the signal to our dataset
    Outputs:
    - output_features: MxN output vector of features for M feature vectors producing classifier probability >= thresh
    """

    # Run model across feature vectors and save vectors with positive predictions
    output_feature_array = []
    for feature_vector in feature_array:
        try:
            output_prob = classifier.predict_proba(feature_vector.reshape(1,-1))
            if (output_prob[0][1] >= thresh):
                output_feature_array.append(feature_vector)
        except:
            continue
                
    return np.array(output_feature_array)

def report_sota_metrics(y_true, y_prob, thresh, window_len):
    """Report state-of-the-art sample-based metrics for a given classifier"""
    auc = roc_auc_score(y_true,y_prob)
    y_pred = y_prob >= thresh
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = (tp)/(tp+fn)
    sp = (tn)/(tn+fp)
    acc = (tp+tn)/(tp+tn+fp+fn)
    pr = (tp)/(tp+fp)
    f1 = (2*se*pr)/(se + pr)
    npv = (tn)/(tn+fn)
    fphr = fp/(window_len*len(y_true)/3600)
    return tn, fp, fn, tp, se, sp, acc, pr, f1, auc, npv, fphr

def run_classifier(air, skin, imu, extra_features, classifier, window_len, overlap, feat_extr_audio_in=None, feat_extr_audio_out=None, feat_extr_imu=None):
    """
    Run classifier across signal in an edge-AI fashion, averaging the classifier output probabilities for each overlapping window segment
    Inputs:
    - air: outward-facing mic signal
    - skin: inner-facing mic signal
    - feature_extr_audio_in: feature extractor for the inner microphone signal
    - feature_extr_audio_out: feature extractor for the outer microphone signal
    - feature_extr_imu: feature extractor for the imu signals
    - extra_features: biodata features (bmi and gender)
    - classifier: model used for prediction
    - window_len: classifier window length in seconds
    - overlap: integer 0-99, percentage of overlap between one window and the next
    Outputs:
    - output_prob: Mx1 output vector of predicted probabilities at each input index (except the end)
    - model_out: output probability vector at each run of the model (i.e. not taking overlaps into account)
    """
    window_samp = int(window_len*FS_AUDIO)
    overlap_samp = int(window_samp*overlap/100)
    step_samp = int(np.rint(window_samp*(1 - overlap/100)))
    n_overlaps = int(window_samp/step_samp -1)

    window_samp_imu = int(window_len*FS_IMU)
    step_samp_imu = int(np.rint(window_samp_imu*(1 - overlap/100)))
    
    if imu is not None:

        imu_data = np.stack((imu.x,imu.y,imu.z,imu.Y,imu.P,imu.R),axis=1)

        n_runs = min(int(len(air)/step_samp), int(len(imu.x)/step_samp_imu)) - n_overlaps

        imu = imu_data
    
    else:
        
        n_runs = int(len(air)/step_samp) - n_overlaps
        
    
    if n_runs>0:

        # Run model across signal with a given overlap and save predictions
        model_out = np.zeros((n_runs,1))
        indices_out = np.zeros((n_runs, window_samp_imu))
        for j in range(n_runs):
            feat_array = []
            if feat_extr_audio_out is not None:
                seg_out = air[j*step_samp:j*step_samp+window_samp]
                feats_audio_out, _ = feat_extr_audio_out.compute_features(seg_out,FS_AUDIO)
            if feat_extr_audio_in is not None:
                seg_in = skin[j*step_samp:j*step_samp+window_samp]
                feats_audio_in, _ = feat_extr_audio_in.compute_features(seg_in,FS_AUDIO)

            if feat_extr_imu is not None:
                seg_imu = imu[j*step_samp_imu:j*step_samp_imu+window_samp_imu,:]
                feats_imu = []
                for i, signal in enumerate(IMU_Signal):
                    if i<6:
                        sig = seg_imu[:,i]
                    elif i==6:
                        sig = np.linalg.norm((seg_imu[:,0],seg_imu[:,1],seg_imu[:,2]), axis=0)
                    elif i==7:
                        sig = np.linalg.norm((seg_imu[:,3],seg_imu[:,4],seg_imu[:,5]), axis=0)
                    feats, names = feat_extr_imu.compute_features(sig,FS_IMU,signal)
                    for feat in feats:
                        feats_imu.append(feat)
                feats_imu = np.array(feats_imu)

            if feat_extr_audio_in is not None:
                feat_array = np.concatenate((feat_array,feats_audio_in))
            if feat_extr_audio_out is not None:
                feat_array = np.concatenate((feat_array,feats_audio_out))
            if feat_extr_imu is not None:
                feat_array = np.concatenate((feat_array,feats_imu))
            if extra_features is not None:
                feat_array = np.concatenate((feat_array,extra_features))
            feat_array = np.array(feat_array).reshape(1, -1)
            output_prob = classifier.predict_proba(feat_array)
            model_out[j] = output_prob[0][1]
            indices_out[j,:] = np.arange(j*step_samp_imu,j*step_samp_imu+window_samp_imu)

        # Average probabilities for each overlapping segment
        last_imu_index = int(indices_out[-1,-1])
        output_prob = np.zeros(last_imu_index+1)
        for i in range(last_imu_index+1):
            output_prob[i] = np.mean(model_out[np.any(indices_out==i,axis=1)])
    else:
        output_prob = None
        model_out = None
    return output_prob, model_out

def run_classifier_with_postprocessing(air, skin, imu, extra_features, fs_audio, fs_downsample, tolerance_multiplier, classifier, window_len, overlap, feat_extr_audio_in=None, feat_extr_audio_out=None, feat_extr_imu=None):
    """
    Run classifier across signal in an edge-AI fashion, averaging the classifier output probabilities for each overlapping window segment
    Inputs:
    - air: outward-facing mic signal
    - skin: inner-facing mic signal
    - feature_extr_audio_in: feature extractor for the inner microphone signal
    - feature_extr_audio_out: feature extractor for the outer microphone signal
    - feature_extr_imu: feature extractor for the imu signals
    - extra_features: biodata features (bmi and gender)
    - classifier: model used for prediction
    - window_len: classifier window length in seconds
    - overlap: integer 0-99, percentage of overlap between one window and the next
    Outputs:
    - output_prob: Mx1 output vector of predicted probabilities at each input index (except the end)
    - model_out: output probability vector at each run of the model (i.e. not taking overlaps into account)
    """
    window_samp = int(window_len*fs_audio)
    step_samp = int(np.rint(window_samp*(1 - overlap/100)))
    n_overlaps = int(window_samp/step_samp -1)

    window_samp_imu = int(window_len*FS_IMU)
    step_samp_imu = int(np.rint(window_samp_imu*(1 - overlap/100)))
    
    if imu is not None:

        imu_data = np.stack((imu.x,imu.y,imu.z,imu.Y,imu.P,imu.R),axis=1)

        n_runs = min(int(len(air)/step_samp), int(len(imu.x)/step_samp_imu)) - n_overlaps

        imu = imu_data
    
    else:
        
        n_runs = int(len(air)/step_samp) - n_overlaps

    # Arrays of cough locations and peaks
    segment_indices_list = []
    peak_locs_list = []
    peak_amp_list = []
    model_confidences = []

        
    if n_runs>0:

        # Run model across signal with a given overlap and save predictions
        for j in range(n_runs):
            feat_array = []
            if feat_extr_audio_out is not None:
                seg_out = air[j*step_samp:j*step_samp+window_samp]
                feats_audio_out, _ = feat_extr_audio_out.compute_features(seg_out,fs_audio)
            if feat_extr_audio_in is not None:
                seg_in = skin[j*step_samp:j*step_samp+window_samp]
                feats_audio_in, _ = feat_extr_audio_in.compute_features(seg_in,fs_audio)
            if feat_extr_imu is not None:
                seg_imu = imu[j*step_samp_imu:j*step_samp_imu+window_samp_imu,:]
                feats_imu = []
                for i, signal in enumerate(IMU_Signal):
                    if i<6:
                        sig = seg_imu[:,i]
                    elif i==6:
                        sig = np.linalg.norm((seg_imu[:,0],seg_imu[:,1],seg_imu[:,2]), axis=0)
                    elif i==7:
                        sig = np.linalg.norm((seg_imu[:,3],seg_imu[:,4],seg_imu[:,5]), axis=0)
                    feats, _ = feat_extr_imu.compute_features(sig,FS_IMU,signal)
                    for feat in feats:
                        feats_imu.append(feat)
                feats_imu = np.array(feats_imu)

            if feat_extr_audio_in is not None:
                feat_array = np.concatenate((feat_array,feats_audio_in))
            if feat_extr_audio_out is not None:
                feat_array = np.concatenate((feat_array,feats_audio_out))
            if feat_extr_imu is not None:
                feat_array = np.concatenate((feat_array,feats_imu))
            if extra_features is not None:
                feat_array = np.concatenate((feat_array,extra_features))
            feat_array = np.array(feat_array).reshape(1, -1)
            output_prob = classifier.predict_proba(feat_array)
            model_confidence = output_prob[0][1]

            # Post-process segment
            if feat_extr_audio_out is not None:
                segment_indices, peak_locs, peaks = get_cough_peaks(seg_out,fs_audio, fs_downsample, tolerance_multiplier)
            elif feat_extr_audio_in is not None:
                segment_indices, peak_locs, peaks = get_cough_peaks(seg_in,fs_audio, fs_downsample, tolerance_multiplier)
            for si, pl, p in zip(np.array(segment_indices), np.array(peak_locs), np.array(peaks)):
                segment_indices_list.append(si+(j*step_samp))
                peak_locs_list.append(pl+(j*step_samp))
                peak_amp_list.append(p)
                model_confidences.append(model_confidence)
        segment_indices_list = np.array(segment_indices_list)
        peak_locs_list = np.array(peak_locs_list)
        peak_amp_list = np.array(peak_amp_list)
        model_confidences = np.array(model_confidences)

    else:
        segment_indices_list = peak_locs_list = peak_amp_list = model_confidences = None
    
    return segment_indices_list, peak_locs_list, peak_amp_list, model_confidences

def get_data_matrices(subj_train, dataset_folder, features_folder, experiment_conditions, n_audio_features,n_imu_features, in_mic = True, out_mic = True, imu=True, biodata = True):
    """
    Load extracted feature data matrices and compile them into one matrix for an array of subjects (specified by subj_train).
    """
    if (in_mic & out_mic & imu & biodata):
        X_train = np.zeros((1,n_audio_features*2 + n_imu_features + 2))
    elif (in_mic & out_mic & imu):
        X_train = np.zeros((1,n_audio_features*2 + n_imu_features))
    elif (out_mic & imu & biodata) | (in_mic & imu & biodata):
        X_train = np.zeros((1,n_audio_features + n_imu_features + 2))
    elif (out_mic & imu) | (in_mic & imu):
        X_train = np.zeros((1,n_audio_features + n_imu_features))
    elif (out_mic & biodata ) | (in_mic & biodata ):
        X_train = np.zeros((1,n_audio_features + 2))
    elif (out_mic) | (in_mic):
        X_train = np.zeros((1,n_audio_features))
    elif (imu & biodata):
        X_train = np.zeros((1,n_imu_features + 2))
    elif (imu):
        X_train = np.zeros((1,n_imu_features))
    else: 
        print("ERROR invalid combination of sensors")
    y_train = np.zeros(1)
    subjects_train = np.zeros(1)
    subj_count = 0
    trial_logs = np.array([])
    mov_logs = np.array([])
    noise_logs = np.array([])
    for s in subj_train:
        out_subj_folder = features_folder + s + "/"
        if in_mic:
            X_audio_in = np.load(out_subj_folder + "audio_in_features.npy")
            n_samples = X_audio_in.shape[0]
        
        if (in_mic | out_mic):
            if out_mic:
                names_audio = np.load(out_subj_folder + "audio_out_names.npy")
            else:
                names_audio = np.load(out_subj_folder + "audio_in_names.npy")
            names_audio_out = np.array([feat + "_out" for feat in names_audio])
            names_audio_in = np.array([feat + "_in" for feat in names_audio])
        
        if out_mic:
            X_audio_out = np.load(out_subj_folder + "audio_out_features.npy")
            n_samples = X_audio_out.shape[0]
        if imu:
            X_imu = np.load(out_subj_folder + "imu_features.npy")
            names_imu = np.load(out_subj_folder + "imu_names.npy")
            n_samples = X_imu.shape[0]
        
        labels = np.load(out_subj_folder + "labels.npy")
        
        
        fn = dataset_folder + s + '/biodata.json'
        trial_log = np.load(out_subj_folder + "trial_log.npy")
        mov_log = np.load(out_subj_folder + "mov_log.npy")
        noise_log = np.load(out_subj_folder + "noise_log.npy")
        
        
        if biodata:
            X_biodata = np.zeros((n_samples,2))
            with open(fn, 'rb') as f:
                bio = json.load(f)
            if bio["Gender"] == "Female":
                X_biodata[:,0] = 1
            X_biodata[:,1] = bio["BMI"]
            names_biodata = np.array(["gender","bmi"])
        if (in_mic & out_mic & imu & biodata):
            X = np.concatenate((X_audio_in, X_audio_out, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in, names_audio_out, names_imu, names_biodata))
        elif (in_mic & out_mic & imu):
            X = np.concatenate((X_audio_in, X_audio_out, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_in, names_audio_out, names_imu))
        elif (out_mic & imu & biodata):
            X = np.concatenate((X_audio_out, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_out, names_imu, names_biodata))
        elif (out_mic & imu):
            X = np.concatenate((X_audio_out, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_out, names_imu))
        elif (in_mic & imu & biodata):
            X = np.concatenate((X_audio_in, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in, names_imu, names_biodata))
        elif (in_mic & imu):
            X = np.concatenate((X_audio_in, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_in, names_imu))
        elif (out_mic & biodata):
            X = np.concatenate((X_audio_out, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_out,names_biodata))
        elif (out_mic):
            X = X_audio_out
            feat_names = names_audio_out
        elif (in_mic & biodata):
            X = np.concatenate((X_audio_in, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in,names_biodata))
        elif (in_mic):
            X = X_audio_in
            feat_names = names_audio_in
        elif (imu & biodata):
            X =  np.concatenate((X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_imu,names_biodata))
        elif (imu):
            X =  X_imu
            feat_names = names_imu
        
        # Remove samples from undesired experimental conditions
        desired_trials = np.in1d(trial_log,[tr for tr in experiment_conditions.trials])
        desired_movs = np.in1d(mov_log,[mv for mv in experiment_conditions.movements])
        desired_noises = np.in1d(noise_log,[ns for ns in experiment_conditions.noises])
        sounds = [sd for sd in experiment_conditions.sounds]
        sound_label_vec = []
        if np.in1d(Sound.COUGH.value,sounds):
            sound_label_vec.append(0)
            sound_label_vec.append(1)
        if np.in1d(Sound.LAUGH.value,sounds):
            sound_label_vec.append(-1)
        if np.in1d(Sound.BREATH.value,sounds):
            sound_label_vec.append(-2)
        if np.in1d(Sound.THROAT.value,sounds):
            sound_label_vec.append(-3)
        if np.in1d(Sound.TALK.value,sounds):
            sound_label_vec.append(-4)
        desired_sounds = np.in1d(labels,sound_label_vec)
        desired_samples = desired_trials & desired_movs & desired_noises & desired_sounds
        X = X[desired_samples,:]
        labels = labels[desired_samples]
        trial_log = trial_log[desired_samples]
        mov_log = mov_log[desired_samples]
        noise_log = noise_log[desired_samples]
            
        X_train = np.concatenate((X_train,X), axis=0)
        y_train = np.concatenate((y_train,labels))
        subjects_train = np.concatenate((subjects_train,np.tile(subj_count,X.shape[0])))
        
        trial_logs = np.concatenate((trial_logs,trial_log))
        mov_logs = np.concatenate((mov_logs,mov_log))
        noise_logs = np.concatenate((noise_logs,noise_log))
        subj_count += 1
    X_train = np.delete(X_train,0,axis=0)
    y_train = np.delete(y_train,0)
    subjects_train = np.delete(subjects_train,0)
    return X_train, y_train, subjects_train, feat_names, trial_logs, mov_logs, noise_logs

def get_data_matrices_no_hydra(subj_train, dataset_folder, features_folder, experiment_conditions, n_audio_features,n_imu_features, in_mic = True, out_mic = True, imu=True, biodata = True):
    """
    Load extracted feature data matrices and compile them into one matrix for an array of subjects (specified by subj_train).
    """
    if (in_mic & out_mic & imu & biodata):
        X_train = np.zeros((1,n_audio_features*2 + n_imu_features + 2))
    elif (in_mic & out_mic & imu):
        X_train = np.zeros((1,n_audio_features*2 + n_imu_features))
    elif (out_mic & imu & biodata) | (in_mic & imu & biodata):
        X_train = np.zeros((1,n_audio_features + n_imu_features + 2))
    elif (out_mic & imu) | (in_mic & imu):
        X_train = np.zeros((1,n_audio_features + n_imu_features))
    elif (out_mic & biodata ) | (in_mic & biodata ):
        X_train = np.zeros((1,n_audio_features + 2))
    elif (out_mic) | (in_mic):
        X_train = np.zeros((1,n_audio_features))
    elif (imu & biodata):
        X_train = np.zeros((1,n_imu_features + 2))
    elif (imu):
        X_train = np.zeros((1,n_imu_features))
    else: 
        print("ERROR invalid combination of sensors")
    y_train = np.zeros(1)
    subjects_train = np.zeros(1)
    subj_count = 0
    trial_logs = np.array([])
    mov_logs = np.array([])
    noise_logs = np.array([])
    for s in subj_train:
        out_subj_folder = features_folder + s + "/"
        if in_mic:
            X_audio_in = np.load(out_subj_folder + "audio_in_features.npy")
            n_samples = X_audio_in.shape[0]
        
        if (in_mic | out_mic):
            if out_mic:
                names_audio = np.load(out_subj_folder + "audio_out_names.npy")
            else:
                names_audio = np.load(out_subj_folder + "audio_in_names.npy")
            names_audio_out = np.array([feat + "_out" for feat in names_audio])
            names_audio_in = np.array([feat + "_in" for feat in names_audio])
        
        if out_mic:
            X_audio_out = np.load(out_subj_folder + "audio_out_features.npy")
            n_samples = X_audio_out.shape[0]
        if imu:
            X_imu = np.load(out_subj_folder + "imu_features.npy")
            names_imu = np.load(out_subj_folder + "imu_names.npy")
            n_samples = X_imu.shape[0]
        
        labels = np.load(out_subj_folder + "labels.npy")
        
        
        fn = dataset_folder + s + '/biodata.json'
        trial_log = np.load(out_subj_folder + "trial_log.npy")
        mov_log = np.load(out_subj_folder + "mov_log.npy")
        noise_log = np.load(out_subj_folder + "noise_log.npy")
        
        
        if biodata:
            X_biodata = np.zeros((n_samples,2))
            with open(fn, 'rb') as f:
                bio = json.load(f)
            if bio["Gender"] == "Female":
                X_biodata[:,0] = 1
            X_biodata[:,1] = bio["BMI"]
            names_biodata = np.array(["gender","bmi"])
        if (in_mic & out_mic & imu & biodata):
            X = np.concatenate((X_audio_in, X_audio_out, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in, names_audio_out, names_imu, names_biodata))
        elif (in_mic & out_mic & imu):
            X = np.concatenate((X_audio_in, X_audio_out, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_in, names_audio_out, names_imu))
        elif (out_mic & imu & biodata):
            X = np.concatenate((X_audio_out, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_out, names_imu, names_biodata))
        elif (out_mic & imu):
            X = np.concatenate((X_audio_out, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_out, names_imu))
        elif (in_mic & imu & biodata):
            X = np.concatenate((X_audio_in, X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in, names_imu, names_biodata))
        elif (in_mic & imu):
            X = np.concatenate((X_audio_in, X_imu), axis=1)
            feat_names = np.concatenate((names_audio_in, names_imu))
        elif (out_mic & biodata):
            X = np.concatenate((X_audio_out, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_out,names_biodata))
        elif (out_mic):
            X = X_audio_out
            feat_names = names_audio_out
        elif (in_mic & biodata):
            X = np.concatenate((X_audio_in, X_biodata), axis=1)
            feat_names = np.concatenate((names_audio_in,names_biodata))
        elif (in_mic):
            X = X_audio_in
            feat_names = names_audio_in
        elif (imu & biodata):
            X =  np.concatenate((X_imu, X_biodata), axis=1)
            feat_names = np.concatenate((names_imu,names_biodata))
        elif (imu):
            X =  X_imu
            feat_names = names_imu
        
        # Remove samples from undesired experimental conditions
        desired_trials = np.in1d(trial_log,[tr for tr in experiment_conditions["trials"]])
        desired_movs = np.in1d(mov_log,[mv for mv in experiment_conditions["movements"]])
        desired_noises = np.in1d(noise_log,[ns for ns in experiment_conditions["noises"]])
        sounds = [sd for sd in experiment_conditions["sounds"]]
        sound_label_vec = []
        if np.in1d(Sound.COUGH.value,sounds):
            sound_label_vec.append(0)
            sound_label_vec.append(1)
        if np.in1d(Sound.LAUGH.value,sounds):
            sound_label_vec.append(-1)
        if np.in1d(Sound.BREATH.value,sounds):
            sound_label_vec.append(-2)
        if np.in1d(Sound.THROAT.value,sounds):
            sound_label_vec.append(-3)
        if np.in1d(Sound.TALK.value,sounds):
            sound_label_vec.append(-4)
        desired_sounds = np.in1d(labels,sound_label_vec)
        desired_samples = desired_trials & desired_movs & desired_noises & desired_sounds
        X = X[desired_samples,:]
        labels = labels[desired_samples]
        trial_log = trial_log[desired_samples]
        mov_log = mov_log[desired_samples]
        noise_log = noise_log[desired_samples]
            
        X_train = np.concatenate((X_train,X), axis=0)
        y_train = np.concatenate((y_train,labels))
        subjects_train = np.concatenate((subjects_train,np.tile(subj_count,X.shape[0])))
        
        trial_logs = np.concatenate((trial_logs,trial_log))
        mov_logs = np.concatenate((mov_logs,mov_log))
        noise_logs = np.concatenate((noise_logs,noise_log))
        subj_count += 1
    X_train = np.delete(X_train,0,axis=0)
    y_train = np.delete(y_train,0)
    subjects_train = np.delete(subjects_train,0)
    return X_train, y_train, subjects_train, feat_names, trial_logs, mov_logs, noise_logs

def get_ground_truth_regions(signal,starts, ends):
    gt = np.zeros(signal.shape)
    for (s,e) in zip(starts,ends):
        gt[int(round(s*FS_IMU)):int(round(e*FS_IMU))] = 1
        if (int(round(s*FS_IMU)) > 0):
            #make sure that the index before a start is always zero to account for overlapping coughs
            gt[int(round(s*FS_IMU))-1] = 0
    return gt