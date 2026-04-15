#include <stdio.h>
#include <inttypes.h>

#include <time_domain_feat.h>
#include <feature_extraction.h>
#include <frequency_features.h>
#include <azc.h>
#include <helpers.h>

#include <audio_features.h>
#include <imu_features.h>

#ifdef FXP_MODE
#include <core/fxp_convert.h>
#endif
#include <range_analysis.h>

#ifdef RANGE_ANALYSIS
const char *_ra_imu_signal_ctx = "UNKNOWN";
int _ra_imu_active = 0;
#endif
#include <imu/imu_dispatch.h>


//////////////////////////////////////////////////////////////////////////////////
/*                      Local functions declaration                             */
//////////////////////////////////////////////////////////////////////////////////


/**
    Given the features selector vector and two indexes (start and end), it
    returns 1 if feature_selector has at least a 1 in the range specified 
    by the indexes, 0 otherwise

    @param *features_selector   :   one-hot vector for which features to extract
    @param start_index          :   starting index from which to check the `features_selector`
    @param end_index            :   end index to check in the `features_selector`
*/
int is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index);


/**
    Computes the required FFT-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param fs                   :   sampling frequency
    @param *feats               :   array of extracted features
*/
void fft_based_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats);


/**
    Computes the required periodogram-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void periodogram_based_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats);


/**
    Computes the required MFCC features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mfcc_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats);


/**
    Computes the required Mel Spectrogram features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mel_spectrogram_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats);

/**
    Computes the required mean-based features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void mean_based_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats);


/**
    Computes the required EEPD features of the audio signal

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param fs                   :   sampling frequency
    @param *feats               :   array of extracted features
*/
void eepd_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats);


/**
    Process one IMU signal, checks which features are required and computes them

    @param *features_selector   :   one-hot vector for which features to extract
    @param *sig                 :   signal to process
    @param len                  :   length of the signal
    @param *feats               :   array of extracted features
*/
void imu_signal_features(const int8_t *features_selector, float *sig, int16_t len, float *feats);


/**
    This function triggers the feature extraction process for a specific IMU feature family.
    First it checks the the features has to be computed, by means of the features_selector array.
    Then it retrieves the proper data and it calls the feature extraction function.

    The discrimination between different IMU signal here is done through the use of the two
    input parameters "signal_idx" and "sig_feat_idx".

    @param *features_selector   :   one-hot vector for which features to extract
    @param signal               :   signal to process
    @param len                  :   length of the signal
    @param signal_idx           :   index of the specific IMU signal
    @param sig_feat_idx         :   starting index of the features for the IMU signal inside the features_selector vector
    @param *feats               :   array of extracted features
*/
#ifdef FXP_MODE
void compute_imu_family_fxp(const int8_t *features_selector, const q11_5_t signal[][Num_IMU_signals], int16_t len, int8_t signal_idx, int8_t sig_feat_idx, float *feats);
void fxp_convert_imu_inputs(const float sig[][Num_IMU_signals], int16_t len,
                            q11_5_t raw[][Num_IMU_signals], uq10_6_t *l2a, uq5_11_t *l2g);
#endif
void compute_imu_family(const int8_t *features_selector, const float signal[][Num_IMU_signals], int16_t len, int8_t signal_idx, int8_t sig_feat_idx, float *feats);

//////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////
/*                      Local functions definitions                             */
//////////////////////////////////////////////////////////////////////////////////
    

int is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index){

    for(uint16_t i=start_index; i<=end_index; i++){
        if(features_selector[i] == 1){
            return 1;   // It is sufficient to check if one is needed
        }
    }
    return 0;
}


void fft_based_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats){

    // FFT-dependent features' indexes
    // 0  : 6  the singular ones

    if(is_required(features_selector, SPECTRAL_DECREASE, SPECTRAL_SKEW)){

        RA_LOG_ARRAY("AUDIO_FFT", "fft_based_features", "sig_input", sig, len);

        int16_t fft_size = (len / 2) + 1;
        float *magnitudes = (float*) malloc(fft_size * sizeof(float));
        float *frequencies = (float*) malloc(fft_size *sizeof(float));
        float sum_mags = 0.0;

        compute_rfft(sig, len, fs, magnitudes, frequencies, &sum_mags);

        if(features_selector[SPECTRAL_DECREASE]){
            // compute SPEC_DECR
            float spectral_decrease = compute_spec_decrease(magnitudes, frequencies, (len/2)+1, sum_mags);
            feats[SPECTRAL_DECREASE] = spectral_decrease;
        }

        if(features_selector[SPECTRAL_SLOPE]){
            // compute SPEC_SLOPE
            float spectral_slope = compute_spectral_slope(magnitudes, frequencies, (len/2)+1, sum_mags);
            feats[SPECTRAL_SLOPE] = spectral_slope;
        }
        
        if(features_selector[SPECTRAL_ROLLOFF]){
            // compute SPEC_ROLL
            float spectral_rolloff = compute_rolloff(magnitudes, frequencies, (len/2)+1, sum_mags);
            feats[SPECTRAL_ROLLOFF] = spectral_rolloff;
        }
        
        if(is_required(features_selector, SPECTRAL_CENTROID, SPECTRAL_SKEW)){
            // compute SPEC_CENTROID
            float spectral_cetroid = compute_centroid(magnitudes, frequencies, (len/2)+1, sum_mags);

            if(features_selector[SPECTRAL_CENTROID]){
                // append SPEC_CENTROID
                feats[SPECTRAL_CENTROID] = spectral_cetroid;
            }

            if(is_required(features_selector, SPECTRAL_SPREAD, SPECTRAL_SKEW)){
                // compute SPEC_SPREAD
                float spectral_spread = compute_spread(magnitudes, frequencies, (len/2)+1, sum_mags, spectral_cetroid);

                if(features_selector[SPECTRAL_SPREAD]){
                    // append SPEC_SPREAD
                    feats[SPECTRAL_SPREAD] = spectral_spread;
                }

                if(features_selector[SPECTRAL_KURTOSIS]){
                    // compute SPEC_KURT
                    float kurt = compute_kurt(magnitudes, frequencies, (len/2)+1, sum_mags, spectral_cetroid, spectral_spread);
                    feats[SPECTRAL_KURTOSIS] = kurt;
                }

                if(features_selector[SPECTRAL_SKEW]){
                    // compute SPEC_SKEW
                    float skew = compute_skew(magnitudes, frequencies, (len/2)+1, sum_mags, spectral_cetroid, spectral_spread);
                    feats[SPECTRAL_SKEW] = skew;
                }
            }
        }

        free(magnitudes);
        free(frequencies);
    }
}


void periodogram_based_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats){

    // Periodogram dependent features' indexes
    // 7  : 9 for the singular ones
    // 10 : 12 for the PSD ones
    if(is_required(features_selector, SPECTRAL_FLATNESS, POWER_SPECTRAL_DENSITY + N_PSD - 1)){

        RA_LOG_ARRAY("AUDIO_PSD", "periodogram_based_features", "sig_input", sig, len);

        // compute Periodogram
        int16_t psd_size = (NPERSEG / 2) + 1;
        float *psd = (float*)malloc(psd_size * sizeof(float));
        float *freqs = (float*)malloc(psd_size * sizeof(float));
        compute_periodogram(sig, len, fs, psd, freqs);

        if(features_selector[SPECTRAL_FLATNESS]){
           // compute SPEC_FLAT
           float spectral_flatness = compute_flatness(psd, psd_size);
           feats[SPECTRAL_FLATNESS] = spectral_flatness;
        }

        if(features_selector[SPECTRAL_STD]){
            // compute SPEC_STD
            float spectral_std = compute_std(psd, psd_size);
            feats[SPECTRAL_STD] = spectral_std;
        }


        if(features_selector[SPECTRAL_ENTROPY]){
            // compute SPEC_ENTR
            float spectral_entr = compute_spectral_entropy(psd, psd_size);
            feats[SPECTRAL_ENTROPY] = spectral_entr;
        }


        if(features_selector[DOMINANT_FREQUENCY]){
            // compute DOM_FREQ
            float dominant_freq = get_domiant_freq(psd, freqs, psd_size);
            feats[DOMINANT_FREQUENCY] = dominant_freq;
        }

        if(is_required(features_selector, POWER_SPECTRAL_DENSITY, POWER_SPECTRAL_DENSITY + N_PSD - 1)){
            // compute PSD
            float *band_powers = (float*)malloc(N_PSD * sizeof(float));
            normalized_bandpowers(psd, freqs, psd_size, &features_selector[POWER_SPECTRAL_DENSITY], band_powers);
            for(int8_t i=0; i<N_PSD; i++){
                feats[POWER_SPECTRAL_DENSITY + i] = band_powers[i];
            }

            free(band_powers);
        }

        free(psd);
        free(freqs);
    }
}


void mfcc_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats){

    // 13 : 38 for the MFCCs features
    if(is_required(features_selector, MEL_FREQUENCY_CEPSTRAL_COEFFICIENT, ZERO_CROSSING_RATE - 1)){
        // compute MFCCs

        float *mean_mfcc = (float*)malloc(N_MFCC * sizeof(float));
        float *std_mfcc = (float*)malloc(N_MFCC * sizeof(float));
        get_mfcc_features(sig, len, mean_mfcc, std_mfcc);

        // stores first the mean and then the std, one after the other
        for(int16_t i=0; i<N_MFCC; i++){
            feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = mean_mfcc[i];
            feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = std_mfcc[i];
        }


        free(mean_mfcc);
        free(std_mfcc);
    }    
}


void mel_spectrogram_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats){

    if(is_required(features_selector, MEL_FREQUENCY_CEPSTRAL_COEFFICIENT, ZERO_CROSSING_RATE - 1)){

        RA_LOG_ARRAY("AUDIO_MEL", "mel_spectrogram_features", "sig_input", sig, len);

        // compute MEL SPECTROGRAM

        // Indexes of the Mel bins required for the features computation
        uint8_t *idxs_needed = (uint8_t*)malloc(N_MFCC * sizeof(uint8_t));

        // Counts the number of MEL features needed and fills the indexes needed
        uint8_t n_mels_needed = 0;
        for(uint8_t i=0; i<N_MFCC; i++){
            if(
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT+i] == 1) ||               // mean
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT+i+N_MFCC] == 1) ||        // std
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT+i+(2*N_MFCC)] == 1) ||    // max
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT+i+(3*N_MFCC)] == 1)       // entropy
              ){
                idxs_needed[n_mels_needed] = i;
                n_mels_needed++;
            }
        }

        // Arrays to temporary store the features
        float *mean_mel_spectr = (float*)malloc(n_mels_needed * sizeof(float));
        float *std_mel_spectr = (float*)malloc(n_mels_needed * sizeof(float));
        float *max_mel_spectr = (float*)malloc(n_mels_needed * sizeof(float));
        float *entropy_mel_spectr = (float*)malloc(n_mels_needed * sizeof(float));

        get_mel_spectrogram_features(sig, len, idxs_needed, n_mels_needed, mean_mel_spectr, std_mel_spectr, max_mel_spectr, entropy_mel_spectr);

        // stores first the mean, the std, the max and the entropy, one after the other
        int idx = 0;
        for(int16_t i=0; i<N_MFCC; i++){
            if(i == idxs_needed[idx]){  // Only if the feature is one of the needed ones
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = mean_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = std_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2*N_MFCC) + i] = max_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3*N_MFCC) + i] = entropy_mel_spectr[idx];
                idx++;
            }
        }

        free(idxs_needed);
        free(mean_mel_spectr);
        free(std_mel_spectr);
        free(max_mel_spectr);
        free(entropy_mel_spectr);
    }
}


void mean_based_features(const int8_t *features_selector, const float *sig, int16_t len, float *feats){

    // 39 : 41 for the singular ones
    if(is_required(features_selector, ROOT_MEANS_SQUARED, CREST_FACTOR)){

        RA_LOG_ARRAY("AUDIO_FFT", "mean_based_features", "sig_input", sig, len);

        // compute mean
        float *zero_mean = (float *) malloc(len * sizeof(float));  // to store the signal after subtracting the mean
        sub_mean(sig, zero_mean, len);
        RA_LOG_ARRAY("AUDIO_FFT", "mean_based_features", "zero_mean", zero_mean, len);


        if(features_selector[ZERO_CROSSING_RATE]){
            // compute ZCR
            float zcr = compute_zrc(zero_mean, len);
            feats[ZERO_CROSSING_RATE] = zcr;
        }

        if(features_selector[ROOT_MEANS_SQUARED] || features_selector[CREST_FACTOR]){
            // compute RMS
            float rms = get_rms(zero_mean, len);
            RA_LOG_SCALAR("AUDIO_FFT", "audio_rms", "result", rms);

            if(features_selector[ROOT_MEANS_SQUARED]){
                // append RMS
                feats[ROOT_MEANS_SQUARED] = rms;
            }

            if(features_selector[CREST_FACTOR]){
                // compute CREST
                float peak = get_max(zero_mean, len);
                float crest_factor = peak / rms;
                RA_LOG_SCALAR("AUDIO_FFT", "audio_crest", "peak", peak);
                RA_LOG_SCALAR("AUDIO_FFT", "audio_crest", "result", crest_factor);
                feats[CREST_FACTOR] = crest_factor;
            }
        }

        free(zero_mean);
    }
}


void eepd_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats){

    // 42 : 61 for the singular ones
    if(is_required(features_selector, ENERGY_ENVELOPE_PEAK_DETECT, (ENERGY_ENVELOPE_PEAK_DETECT+N_EEPD-1))){    // -1 since it's the last one, otherwise it will check one index more

        int16_t *eepds = (int16_t*)malloc(N_EEPD * sizeof(int16_t));

        // compute EEPD
        eepd(sig, len, fs, &features_selector[ENERGY_ENVELOPE_PEAK_DETECT], eepds);

        for(int16_t i=0; i<N_EEPD; i++){
            feats[ENERGY_ENVELOPE_PEAK_DETECT + i] = eepds[i];
        }

        free(eepds);
    }
}


void imu_signal_features(const int8_t *features_selector, float *sig, int16_t len, float *feats){

    imu_sig_float_t s = { .data = sig, .len = len };
    imu_run_feature_table(features_selector, imu_view_from_float(s), feats);
}


#ifdef FXP_MODE
void fxp_convert_imu_inputs(const float sig[][Num_IMU_signals], int16_t len,
                            q11_5_t raw[][Num_IMU_signals], uq10_6_t *l2a, uq5_11_t *l2g)
{
    for (int16_t i = 0; i < len; i++) {
        // Single conversion boundary from float into fixed-point IMU carriers.
        raw[i][0] = FXP_IMU_RAW_FROM_FLOAT(sig[i][0]);
        raw[i][1] = FXP_IMU_RAW_FROM_FLOAT(sig[i][1]);
        raw[i][2] = FXP_IMU_RAW_FROM_FLOAT(sig[i][2]);
        raw[i][3] = FXP_IMU_RAW_FROM_FLOAT(sig[i][3]);
        raw[i][4] = FXP_IMU_RAW_FROM_FLOAT(sig[i][4]);
        raw[i][5] = FXP_IMU_RAW_FROM_FLOAT(sig[i][5]);

        l2a[i] = fxp_l2_norm_accel_from_raw(raw[i][0], raw[i][1], raw[i][2]);
        l2g[i] = fxp_l2_norm_gyro_from_raw(raw[i][3], raw[i][4], raw[i][5]);
    }
}

void compute_imu_family_fxp(const int8_t *features_selector, const q11_5_t signal[][Num_IMU_signals], int16_t len, int8_t signal_idx, int8_t sig_feat_idx, float *feats){

    if(is_required(features_selector, sig_feat_idx, sig_feat_idx+Num_imu_feat_families-1)){
        q11_5_t *signal_samples = (q11_5_t*)malloc((size_t)len * sizeof(q11_5_t));
        for(int16_t i=0; i<len; i++){
            signal_samples[i] = signal[i][signal_idx];
        }

        imu_sig_raw_t s = { .data = signal_samples, .len = len };
        imu_run_feature_table(&features_selector[sig_feat_idx], imu_view_from_raw(s), &feats[sig_feat_idx]);
        free(signal_samples);
    }
}
#endif

#ifdef RANGE_ANALYSIS
static const char *_imu_signal_names[] = {
    "accel_x", "accel_y", "accel_z", "gyro_y", "gyro_p", "gyro_r"
};
#endif

void compute_imu_family(const int8_t *features_selector, const float signal[][Num_IMU_signals], int16_t len, int8_t signal_idx, int8_t sig_feat_idx, float *feats){

    if(is_required(features_selector, sig_feat_idx, sig_feat_idx+Num_imu_feat_families-1)){

        // Extract samples for the required signal axis
        float *signal_samples = (float*)malloc(len * sizeof(float));
        for(int16_t i=0; i<len; i++){
            signal_samples[i] = signal[i][signal_idx];
        }

        RA_LOG_ARRAY("IMU_RAW", "imu_features", _imu_signal_names[signal_idx], signal_samples, len);

        RA_SET_IMU_CTX("IMU_RAW");
        imu_sig_float_t s = { .data = signal_samples, .len = len };
        imu_run_feature_table(&features_selector[sig_feat_idx], imu_view_from_float(s), &feats[sig_feat_idx]);
        RA_CLEAR_IMU_CTX();
        free(signal_samples);
    }
}


//////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////
/*                      Global functions definitions                            */
//////////////////////////////////////////////////////////////////////////////////

void audio_features(const int8_t *features_selector, const float *sig, int16_t len, int16_t fs, float *feats){

    /* FFT based features */
    fft_based_features(features_selector, sig, len, fs, feats);

    /* Periodogram-based features */
    periodogram_based_features(features_selector, sig, len, fs, feats);

    // /* MFCCs features */
    // mfcc_features(features_selector, sig, len, feats);

    /* MEL SPECTROGRAM features */
    mel_spectrogram_features(features_selector, sig, len, feats);

    /* Mean-based features */
    mean_based_features(features_selector, sig, len, feats);

    /* EEPD features */
    eepd_features(features_selector, sig, len, fs, feats);

}



void imu_features(const int8_t *features_selector, const float sig[][Num_IMU_signals], int16_t len, float *feats){

    // Here len is the IMU_DIM_1 macro in the hardcoded samples

#ifdef FXP_MODE
    q11_5_t (*raw_fxp)[Num_IMU_signals] = (q11_5_t (*)[Num_IMU_signals])malloc((size_t)len * sizeof(*raw_fxp));
    uq10_6_t *combo_l2a = (uq10_6_t*)malloc((size_t)len * sizeof(uq10_6_t));
    uq5_11_t *combo_l2g = (uq5_11_t*)malloc((size_t)len * sizeof(uq5_11_t));

    fxp_convert_imu_inputs(sig, len, raw_fxp, combo_l2a, combo_l2g);

    // Raw-axis families
    compute_imu_family_fxp(features_selector, raw_fxp, len, ACCELEROMETER_X, ACCEL_X_FEAT, feats);
    compute_imu_family_fxp(features_selector, raw_fxp, len, ACCELEROMETER_Y, ACCEL_Y_FEAT, feats);
    compute_imu_family_fxp(features_selector, raw_fxp, len, ACCELEROMETER_Z, ACCEL_Z_FEAT, feats);
    compute_imu_family_fxp(features_selector, raw_fxp, len, GYROSCOPE_Y, GYRO_Y_FEAT, feats);
    compute_imu_family_fxp(features_selector, raw_fxp, len, GYROSCOPE_P, GYRO_P_FEAT, feats);
    compute_imu_family_fxp(features_selector, raw_fxp, len, GYROSCOPE_R, GYRO_R_FEAT, feats);

    // Combined signals
    imu_sig_l2a_t s_l2a = { .data = combo_l2a, .len = len };
    imu_run_feature_table(&features_selector[ACCEL_COMBO], imu_view_from_l2a(s_l2a), &feats[ACCEL_COMBO]);

    imu_sig_l2g_t s_l2g = { .data = combo_l2g, .len = len };
    imu_run_feature_table(&features_selector[GYRO_COMBO], imu_view_from_l2g(s_l2g), &feats[GYRO_COMBO]);

    free(combo_l2g);
    free(combo_l2a);
    free(raw_fxp);
#else
    // ACCEL_X  
    compute_imu_family(features_selector, sig, len, ACCELEROMETER_X, ACCEL_X_FEAT, feats);

    // ACCEL_Y
    compute_imu_family(features_selector, sig, len, ACCELEROMETER_Y, ACCEL_Y_FEAT, feats);

    // ACCEL_Z
    compute_imu_family(features_selector, sig, len, ACCELEROMETER_Z, ACCEL_Z_FEAT, feats);



    // GYRO_Y
    compute_imu_family(features_selector, sig, len, GYROSCOPE_Y, GYRO_Y_FEAT, feats);

    // GYRO_P
    compute_imu_family(features_selector, sig, len, GYROSCOPE_P, GYRO_P_FEAT, feats);

    // GYRO_R
    compute_imu_family(features_selector, sig, len, GYROSCOPE_R, GYRO_R_FEAT, feats);

    // Combine signals via L2 norm (float mode)
    float *combo_signal = (float*)malloc(len * sizeof(float));

    RA_SET_IMU_CTX("IMU_L2_ACCEL");
    for(int16_t i=0; i<len; i++){
        combo_signal[i] = L2_norm(&sig[i][0], 3);
    }
    RA_IMU_LOG_ARRAY("imu_features", "sig_input", combo_signal, len);
    imu_sig_float_t s_accel = { .data = combo_signal, .len = len };
    imu_run_feature_table(&features_selector[ACCEL_COMBO], imu_view_from_float(s_accel), &feats[ACCEL_COMBO]);
    RA_CLEAR_IMU_CTX();

    RA_SET_IMU_CTX("IMU_L2_GYRO");
    for(int16_t i=0; i<len; i++){
        combo_signal[i] = L2_norm(&sig[i][3], 3);
    }
    RA_IMU_LOG_ARRAY("imu_features", "sig_input", combo_signal, len);
    imu_sig_float_t s_gyro = { .data = combo_signal, .len = len };
    imu_run_feature_table(&features_selector[GYRO_COMBO], imu_view_from_float(s_gyro), &feats[GYRO_COMBO]);
    RA_CLEAR_IMU_CTX();

    free(combo_signal);
#endif

}


//////////////////////////////////////////////////////////////////////////////////
