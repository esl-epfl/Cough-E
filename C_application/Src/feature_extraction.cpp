#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <types.h>

#include <azc.h>
#include <feature_extraction.h>
#include <frequency_features.h>
#include <helpers.h>
#include <time_domain_feat.h>

#include <audio_features.h>
#include <imu_features.h>

#include <range_analysis.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)
#include <FxP/audio/audio_pipeline_fxp.h>
#include <FxP/core/fxp_core.h>
#include <FxP/imu/imu_pipeline.h>
#endif

#ifdef RANGE_ANALYSIS
const char *_ra_imu_signal_ctx = "UNKNOWN";
int _ra_imu_active = 0;
#endif

#ifndef FXP_MODE

//////////////////////////////////////////////////////////////////////////////////
/*                      Local functions definitions                             */
//////////////////////////////////////////////////////////////////////////////////

int is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index) {

    for (uint16_t i = start_index; i <= end_index; i++) {
        if (features_selector[i] == 1) {
            return 1; // It is sufficient to check if one is needed
        }
    }
    return 0;
}

static void imu_run_float_features(const int8_t *features_selector, imu_data_t *sig, int16_t len, imu_data_t *feats) {
    if (features_selector[LINE_LENGTH]) {
        PERF_KERNEL_TIMER_START();
        feats[LINE_LENGTH] = get_line_length(sig, len);
        PERF_KERNEL_TIMER_STOP("IMU|LINE_LENGTH");
        // PA_LOG("imu", "line_length", feats[LINE_LENGTH]);
    }
    if (features_selector[ZERO_CROSSING_RATE_IMU]) {
        PERF_KERNEL_TIMER_START();
        feats[ZERO_CROSSING_RATE_IMU] = compute_zrc(sig, len);
        PERF_KERNEL_TIMER_STOP("IMU|ZRC");
        // PA_LOG("imu", "zrc", feats[ZERO_CROSSING_RATE_IMU]);
    }
    if (features_selector[KURTOSIS]) {
        PERF_KERNEL_TIMER_START();
        feats[KURTOSIS] = get_kurtosis(sig, len);
        PERF_KERNEL_TIMER_STOP("IMU|KURTOSIS");
        // PA_LOG("imu", "kurtosis", feats[KURTOSIS]);
    }
    PERF_KERNEL_TIMER_START();
    imu_data_t rms = get_rms<imu_data_t, real_t>(sig, len);
    PERF_KERNEL_TIMER_STOP("IMU|RMS");
    if (features_selector[ROOT_MEANS_SQUARED_IMU]) {
        feats[ROOT_MEANS_SQUARED_IMU] = rms;
        // PA_LOG("imu", "rms", feats[ROOT_MEANS_SQUARED_IMU]);
    }
    if (features_selector[CREST_FACTOR_IMU]) {
        PERF_KERNEL_TIMER_START();
        feats[CREST_FACTOR_IMU] = rms > 0.0f ? get_crest(sig, len, rms) : 0.0f;
        PERF_KERNEL_TIMER_STOP("IMU|CREST");
        // PA_LOG("imu", "crest", feats[CREST_FACTOR_IMU]);
    }
    for (uint8_t i = 0; i < N_AZC; i++) {
        PERF_KERNEL_TIMER_START();
        uint8_t idx = (uint8_t)(APPROXIMATE_ZERO_CROSSING + i);
        if (features_selector[idx]) {
            imu_data_t eps = EPSILON_START + (EPSILON_STEP * i);
            feats[idx] = azc_computation(sig, len, eps);
            // PA_LOG("imu", "azc", feats[idx]);
        }
        PERF_KERNEL_TIMER_STOP("IMU|AZC");
    }
}

void fft_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                        audio_data_t *feats) {

    // FFT-dependent features' indexes
    // 0  : 6  the singular ones

    if (!is_required(features_selector, SPECTRAL_DECREASE, SPECTRAL_SKEW)) {
        return;
    }

    RA_LOG_ARRAY("AUDIO_FFT", "fft_based_features", "sig_input", sig, len);

    int16_t fft_size = (len / 2) + 1;
    audio_data_t *magnitudes = (audio_data_t *)malloc(fft_size * sizeof(audio_data_t));
    audio_data_t *frequencies = (audio_data_t *)malloc(fft_size * sizeof(audio_data_t));
    audio_data_t sum_mags = 0.0f;

    if (!magnitudes || !frequencies) {
        free(magnitudes);
        free(frequencies);
        return;
    }

    PERF_KERNEL_TIMER_START();
    compute_rfft(sig, len, fs, magnitudes, frequencies, &sum_mags);
    PERF_KERNEL_TIMER_STOP("AUDIO|RFFT");

    if (features_selector[SPECTRAL_DECREASE]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_decrease = compute_spec_decrease(magnitudes, frequencies, (len / 2) + 1, sum_mags);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_DECREASE");
        // PA_LOG("audio_fft", "spectral_decrease", spectral_decrease);
        feats[SPECTRAL_DECREASE] = spectral_decrease;
    }

    if (features_selector[SPECTRAL_SLOPE]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_slope = compute_spectral_slope(magnitudes, frequencies, (len / 2) + 1, sum_mags);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_SLOPE");
        // PA_LOG("audio_fft", "spectral_slope", spectral_slope);
        feats[SPECTRAL_SLOPE] = spectral_slope;
    }

    if (features_selector[SPECTRAL_ROLLOFF]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_rolloff = compute_rolloff(magnitudes, frequencies, (len / 2) + 1, sum_mags);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_ROLLOFF");
        // PA_LOG("audio_fft", "spectral_rolloff", spectral_rolloff);
        feats[SPECTRAL_ROLLOFF] = spectral_rolloff;
    }

    if (is_required(features_selector, SPECTRAL_CENTROID, SPECTRAL_SKEW)) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_cetroid = compute_centroid(magnitudes, frequencies, (len / 2) + 1, sum_mags);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_ROLLOFF");
        // PA_LOG("audio_fft", "spectral_cetroid", spectral_cetroid);

        if (features_selector[SPECTRAL_CENTROID]) {
            feats[SPECTRAL_CENTROID] = spectral_cetroid;
        }

        if (is_required(features_selector, SPECTRAL_SPREAD, SPECTRAL_SKEW)) {
            PERF_KERNEL_TIMER_START();
            audio_data_t spectral_spread =
                compute_spread(magnitudes, frequencies, (len / 2) + 1, sum_mags, spectral_cetroid);
            PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_SPREAD");
            // PA_LOG("audio_fft", "spectral_spread", spectral_spread);

            if (features_selector[SPECTRAL_SPREAD]) {
                feats[SPECTRAL_SPREAD] = spectral_spread;
            }

            if (features_selector[SPECTRAL_KURTOSIS]) {
                PERF_KERNEL_TIMER_START();
                audio_data_t kurt =
                    compute_kurt(magnitudes, frequencies, (len / 2) + 1, sum_mags, spectral_cetroid, spectral_spread);
                PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_KURTOSIS");
                // PA_LOG("audio_fft", "spectral_kurt", kurt);
                feats[SPECTRAL_KURTOSIS] = kurt;
            }

            if (features_selector[SPECTRAL_SKEW]) {
                PERF_KERNEL_TIMER_START();
                audio_data_t skew =
                    compute_skew(magnitudes, frequencies, (len / 2) + 1, sum_mags, spectral_cetroid, spectral_spread);
                PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_SKEW");
                // PA_LOG("audio_fft", "spectral_skew", skew);
                feats[SPECTRAL_SKEW] = skew;
            }
        }
    }

    free(magnitudes);
    free(frequencies);
}

void periodogram_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                                audio_data_t *feats) {

    // Periodogram dependent features' indexes
    // 7  : 9 for the singular ones
    // 10 : 12 for the PSD ones
    if (!is_required(features_selector, SPECTRAL_FLATNESS, POWER_SPECTRAL_DENSITY + N_PSD - 1)) {
        return;
    }

    RA_LOG_ARRAY("AUDIO_PSD", "periodogram_based_features", "sig_input", sig, len);

    int16_t psd_size = (NPERSEG / 2) + 1;
    audio_data_t *psd = (audio_data_t *)malloc(psd_size * sizeof(audio_data_t));
    audio_data_t *freqs = (audio_data_t *)malloc(psd_size * sizeof(audio_data_t));
    if (!psd || !freqs) {
        free(psd);
        free(freqs);
        return;
    }

    PERF_KERNEL_TIMER_START();
    compute_periodogram<audio_data_t, real_t>(sig, len, fs, psd, freqs);
    PERF_KERNEL_TIMER_STOP("AUDIO|PERIODOGRAM");

    if (features_selector[SPECTRAL_FLATNESS]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_flatness = compute_flatness<audio_data_t, real_t>(psd, psd_size);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_FLATNESS");
        // PA_LOG("audio_periodogram", "spectral_flatness", spectral_flatness);
        feats[SPECTRAL_FLATNESS] = spectral_flatness;
    }

    if (features_selector[SPECTRAL_STD]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_std = compute_std(psd, psd_size);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_STD");
        // PA_LOG("audio_periodogram", "spectral_std", spectral_std);
        feats[SPECTRAL_STD] = spectral_std;
    }

    if (features_selector[SPECTRAL_ENTROPY]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t spectral_entr = compute_spectral_entropy(psd, psd_size);
        PERF_KERNEL_TIMER_STOP("AUDIO|SPECTRAL_ENTROPY");
        // PA_LOG("audio_periodogram", "spectral_entropy", spectral_entr);
        feats[SPECTRAL_ENTROPY] = spectral_entr;
    }

    if (features_selector[DOMINANT_FREQUENCY]) {
        PERF_KERNEL_TIMER_START();
        audio_data_t dominant_freq = get_domiant_freq(psd, freqs, psd_size);
        PERF_KERNEL_TIMER_STOP("AUDIO|DOMINANT_FREQ");
        // PA_LOG("audio_periodogram", "dominant_freq", dominant_freq);
        feats[DOMINANT_FREQUENCY] = dominant_freq;
    }

    if (is_required(features_selector, POWER_SPECTRAL_DENSITY, POWER_SPECTRAL_DENSITY + N_PSD - 1)) {
        audio_data_t *band_powers = (audio_data_t *)malloc(N_PSD * sizeof(audio_data_t));
        PERF_KERNEL_TIMER_START();
        normalized_bandpowers(psd, freqs, psd_size, &features_selector[POWER_SPECTRAL_DENSITY], band_powers);
        PERF_KERNEL_TIMER_STOP("AUDIO|BAND_POWER");
        for (int8_t i = 0; i < N_PSD; i++) {
            // PA_LOG("audio_periodogram", "band_powers", band_powers[i]);
            feats[POWER_SPECTRAL_DENSITY + i] = band_powers[i];
        }

        free(band_powers);
    }

    free(psd);
    free(freqs);
}

void mfcc_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, audio_data_t *feats) {

    // 13 : 38 for the MFCCs features
    if (is_required(features_selector, MEL_FREQUENCY_CEPSTRAL_COEFFICIENT, ZERO_CROSSING_RATE - 1)) {
        // compute MFCCs

        audio_data_t *mean_mfcc = (audio_data_t *)malloc(N_MFCC * sizeof(audio_data_t));
        audio_data_t *std_mfcc = (audio_data_t *)malloc(N_MFCC * sizeof(audio_data_t));
        PERF_KERNEL_TIMER_START();
        get_mfcc_features(sig, len, mean_mfcc, std_mfcc);
        PERF_KERNEL_TIMER_STOP("AUDIO|MFCC");

        // stores first the mean and then the std, one after the other
        for (int16_t i = 0; i < N_MFCC; i++) {
            // PA_LOG("audio_mfcc", "mean_mfcc", mean_mfcc[i]);
            // PA_LOG("audio_mfcc", "std_mfcc", std_mfcc[i]);
            feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = mean_mfcc[i];
            feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = std_mfcc[i];
        }

        free(mean_mfcc);
        free(std_mfcc);
    }
}

void mel_spectrogram_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len,
                              audio_data_t *feats) {

    if (is_required(features_selector, MEL_FREQUENCY_CEPSTRAL_COEFFICIENT, ZERO_CROSSING_RATE - 1)) {

        RA_LOG_ARRAY("AUDIO_MEL", "mel_spectrogram_features", "sig_input", sig, len);

        // compute MEL SPECTROGRAM

        // Indexes of the Mel bins required for the features computation
        uint8_t *idxs_needed = (uint8_t *)malloc(N_MFCC * sizeof(uint8_t));
        memset(idxs_needed, 0, N_MFCC * sizeof(uint8_t));

        // Counts the number of MEL features needed and fills the indexes needed
        uint8_t n_mels_needed = 0;
        for (uint8_t i = 0; i < N_MFCC; i++) {
            if ((features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] == 1) ||                // mean
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i + N_MFCC] == 1) ||       // std
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i + (2 * N_MFCC)] == 1) || // max
                (features_selector[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i + (3 * N_MFCC)] == 1)    // entropy
            ) {
                idxs_needed[n_mels_needed] = i;
                n_mels_needed++;
            }
        }

        // Arrays to temporary store the features
        audio_data_t *mean_mel_spectr = (audio_data_t *)malloc(n_mels_needed * sizeof(audio_data_t));
        audio_data_t *std_mel_spectr = (audio_data_t *)malloc(n_mels_needed * sizeof(audio_data_t));
        audio_data_t *max_mel_spectr = (audio_data_t *)malloc(n_mels_needed * sizeof(audio_data_t));
        audio_data_t *entropy_mel_spectr = (audio_data_t *)malloc(n_mels_needed * sizeof(audio_data_t));

        PERF_KERNEL_TIMER_START();
        get_mel_spectrogram_features(sig, len, idxs_needed, n_mels_needed, mean_mel_spectr, std_mel_spectr,
                                     max_mel_spectr, entropy_mel_spectr);
        PERF_KERNEL_TIMER_STOP("AUDIO|MEL_SPECTROGRAM");

        // stores first the mean, the std, the max and the entropy, one after the other
        int idx = 0;
        for (int16_t i = 0; i < N_MFCC; i++) {
            if (i == idxs_needed[idx]) { // Only if the feature is one of the needed ones
                // PA_LOG("audio_mel", "mean_mel_spectr", mean_mel_spectr[idx]);
                // PA_LOG("audio_mel", "std_mel_spectr", std_mel_spectr[idx]);
                // PA_LOG("audio_mel", "max_mel_spectr", max_mel_spectr[idx]);
                // PA_LOG("audio_mel", "entropy_mel_spectr", entropy_mel_spectr[idx]);
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + i] = mean_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + N_MFCC + i] = std_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (2 * N_MFCC) + i] = max_mel_spectr[idx];
                feats[MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (3 * N_MFCC) + i] = entropy_mel_spectr[idx];
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

void mean_based_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, audio_data_t *feats) {

    // 39 : 41 for the singular ones
    if (is_required(features_selector, ROOT_MEANS_SQUARED, CREST_FACTOR)) {

        RA_LOG_ARRAY("AUDIO_FFT", "mean_based_features", "sig_input", sig, len);

        // compute mean
        audio_data_t *zero_mean =
            (audio_data_t *)malloc(len * sizeof(audio_data_t)); // to store the signal after subtracting the mean
        PERF_KERNEL_TIMER_START();
        sub_mean(sig, zero_mean, len);
        PERF_KERNEL_TIMER_STOP("AUDIO|SUB_MEAN");
        RA_LOG_ARRAY("AUDIO_FFT", "mean_based_features", "zero_mean", zero_mean, len);

        if (features_selector[ZERO_CROSSING_RATE]) {
            // compute ZCR
            PERF_KERNEL_TIMER_START();
            audio_data_t zcr = compute_zrc(zero_mean, len);
            PERF_KERNEL_TIMER_STOP("AUDIO|ZRC");
            // PA_LOG("audio_mean", "zcr", zcr);
            feats[ZERO_CROSSING_RATE] = zcr;
        }

        if (features_selector[ROOT_MEANS_SQUARED] || features_selector[CREST_FACTOR]) {
            // compute RMS
            PERF_KERNEL_TIMER_START();
            audio_data_t rms = get_rms<audio_data_t, real_t>(zero_mean, len);
            PERF_KERNEL_TIMER_STOP("AUDIO|RMS");
            // PA_LOG("audio_mean", "rms", rms);
            RA_LOG_SCALAR("AUDIO_FFT", "audio_rms", "result", rms);

            if (features_selector[ROOT_MEANS_SQUARED]) {
                // append RMS
                feats[ROOT_MEANS_SQUARED] = rms;
            }

            if (features_selector[CREST_FACTOR]) {
                // compute CREST
                PERF_KERNEL_TIMER_START();
                audio_data_t crest_factor = get_crest(zero_mean, len, rms);
                PERF_KERNEL_TIMER_STOP("AUDIO|CREST");
                // PA_LOG("audio_mean", "crest", crest_factor);
                RA_LOG_SCALAR("AUDIO_FFT", "audio_crest", "result", crest_factor);
                feats[CREST_FACTOR] = crest_factor;
            }
        }

        free(zero_mean);
    }
}

void eepd_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                   audio_data_t *feats) {

    // 42 : 61 for the singular ones
    if (is_required(features_selector, ENERGY_ENVELOPE_PEAK_DETECT,
                    (ENERGY_ENVELOPE_PEAK_DETECT + N_EEPD -
                     1))) { // -1 since it's the last one, otherwise it will check one index more

        int16_t *eepds = (int16_t *)malloc(N_EEPD * sizeof(int16_t));

        // compute EEPD
        PERF_KERNEL_TIMER_START();
        eepd(sig, len, fs, &features_selector[ENERGY_ENVELOPE_PEAK_DETECT], eepds);
        PERF_KERNEL_TIMER_STOP("AUDIO|EEPD");

        for (int16_t i = 0; i < N_EEPD; i++) {
            // PA_LOG("audio_eepd", "eepd", eepds[i]);
            feats[ENERGY_ENVELOPE_PEAK_DETECT + i] = eepds[i];
        }

        free(eepds);
    }
}

#ifdef RANGE_ANALYSIS
static const char *_imu_signal_names[] = {"accel_x", "accel_y", "accel_z", "gyro_y", "gyro_p", "gyro_r"};
#endif

void compute_imu_family(const int8_t *features_selector, const imu_data_t signal[][Num_IMU_signals], int16_t len,
                        int8_t signal_idx, int8_t sig_feat_idx, imu_data_t *feats) {
    if (is_required(features_selector, sig_feat_idx, sig_feat_idx + Num_imu_feat_families - 1)) {
        // Extract samples for the required signal axis
        imu_data_t *signal_samples = (imu_data_t *)malloc(len * sizeof(imu_data_t));

        for (int16_t i = 0; i < len; i++) {
            signal_samples[i] = signal[i][signal_idx];
        }

        RA_LOG_ARRAY("IMU_RAW", "imu_features", _imu_signal_names[signal_idx], signal_samples, len);

        RA_SET_IMU_CTX("IMU_RAW");
        imu_run_float_features(&features_selector[sig_feat_idx], signal_samples, len, &feats[sig_feat_idx]);
        RA_CLEAR_IMU_CTX();
        free(signal_samples);
    }
}

void compute_imu_l2(const int8_t *features_selector, const imu_data_t signal[][Num_IMU_signals], int16_t len,
                    int8_t signal_idx, int8_t sig_feat_idx, imu_data_t *feats) {
    imu_data_t *combo_signal = (imu_data_t *)malloc(len * sizeof(imu_data_t));

    for (int16_t i = 0; i < len; i++) {
        combo_signal[i] = L2_norm(&signal[i][signal_idx], 3);
    }
    RA_IMU_LOG_ARRAY("imu_features", "sig_input", combo_signal, len);

    imu_run_float_features(&features_selector[sig_feat_idx], combo_signal, len, &feats[sig_feat_idx]);

    free(combo_signal);
}

//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
/*                      Global functions definitions                            */
//////////////////////////////////////////////////////////////////////////////////

void audio_features(const int8_t *features_selector, const audio_data_t *sig, int16_t len, int16_t fs,
                    audio_data_t *feats) {

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

void imu_features(const int8_t *features_selector, const imu_data_t sig[][Num_IMU_signals], int16_t len,
                  imu_data_t *feats) {

    // Here len is the IMU_DIM_1 macro in the hardcoded samples

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
    RA_SET_IMU_CTX("IMU_L2_ACCEL");
    compute_imu_l2(features_selector, sig, len, 0, ACCEL_COMBO, feats);
    RA_CLEAR_IMU_CTX();

    RA_SET_IMU_CTX("IMU_L2_GYRO");
    compute_imu_l2(features_selector, sig, len, 3, GYRO_COMBO, feats);
    RA_CLEAR_IMU_CTX();
}

//////////////////////////////////////////////////////////////////////////////////

#else

static int fxp_is_required(const int8_t *features_selector, uint16_t start_index, uint16_t end_index) {
    for (uint16_t i = start_index; i <= end_index; i++) {
        if (features_selector[i] == 1)
            return 1;
    }
    return 0;
}

void audio_features(const int8_t *features_selector, const q2_14_t *sig, int16_t len, int16_t fs, fxp_feat_t *feats) {
    if (!features_selector || !sig || !feats || len <= 0 || fs <= 0)
        return;

    audio_fft_features(features_selector, sig, len, fs, feats);
    audio_psd_features(features_selector, sig, len, fs, feats);
    audio_mel_features(features_selector, sig, len, feats);
    audio_crest_factor(features_selector, sig, len, feats);
}

void imu_features(const int8_t *features_selector, const q11_5_t sig[][Num_IMU_signals], int16_t len,
                  fxp_feat_t *feats) {
    if (!features_selector || !sig || !feats || len <= 0)
        return;

    uq10_6_t *combo_l2a = (uq10_6_t *)malloc((size_t)len * sizeof(*combo_l2a));
    uq5_11_t *combo_l2g = (uq5_11_t *)malloc((size_t)len * sizeof(*combo_l2g));
    q11_5_t *axis_samples = (q11_5_t *)malloc((size_t)len * sizeof(*axis_samples));
    if (!combo_l2a || !combo_l2g || !axis_samples) {
        free(combo_l2a);
        free(combo_l2g);
        free(axis_samples);
        return;
    }

    for (int16_t i = 0; i < len; i++) {
        combo_l2a[i] = imu_l2a(sig[i][0], sig[i][1], sig[i][2]);
        combo_l2g[i] = imu_l2g(sig[i][3], sig[i][4], sig[i][5]);
    }

    const int8_t axis_ids[Num_IMU_signals] = {ACCELEROMETER_X, ACCELEROMETER_Y, ACCELEROMETER_Z,
                                              GYROSCOPE_Y,     GYROSCOPE_P,     GYROSCOPE_R};
    const int8_t axis_feat_base[Num_IMU_signals] = {ACCEL_X_FEAT, ACCEL_Y_FEAT, ACCEL_Z_FEAT,
                                                    GYRO_Y_FEAT,  GYRO_P_FEAT,  GYRO_R_FEAT};

    for (int s = 0; s < Num_IMU_signals; s++) {
        int8_t base = axis_feat_base[s];
        if (!fxp_is_required(features_selector, base, (uint16_t)(base + Num_imu_feat_families - 1)))
            continue;

        for (int16_t i = 0; i < len; i++) {
            axis_samples[i] = sig[i][axis_ids[s]];
        }
        imu_run_raw_features(&features_selector[base], axis_samples, len, &feats[base]);
    }

    if (fxp_is_required(features_selector, ACCEL_COMBO, (uint16_t)(ACCEL_COMBO + Num_imu_feat_families - 1))) {
        imu_run_l2a_features(&features_selector[ACCEL_COMBO], combo_l2a, len, &feats[ACCEL_COMBO]);
    }

    if (fxp_is_required(features_selector, GYRO_COMBO, (uint16_t)(GYRO_COMBO + Num_imu_feat_families - 1))) {
        imu_run_l2g_features(&features_selector[GYRO_COMBO], combo_l2g, len, &feats[GYRO_COMBO]);
    }

    free(combo_l2a);
    free(combo_l2g);
    free(axis_samples);
}

#endif
