#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>

#include <feature_extraction.h>
#include <frequency_features.h>
#include <welch_psd.h>
#include <mfcc_module.h>
#include <mel_basis.h>
#include <helpers.h>

#include <audio_features.h>

#include <kiss_fftr.h>


// Helper function for the RFFT
void _rfft(const float *sig, int16_t len, float *real, float *imag);


/*
    Computes the Real FFT of a time signal sig.
    It stores the magnitudes, the frequencies and the sum of all the magnitudes inside
    *mags, *freqs abd *sum_mags, respectively.
    Note that also the sampling_frequency (fs) and the length (len) of the signal are required as inputs
*/
void compute_rfft(const float *sig, int16_t len, int16_t fs, float *mags, float *freqs, float *sum_mags){

    float *re = (float*)malloc(len * sizeof(float));
    float *im = (float*)malloc(len * sizeof(float));

    _rfft(sig, len, re, im);

    // Compute the magnitude of each FFT output
    for(int16_t i=0; i<(len/2)+1 ; i++){
        mags[i] = sqrtf((re[i] * re[i]) + (im[i] * im[i]));
        *sum_mags += mags[i];
    }

    // Get the frequency bins (only the positive one becaus it's a real FFT)
    for(int16_t i=0; i<(len/2)+1; i++){
        freqs[i] = (float)(i * fs) / len;
    }

    free(re);
    free(im);
}


/*
    Helper function not callable externally.
    This just computes the RFFT of the input signal and 
    stores the Real and Imaginary parts in two separate arrays
*/
void _rfft(const float *sig, int16_t len, float *real, float *imag){

    kiss_fftr_cfg cfg = kiss_fftr_alloc(len, 0, 0, 0);
    kiss_fft_cpx *cx_out = (kiss_fft_cpx *) malloc(len * sizeof(kiss_fft_cpx));

    kiss_fftr(cfg, sig, cx_out);

    for(int16_t i=0; i<len; i++){
        real[i] = cx_out[i].r;
        imag[i] = cx_out[i].i;
    }

    free(cfg);
    free(cx_out);
}


/*
    Computes the periodogram of a signal using the Welch's method
*/
void compute_periodogram(const float *sig, int16_t len, int16_t fs, float *psd, float *freqs){

    float freq_step = ((float)fs / 2) / ( (float)NPERSEG / 2);  // the frequency step for eah bin 
    float *win = (float*)malloc(NPERSEG * sizeof(float));   // to keep the data of the current processed window
    float *cumul_sums = (float*)malloc(NPERSEG * sizeof(float));  // To store the cumulative sum of the FFT of each frequency bin
    memset(cumul_sums, 0, NPERSEG * sizeof(float));

    // To store the real and imaginary parts of each FFT
    float *re = (float*)malloc(NPERSEG * sizeof(float));
    float *im = (float*)malloc(NPERSEG * sizeof(float));

    // To store the magnitudes squared after the FFT
    float *mags_squared = (float*)malloc(((NPERSEG/2)+1) * sizeof(float));


    float mean = 0.0;
    float scale = 0.0;
    float sum = 0.0;

    for(int16_t i=0; i<NPERSEG; i++){
        sum += hann_window[i] * hann_window[i];
    }

    scale = 1 / (fs * sum); 

    // start and end indexes of the current processed window
    int16_t start = 0;
    int16_t end = start + NPERSEG;

    int16_t steps = (len - NOVERLAP) / (NPERSEG - NOVERLAP);    // Number or windows that will be processed
    for(int16_t i=0; i<steps; i++){

        vect_copy(sig, start, NPERSEG, win);    // copies the current window from the signal

        // subtract the mean
        mean = vect_mean(win, NPERSEG);
        sub_constant(win, NPERSEG, mean, win);

        // Apply the window function
        for(int16_t i=0; i<NPERSEG; i++){
            win[i] *= hann_window[i];
        }  


        _rfft(win, NPERSEG, re, im);    // Actual Real FFT computation

        for(int16_t i=0; i<(NPERSEG/2)+1; i++){
            mags_squared[i] = (re[i] * re[i]) + (im[i] * im[i]);
            mags_squared[i] *= scale;

            if(i != 0 && i != (NPERSEG/2)){
                mags_squared[i] *= 2;       // Multiply by 2, apart from DC frequency (first element) and last element
            }
            cumul_sums[i] += mags_squared[i];   // Update the cumulative sum (element-wise across FFT result of different windows)
        }

        start = end - NOVERLAP;
        end = start + NPERSEG;
    }

    // Compute the result psd as the avarage of each FFT result and compute the frequencies
    for(int16_t i=0; i<(NPERSEG/2)+1; i++){
        psd[i] = cumul_sums[i] / steps;
        freqs[i] = freq_step * i;
    }

    free(win);
    free(cumul_sums);
    free(re);
    free(im);
    free(mags_squared);
}


/*
    Returns the spectral decrease computed from the magnitudes and the frequencies of
    the spectrum of a signal
*/
float compute_spec_decrease(float* mags, float* freqs, int16_t len, float sum_mags){

    float sum = 0.0;
    float dc_mag = mags[0];

    for(int16_t i=0; i<len; i++){
        sum += (mags[i] - dc_mag) / (i + 1);
    }

    return sum / sum_mags;
}


/*
    Returns the spectral slope computed from the magnitudes and the frequencies of
    the spectrum of a signal
*/
float compute_spectral_slope(float *mags, float *freqs, int16_t len, float sum_mags){

    float mean_mag = sum_mags / len;
    float mean_freq = 0.0;

    mean_freq = vect_mean(freqs, len);

    // Numerator and denominator for the final slope computation
    float num = 0.0;
    float den = 0.0;

    for(int16_t i=0; i<len; i++){
        num += (freqs[i] - mean_freq) * (mags[i] - mean_mag);
        den += (freqs[i] - mean_freq) * (freqs[i] - mean_freq); 
    }

    return num / den;
}


/*
    Returns the spectral roll off computed from the magnitudes and the frequencies of
    the spectrum of a signal
*/
float compute_rolloff(float *mags, float *freqs, int16_t len, float sum_mags){

    float rolloff_energy = 0.95 * sum_mags;
    float sum = 0.0;
    float rolloff = -1.0;   // Error value

    for(int16_t i=0; i<len; i++){
        sum += mags[i];

        if(sum >= rolloff_energy)
            return freqs[i];
    }

    return rolloff;
}


/*
    Returns the spectral centroid computed from the magnitudes and the frequencies of
    the spectrum of a signal
*/
float compute_centroid(float *mags, float *freqs, int16_t len, float sum_mags){

    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        sum += freqs[i] * mags[i];
    }

    return sum / sum_mags;
}


/*
    Returns the spectral spread computed from the magnitudes and the frequencies of
    the spectrum of a signal.
    Note that this also requires the spectral centroid as an input
*/
float compute_spread(float *mags, float *freqs, int16_t len, float sum_mags, float centroid){

    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        sum += (freqs[i] - centroid) * (freqs[i] - centroid) * mags[i];
    }

    return sqrtf(sum / sum_mags);
}


/*
    Returns the spectral kurtosis computed from the magnitudes and the frequencies of
    the spectrum of a signal.
    Note that this also requires the spectral centroid and the spectral spread as inputs
*/
float compute_kurt(float *mags, float *freqs, int16_t len, float sum_mags, float centroid, float spread){

    float spread_4 = spread * spread * spread * spread; // spread^4
    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        register float tmp = (freqs[i] - centroid) * (freqs[i] - centroid);
        sum += tmp * tmp * mags[i];
    }

    return sum / (spread_4 * sum_mags);
}


/*
    Returns the spectral skewness computed from the magnitudes and the frequencies of
    the spectrum of a signal.
    Note that this also requires the spectral centroid and the spectral spread as inputs
*/
float compute_skew(float *mags, float *freqs, int16_t len, float sum_mags, float centroid, float spread){

    float spread_3 = spread * spread * spread;
    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        register float tmp = (freqs[i] - centroid) * (freqs[i] - centroid);
        sum += tmp * (freqs[i] - centroid) * mags[i];
    }

    return sum / (spread_3 * sum_mags);
}


/*
    Returns the spectral flatness of the given signal
*/
float compute_flatness(float *x, int16_t len){

    float gmean = 0.0;  // geometric
    float amean = 0.0;  // arithmetic

    float sum_logs = 0.0;
    for(int16_t i=0; i<len; i++){
        sum_logs += logf(x[i]);
    }
    sum_logs = sum_logs / len;

    gmean = exp(sum_logs);
    amean = vect_mean(x, len);
    return gmean / amean;
}


/*
    Returns the standard deviation of the given signal
*/
float compute_std(float *x, int16_t len){

    return vect_std(x, len);

}

float compute_spectral_entropy(float *x, int16_t len){

    // // Sum each column of the spectrogram
    // row_sum = vect_sum(&spectrogram[i*n_columns], n_columns);
    // // Divide all the row's elements by the sum of the row
    // vect_div_const(&spectrogram[i*n_columns], n_columns, row_sum);

    float *tmp = (float*)malloc(len * sizeof(float));
    float sum = vect_sum(x, len);
    vect_div_const(x, len, sum, tmp);
    entropy_calc(tmp, len, 2);

    free(tmp);

    return vect_sum(tmp, len);
}


/*
    Returns the frequency at which the maximum psd is found
*/
float get_domiant_freq(float *psd, float *freqs, int16_t len){
    
    return freqs[vect_max_index(psd, len)];
}


/*
    Computes the normalized power of each band.
    It requires the psd, the frequencies and also the psd_selector, which is an array
    of 1 or 0. 1 indicates that a specific band has to be computed. The bands
    are defined by the "psd_bands" structure
*/
void normalized_bandpowers(float *psd, float *freqs, int16_t len, const int8_t *psd_selector, float *band_powers){

    float dx_freq = freqs[1] - freqs[0];
    float total_power = simpson(psd, len, dx_freq);

    int16_t start_ind_freq = 0; // index of the first frequency inside the band
    int16_t n_bins = 0;         // number of frequency bins inside the band
    int8_t start_found = 0;     // 1 if the start frequency was found, useful to minimize the if-statements

    float band_power = 0.0;

    //check which PSD bands are needed
    for(int16_t i=0; i<N_PSD; i++){
        if(psd_selector[i] == 1){
            start_found = 0;
            start_ind_freq = 0;
            n_bins = 0;
            for(int16_t j=0; j<len; j++){
                if(!start_found && freqs[j] >= psd_bands[i].start){
                    start_ind_freq = j;
                    start_found = 1;
                }
                if(start_found && freqs[j] <= psd_bands[i].end){
                    n_bins++;
                }
            }

            band_power = simpson(&psd[start_ind_freq], n_bins, dx_freq);
            band_powers[i] = band_power / total_power;
        }
    }
}

/*
    Computes the MFCC coefficients of the given signal x
    This function uses the STFT technique, therefore the MFCC are computed
    for every frame in which the RFFT is computed.
    Basically the output will be a matrix having, on i-th row, the i-th coefficient
    for each signal frame.
    The matrix is stored in a 1-Dimentional array, storing each row one after the other:

    coeffs = [... ROW 0 ... | ... ROW 1 ... | ...]
*/
void mfcc_computation(const float *x, int16_t len, int16_t n_frames, float *coeffs){

    float *db_power = (float*)malloc((MEL_ROWS * n_frames) * sizeof(float));
    memset(db_power, 0.0, (MEL_ROWS * n_frames)*sizeof(float));

    // Mel spectrogram
    mel_spectrogram_full(x, len, n_frames, db_power);

    // Convert the power in dB
    power_to_dB(db_power, (MEL_ROWS * n_frames), db_power);

    // Apply the DCT
    dct_matrix(db_power, MEL_ROWS, n_frames, db_power);

    for(int16_t i=0; i<N_MFCC; i++){
        for(int16_t j=0; j<n_frames; j++){
            coeffs[(i*n_frames) + j] = db_power[(i * n_frames) + j];
        }
    }

    free(db_power);
}


void get_mfcc_features(const float *x, int16_t len, float *mean_mfcc, float *std_mfcc){

    int16_t padded_len = (2 * PAD_LEN) + len;                   // lenght of the padded signal
    int16_t n_frames = ((padded_len - N_FFT) / HOP_LEN) + 1;    // number of frames for the stft

    float *coeffs = (float*)malloc((N_MFCC*n_frames) * sizeof(float));
    mfcc_computation(x, len, n_frames, coeffs);


    for(int16_t i=0; i<N_MFCC; i++){
        mean_mfcc[i] = vect_mean(&coeffs[i*n_frames], n_frames);
        std_mfcc[i] = vect_std(&coeffs[i*n_frames], n_frames);
    }

    free(coeffs);

}


// void get_mel_spectrogram_features(const float *x, int16_t len, float *mean_mel_spectr, float *std_mel_spectr, float *max_mel_spectr, float *entropy_mel_spectr){

//     int16_t padded_len = (2 * PAD_LEN) + len;                   // lenght of the padded signal
//     int16_t n_frames = ((padded_len - N_FFT) / HOP_LEN) + 1;    // number of frames for the stft

//     float *spectrogram = (float*)malloc((MEL_ROWS * n_frames) * sizeof(float));
//     memset(spectrogram, 0.0, (MEL_ROWS * n_frames)*sizeof(float));

//     // Get the spectrogram
//     mel_spectrogram(x, len, n_frames, spectrogram);

//     // Stores the dB of the mel spectrogram
//     float *mel_dB = (float*)malloc((MEL_ROWS * n_frames) * sizeof(float));

//     // Convert the power of the spectrogram in dB
//     power_to_dB(spectrogram, (MEL_ROWS * n_frames), mel_dB);

//     // Computes the entropy of the spectrogram
//     entropy(spectrogram, MEL_ROWS, n_frames, entropy_mel_spectr);

//     // Computes the mean, std and maximum value of each MEL bin
//     for(int8_t i=0; i<MEL_ROWS; i++){
//         mean_mel_spectr[i] = vect_mean(&mel_dB[i*n_frames], n_frames);
//         std_mel_spectr[i] = vect_std(&mel_dB[i*n_frames], n_frames);
//         max_mel_spectr[i] = vect_max_value(&mel_dB[i*n_frames], n_frames);
//     }

//     free(spectrogram);
//     free(mel_dB);
// }

void get_mel_spectrogram_features(const float *x, int16_t len, uint8_t *idx_needed, uint8_t n_mels_needed, float *mean_mel_spectr, float *std_mel_spectr, float *max_mel_spectr, float *entropy_mel_spectr){

    int16_t padded_len = (2 * PAD_LEN) + len;                   // lenght of the padded signal
    int16_t n_frames = ((padded_len - N_FFT) / HOP_LEN) + 1;    // number of frames for the stft

    float *spectrogram = (float*)malloc((n_mels_needed * n_frames) * sizeof(float));
    memset(spectrogram, 0.0, (n_mels_needed * n_frames)*sizeof(float));

    // Get the spectrogram
    mel_spectrogram(x, len, n_frames, idx_needed, spectrogram);

    // Stores the dB of the mel spectrogram
    float *mel_dB = (float*)malloc((n_mels_needed * n_frames) * sizeof(float));

    // Convert the power of the spectrogram in dB
    power_to_dB(spectrogram, (n_mels_needed * n_frames), mel_dB);

    // Computes the entropy of the spectrogram
    entropy(spectrogram, n_mels_needed, n_frames, entropy_mel_spectr);

    // Computes the mean, std and maximum value of each MEL bin
    for(int8_t i=0; i<n_mels_needed; i++){
        mean_mel_spectr[i] = vect_mean(&mel_dB[i*n_frames], n_frames);
        std_mel_spectr[i] = vect_std(&mel_dB[i*n_frames], n_frames);
        max_mel_spectr[i] = vect_max_value(&mel_dB[i*n_frames], n_frames);
    }

    free(spectrogram);
    free(mel_dB);
}
