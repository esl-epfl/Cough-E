#ifndef _FREQ_FEAT_H_  
#define _FREQ_FEAT_H_

#include <inttypes.h>
#include <welch_psd.h>

/*
    Set of functions to compute the RFFT (Real FFT) and spectral features of a signal
*/
void compute_rfft(const float *sig, int16_t len, int16_t fs, float *mags, float *freqs, float *sum_mags);
void compute_periodogram(const float *sig, int16_t len, int16_t fs, float *psd, float *freqs);
float compute_spec_decrease(float* mags, float* freqs, int16_t len, float sum_mags);
float compute_spectral_slope(float *mags, float *freqs, int16_t len, float sum_mags);
float compute_rolloff(float *mags, float *freqs, int16_t len, float sum_mags);
float compute_centroid(float *mags, float *freqs, int16_t len, float sum_mags);
float compute_spread(float *mags, float *freqs, int16_t len, float sum_mags, float centroid);
float compute_kurt(float *mags, float *freqs, int16_t len, float sum_mags, float centroid, float spread);
float compute_skew(float *mags, float *freqs, int16_t len, float sum_mags, float centroid, float spread);
float compute_flatness(float *x, int16_t len);
float compute_std(float *x, int16_t len);
float compute_spectral_entropy(float *x, int16_t len);
float get_domiant_freq(float *psd, float *freqs, int16_t len);
void normalized_bandpowers(float *psd, float *freqs, int16_t len, const int8_t *psd_selector, float *band_powers);
void mfcc_computation(const float *x, int16_t len, int16_t n_frames, float *coeffs);
void get_mfcc_features(const float *x, int16_t len, float *mean_mfcc, float *std_mfcc);
void get_mel_spectrogram_features(const float *x, int16_t len, uint8_t *idx_needed, uint8_t n_mels_needed, float *mean_mel_spectr, float *std_mel_spectr, float *max_mel_spectr, float *entropy_mel_spectr);

#endif