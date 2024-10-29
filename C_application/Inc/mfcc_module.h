#ifndef _MFCC_MODULE_H_
#define _MFCC_MODULE_H_

/*
    This module contains the main helper functions to compute the MFCCs
*/


// Defines and macros
#define N_FFT   2048    // NOTE: if this is changed then the hann window has to be recomputed
#define FFT_RES_LEN ((N_FFT / 2) + 1)
#define HOP_LEN 512     // fro the STFT framing of the signal


#define PAD_LEN 1024

// defines used in the dB conversion
#define F_MIN 1.17549e-038 
#define TOP_DB 80.0

// define used for the DCT computation
#define PI 3.14159265358979323846

void stft(const float *x, int16_t len, int16_t n_frames, float *res);
void mel_spectrogram_full(const float *x, int16_t len, int16_t n_frames, float *res);
void mel_spectrogram(const float *x, int16_t len, int16_t n_frames, uint8_t *idx_required, float *res);

void power_to_dB(float *x, int16_t len, float *res);

void dct_matrix(float *x, int16_t rows, int16_t cols, float *y);

void entropy(float *spectrogram, int16_t n_rows, int16_t n_columns, float *res);

#endif
