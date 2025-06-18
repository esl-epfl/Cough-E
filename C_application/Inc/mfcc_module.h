#ifndef _MFCC_MODULE_H_
#define _MFCC_MODULE_H_

/*
    This module contains the main helper functions to compute the MFCCs
*/


// Defines and macros
#define N_FFT   2048    // NOTE: if this is changed then the hann window has to be recomputed
#define FFT_RES_LEN ((N_FFT / 2) + 1)
#define HOP_LEN 512     // from the STFT framing of the signal


#define PAD_LEN 1024

// defines used in the dB conversion
#define F_MIN 1.17549e-038 
#define TOP_DB 80.0

// define used for the DCT computation
#define PI 3.14159265358979323846



/// @brief Computes the power of the signal using the STFT method.
///
/// The final result will be a matrix stored in the 1D array res.
/// Note that the matrix is stored columns by column, one after the other:
/// res = [.... COLUMN 0 ....|.... COLUMN 1 ....|....].
/// So compared to the `librosa` implementation on python (`stft()`), the first 1025
/// elements here correspond to the first column of the python result.
/// @param *x           pointer to the input signal
/// @param len          lenght of the array
/// @param n_frames     number of frames
/// @param *res         pointer to the result
void stft(const float *x, int16_t len, int16_t n_frames, float *res);



/// @brief Computes the melodic spectrogram by using the STFT method
/// and by multiplying it a mel_basis matrix
/// @param *x           pointer to the input signal
/// @param len          lenght of the array
/// @param n_frames     number of frames
/// @param *res         pointer to the result
void mel_spectrogram_full(const float *x, int16_t len, int16_t n_frames, float *res);



/// @brief Computes the melodic spectrogram by using the STFT method
/// and by multiplying it a mel_basis matrix. This implementation computes
/// only the required frames
/// @param *x               pointer to the input signal
/// @param len              lenght of the array
/// @param n_frames         number of frames
/// @param *idx_required    pointer to the indexes that specify the required frames
/// @param *res             pointer to the result
void mel_spectrogram(const float *x, int16_t len, int16_t n_frames, uint8_t *idx_required, float *res);



/// @brief Converts the input array of powers to dB and stores the
/// result into res array
/// @param *x               pointer to the input signal
/// @param len              lenght of the array
/// @param *res             pointer to the result
void power_to_dB(float *x, int16_t len, float *res);




/**
* @brief Computes the DCT (Discrete Cosine Transform) on the input matrix and 
    stores the result in the output matrix.
    
    Both input and output matrices are stored as 1D arrays.
    
    The matrix is stored on a row by row basis:
    x = [... ROW 0 ... | ... ROW 1 ... | ....]
* @param *x     pointer to the input
* @param rows   number of rows
* @param cols   number of columns
* @param *y     pointer to the result
*/
void dct_matrix(float *x, int16_t rows, int16_t cols, float *y);



/**
 * Computes the entropy of the given spectrogram.
 * 
 * @param *spectrogram  :   pointer to the spectrogram. It has to be
 *                          a matrix store row by row in a linear array
 * @param n_rows        :   number of rows of the spectrogram matrix
 * @param n_columns     :   number of columns of the spectrogram matrix
 * @param *res          :   array where to store the resulting entropy. It should
 *                          be an array of `n_rows` elements
*/
void entropy(float *spectrogram, int16_t n_rows, int16_t n_columns, float *res);

#endif
