#ifndef _MFCC_MODULE_H_
#define _MFCC_MODULE_H_

#include <stdint.h>
#include <types.h>

/*
    This module contains the main helper functions to compute the MFCCs
*/

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
template <RealType T> void stft(const T *x, int16_t len, int16_t n_frames, T *res);

/// @brief Computes the melodic spectrogram by using the STFT method
/// and by multiplying it a mel_basis matrix
/// @param *x           pointer to the input signal
/// @param len          lenght of the array
/// @param n_frames     number of frames
/// @param *res         pointer to the result
template <RealType T> void mel_spectrogram_full(const T *x, int16_t len, int16_t n_frames, T *res);

/// @brief Computes the melodic spectrogram by using the STFT method
/// and by multiplying it a mel_basis matrix. This implementation computes
/// only the required frames
/// @param *x               pointer to the input signal
/// @param len              lenght of the array
/// @param n_frames         number of frames
/// @param *idx_required    pointer to the indexes that specify the required frames
/// @param *res             pointer to the result
template <RealType T> void mel_spectrogram(const T *x, int16_t len, int16_t n_frames, uint8_t *idx_required, T *res);

/// @brief Converts the input array of powers to dB and stores the
/// result into res array
/// @param *x               pointer to the input signal
/// @param len              lenght of the array
/// @param *res             pointer to the result
template <RealType T> void power_to_dB(T *x, int16_t len, T *res);

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
template <RealType T> void dct_matrix(T *x, int16_t rows, int16_t cols, T *y);

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
template <RealType T> void entropy(T *spectrogram, int16_t n_rows, int16_t n_columns, T *res);

#include <mfcc_module.hpp>

#endif
