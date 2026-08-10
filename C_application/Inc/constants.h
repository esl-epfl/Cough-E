#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <stdint.h>
#include <types.h>

// dct_lin.h
template <RealType T> extern const T FLASH1 dct_cos[16384];

// mel_basis.h (To convert the STFT output into MEL domain)
// MEL matrix for projecting a power spectrum into a mel basis.
// The numbers are hardcoded and taken from python.
//////////////////////////////////////////////////////////////////////////////////
// Note that this only works when the RFFT size used is 2048 (so the output		//
// will be 1025 samples)
// //
//////////////////////////////////////////////////////////////////////////////////
#define MEL_ROWS 64
#define MEL_COLUMNS 1025
#define MAX_NZ_ELEMS 73 // max number of non-zero elements
/*
    Indexes of the first and last non-zero elements for each row of the mel basis.
    For each row, the non-zero elements are stored one after the other, so I just
    need to store the indexes of the first and last element
*/
extern const int16_t mel_nz_indexes[MEL_ROWS][2];
// 4672 values * 4 bytes = 18688 bytes
template <RealType T> extern const T DINTL1 mel_basis[MEL_ROWS][MAX_NZ_ELEMS];

// mfcc_hann_wind.h (Hanning window to be used in the STFT method before the RFFT)
#define HANN_SIZE 2048
// 2048 values * 4 bytes = 8192 bytes
template <RealType T> extern const T DINTL1 hann_mfcc_wind[HANN_SIZE];

// welch_psd.h
#define NPERSEG 900  // number of samples in each window of the welch method
#define NOVERLAP 450 // number of overlapping samples between subsequent windows
// 900 values * 4 bytes = 3600 bytes
template <RealType T> extern const T DINTL1 hann_window[NPERSEG];

#endif
