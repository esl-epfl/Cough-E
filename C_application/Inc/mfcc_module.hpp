#include <inttypes.h>
#include <stdlib.h>
#include <types.h>

#include <audio_features.h>
#include <constants.h>
#include <helpers.h>
#include <kiss_fftr.h>
#include <range_analysis.h>

// Defines and macros
#define N_FFT 2048 // NOTE: if this is changed then the hann window has to be recomputed
#define FFT_RES_LEN ((N_FFT / 2) + 1)
#define HOP_LEN 512 // from the STFT framing of the signal

#define PAD_LEN 1024

// defines used in the dB conversion
const sreal_t TOP_DB = sreal_t(80.0);

// define used for the DCT computation
#define PI 3.14159265358979323846

#ifndef FXP_MODE

// internal use function to compute the dct on a linear array
template <RealType T> void _dct_linear(T *x, int16_t len, T *y);

/*
 * Parked helper kept for the upcoming full FxP audio port.
 * It is intentionally disabled in the current mixed float/FxP bridge flow.
 */
#if 0
template <RealType T>
void _cmplx_mag(kiss_fft_cpx *x, int16_t len, T *res){
    for(int16_t i=0; i<len; i++){
        res[i] = sqrtreal((x[i].r * x[i].r) + (x[i].i * x[i].i));
    }
}
#endif

template <RealType T> void stft(const T *x, int16_t len, int16_t n_frames, T *res) {
    RA_LOG_ARRAY("AUDIO_MEL", "stft", "sig_input", x, len);

    // apply padding
    int16_t padded_len = (2 * PAD_LEN) + len;
    T *padded = (T *)malloc(padded_len * sizeof(T));
    // zero_padding(x, len, PAD_LEN, padded);
    reflect_padding(x, len, PAD_LEN, padded);

    T *column = (T *)malloc(N_FFT * sizeof(T));

    // initialize RFFT structures
    kiss_fftr_cfg<T> cfg = kiss_fftr_alloc<T>(N_FFT, 0, 0, 0);
    kiss_fft_cpx<T> *cx_out = (kiss_fft_cpx<T> *)malloc(FFT_RES_LEN * sizeof(kiss_fft_cpx<T>));
    T *fft_res = (T *)malloc(FFT_RES_LEN * sizeof(T));
    T *fft_re = (T *)malloc(FFT_RES_LEN * sizeof(T));
    T *fft_im = (T *)malloc(FFT_RES_LEN * sizeof(T));

    for (int16_t i = 0; i < n_frames; i++) {

        // get the i-th column of the padded input (i.e. i-th frame)
        for (int16_t j = 0; j < N_FFT; j++) {
            column[j] = padded[j + (i * HOP_LEN)];
        }

        // apply the Hann window
        vect_mult(column, hann_mfcc_wind<T>, N_FFT, column);

        RA_LOG_ARRAY("AUDIO_MEL", "stft", "windowed_frame", column, N_FFT);

        kiss_fftr(cfg, column, cx_out);
        for (int16_t j = 0; j < FFT_RES_LEN; j++) {
            fft_re[j] = cx_out[j].r;
            fft_im[j] = cx_out[j].i;
        }

#ifdef RANGE_ANALYSIS
        RA_LOG_ARRAY("AUDIO_MEL", "stft", "re", fft_re, FFT_RES_LEN);
        RA_LOG_ARRAY("AUDIO_MEL", "stft", "im", fft_im, FFT_RES_LEN);
#endif

        for (int16_t j = 0; j < FFT_RES_LEN; j++) {
            column[j] = sqrtreal((fft_re[j] * fft_re[j]) + (fft_im[j] * fft_im[j]));
        }
        RA_LOG_ARRAY("AUDIO_MEL", "stft", "cmplx_mag", column, FFT_RES_LEN);

        vect_mult(column, column, FFT_RES_LEN, column); // element-wise power of 2
        RA_LOG_ARRAY("AUDIO_MEL", "stft", "frames_power", column, FFT_RES_LEN);

        // stores the current column result into the result array
        // Note that the each column is stored sequentially
        //
        // res = [.... COLUMN 0 ....|.... COLUMN 1 ....|....]
        for (int16_t j = 0; j < FFT_RES_LEN; j++) {
            res[(i * FFT_RES_LEN) + j] = column[j];
        }
    }

    free(cfg);
    free(cx_out);
    free(padded);
    free(column);
    free(fft_res);
    free(fft_re);
    free(fft_im);
#ifdef FIXED_POINT
    free(column_q);
#endif
}

template <RealType T> void mel_spectrogram_full(const T *x, int16_t len, int16_t n_frames, T *res) {
    // STFT
    T *frames_power = (T *)malloc((FFT_RES_LEN * n_frames) * sizeof(T));
    stft(x, len, n_frames, frames_power);

    // MULT BY MEL BASIS (matrix multiplication)
    // frames_power is saved one column after the other
    // the result matrix is saved one row after the other
    for (int16_t i = 0; i < MEL_ROWS; i++) {     // rows in the mel basis
        for (int16_t j = 0; j < n_frames; j++) { // columns for frames_powers and raws for mel_basis
            for (int16_t k = mel_nz_indexes[i][0]; k < mel_nz_indexes[i][1]; k++) { // columns in the mel basis
                res[(i * n_frames) + j] +=
                    mel_basis<T>[i][k - mel_nz_indexes[i][0]] * frames_power[(j * MEL_COLUMNS) + k];
            }
        }
    }

    RA_LOG_ARRAY("AUDIO_MEL", "mel_spectrogram_full", "output", res, MEL_ROWS * n_frames);

    free(frames_power);
}

template <RealType T> void mel_spectrogram(const T *x, int16_t len, int16_t n_frames, uint8_t *idx_required, T *res) {
    // STFT
    T *frames_power = (T *)malloc((FFT_RES_LEN * n_frames) * sizeof(T));
    stft(x, len, n_frames, frames_power);

    uint8_t current_idx = 0;

    // MULT BY MEL BASIS (matrix multiplication)
    // frames_power is saved one column after the other
    // the result matrix is saved one row after the other
    for (int16_t i = 0; i < MEL_ROWS; i++) { // rows in the mel basis

        // Take only the requried ones
        if (i == idx_required[current_idx]) {

            for (int16_t j = 0; j < n_frames; j++) { // columns for frames_powers and raws for mel_basis
                for (int16_t k = mel_nz_indexes[i][0]; k < mel_nz_indexes[i][1]; k++) { // columns in the mel basis
                    res[(current_idx * n_frames) + j] +=
                        mel_basis<T>[i][k - mel_nz_indexes[i][0]] * frames_power[(j * MEL_COLUMNS) + k];
                }
            }

            current_idx++;
        }
    }

    RA_LOG_ARRAY("AUDIO_MEL", "mel_spectrogram", "output", res, current_idx * n_frames);

    free(frames_power);
}

template <RealType T> void power_to_dB(T *x, int16_t len, T *res) {
    RA_LOG_ARRAY("AUDIO_MEL", "power_to_dB", "input", x, len);

    T sample = CONST_ZERO;
    for (int16_t i = 0; i < len; i++) {
        if (x[i] == 0) {
            sample = F_MIN;
        } else {
            sample = x[i];
        }
        res[i] = CONST_TEN * log10real(sample);
    }
    T max = vect_max_value(res, len);
    RA_LOG_SCALAR("AUDIO_MEL", "power_to_dB", "max_dB", max);

    for (int16_t i = 0; i < len; i++) {
        if ((max - TOP_DB) > res[i]) {
            res[i] = (max - TOP_DB);
        }
    }
    RA_LOG_ARRAY("AUDIO_MEL", "power_to_dB", "output", res, len);
}

/// @brief Computes the Direct Cosine Transform on a single dimentional array
/// @param *x   pointer to the input signal
/// @param len  lenght
/// @param *y   pointer to the result
template <RealType T> void _dct_linear(T *x, int16_t len, T *y) {
    RA_LOG_ARRAY("AUDIO_MEL", "dct_linear", "input", x, len);

    T sum = CONST_ZERO;
    T scaling = sqrtreal(1.0f / (2 * len));

    T cos_v = CONST_ZERO;

    T *dct_cos_buf = (T *)malloc(len * sizeof(T));

    for (int16_t k = 0; k < len; k++) {
        read_flash(&dct_cos<T>[(len * k)], dct_cos_buf, len * sizeof(T));

        sum = CONST_ZERO;
        for (int16_t n = 0; n < len; n++) {
            cos_v = dct_cos_buf[n];
            sum += x[n] * cos_v;
        }
        y[k] = sum * CONST_TWO;

        if (k > 0) {
            y[k] = y[k] * scaling;
        }
    }
    y[0] = y[0] * sqrtreal(1.0f / (4 * len));

    RA_LOG_ARRAY("AUDIO_MEL", "dct_linear", "output", y, len);
}

template <RealType T> void dct_matrix(T *x, int16_t rows, int16_t cols, T *y) {
    T *column = (T *)malloc(rows * sizeof(T));
    T *c_res = (T *)malloc(rows * sizeof(T));

    for (int16_t c = 0; c < cols; c++) {
        // fill the column array with the current column
        for (int16_t r = 0; r < rows; r++) {
            column[r] = x[(r * cols) + c];
        }

        _dct_linear(column, rows, c_res);

        // copy the result on the output
        for (int16_t r = 0; r < rows; r++) {
            y[(r * cols) + c] = c_res[r];
        }
    }

    free(column);
    free(c_res);
}

template <RealType T> void entropy(T *spectrogram, int16_t n_rows, int16_t n_columns, T *res) {
    RA_LOG_ARRAY("AUDIO_MEL", "entropy", "input", spectrogram, n_rows * n_columns);

    for (int8_t i = 0; i < n_rows; i++) {
        // Sum each column of the spectrogram
        T row_sum = vect_sum(&spectrogram[i * n_columns], n_columns);
        RA_LOG_SCALAR("AUDIO_MEL", "entropy", "row_sum", row_sum);

        // Divide all the row's elements by the sum of the row.
        // Passing the same pointer for input and output --> the input is modified
        vect_div_const(&spectrogram[i * n_columns], n_columns, row_sum, &spectrogram[i * n_columns]);
        RA_LOG_ARRAY("AUDIO_MEL", "entropy", "normalized_row", &spectrogram[i * n_columns], n_columns);
    }

    // Compute the entropy
    entropy_calc(spectrogram, (n_rows * n_columns), 1);
    RA_LOG_ARRAY("AUDIO_MEL", "entropy", "neg_p_ln_p", spectrogram, n_rows * n_columns);

    // Sum each column of the spectrogram
    for (int8_t i = 0; i < n_rows; i++) {
        res[i] = vect_sum(&spectrogram[i * n_columns], n_columns);
        RA_LOG_SCALAR("AUDIO_MEL", "entropy", "result", res[i]);
    }
}

#endif /* !FXP_MODE */
