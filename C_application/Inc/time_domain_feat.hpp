#include <math.h>
#include <stdlib.h>

#include <audio_features.h>
#include <feature_extraction.h>
#include <filtering.h>
#include <filters_parameters.h>
#include <helpers.h>
#include <imu_features.h>
#include <range_analysis.h>
#include <types.h>

// Here I put all the functions to compute time domain features
template <RealType T> int16_t _find_peaks(T *x, int16_t len);

template <RealType T> T get_max(T *sig, int16_t len) {
    T max = sig[0];

    for (int16_t i = 1; i < len; i++) {
        if (sig[i] > max) {
            max = sig[i];
        }
    }
    RA_IMU_LOG_SCALAR("get_max", "get_max", max);
    return max;
}

template <RealType T> T get_crest(T *sig, int16_t len, T rms) {
    T peak = get_max(sig, len);
    T crest_factor = peak / rms;
    return crest_factor;
}

template <RealType T> void sub_mean(const T *sig, T *res, int16_t len) {
    T mean = vect_mean(sig, len);

    sub_constant(sig, len, mean, res);
}

template <RealType T, RealType S = T> T get_rms(T *sig, int16_t len) {
    // NOTE: regarding the use of S type and no quire
    // at baseline SE 0.6029 PR 0.8116 F1 0.6919 FP/hr 90.7 TP 659 FP 153 FN 434
    // if we replace `sum` with a quire-based accumulator, we would get result
    // SE 0.6002 PR 0.8059 F1 0.6880 FP/hr 93.6 TP 656 FP 158 FN 437
    // where both SE and PR dropped, and both FP and FN increased, i.e. a net loss
    // it is likely caused by the threshold values being tuned with rounding error baked in
    // and if we increase precision to reduce rounding error, we actually drift away from the threshold
    // thus we keep S-type and do not replace it with quire, unless threshold is recalibrated with quire

    // option 1: use S type and no quire (better eval / worse performance)
    // S sum = 0;
    // for (int16_t i = 0; i < len; i++) {
    //     T sq = sig[i] * sig[i];
    //     RA_IMU_LOG_SCALAR("get_rms", "sig_sq", sq);
    //     sum += (S)sq;
    // }

    // option 2: use quire (worse eval / better performance)
    quire_t q;
    q.clear();
    for (int16_t i = 0; i < len; i++) {
        q.add_mul(sig[i], sig[i]);
    }
    T sum = q.round();

    RA_IMU_LOG_SCALAR("get_rms", "sum_sq", sum);
    T result = sqrtreal(sum / len);
    RA_IMU_LOG_SCALAR("get_rms", "result", result);
    return result;
}

template <RealType T> T compute_zrc(T *sig, int16_t len) {
    int sum = 0;
    T interm_product = CONST_ZERO;

    for (int16_t i = 0; i < len - 1; i++) {
        interm_product = sig[i] * sig[i + 1];
        RA_IMU_LOG_SCALAR("compute_zrc", "multiplier", interm_product);
        if (interm_product < CONST_ZERO) {
            sum++;
        }
    }
    T result = (T)sum / (len - 1);
    RA_IMU_LOG_SCALAR("compute_zrc", "result", result);
    return result;
}

/*
    Helper function to count the number of local maxima in a signal.
    This function mimics the _local_maxima_1d() function in the python
    "scipy" module.
*/
template <RealType T> int16_t _find_peaks(T *x, int16_t len) {
    int16_t i = 1; // points to the first considered sample (the second one)
    int16_t i_max = len - 1;
    int16_t i_ahead = 0;

    int16_t npeaks = 0;

    while (i < i_max) {
        if (x[i - 1] < x[i]) {
            i_ahead = i + 1;

            while (i_ahead < i_max && x[i_ahead] == x[i]) {
                i_ahead++;
            }

            if (x[i_ahead] < x[i]) {
                npeaks++;
                i = i_ahead;
            }
        }
        i++;
    }

    return npeaks;
}

template <RealType T> void eepd(const T *sig, int16_t len, int16_t fs, const int8_t *select, int16_t *res) {
    T *interm =
        (T *)malloc(len * sizeof(T)); // to store the intermediate result between the first and the second filter
    T *filtered = (T *)malloc(len * sizeof(T)); // temporary to store the result of each filter

    const T *b, *a, *zi;

    for (int16_t i = 0; i < N_EEPD; i++) {
        if (select[i] == 1) {

            b = filters_parameters<T>.filters[i].b;
            a = filters_parameters<T>.filters[i].a;
            zi = filters_parameters<T>.filters[i].zi;

            filtfilt(sig, len, b, a, zi, interm);
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "bandpass_out", interm, len);

            vect_mult(interm, interm, len, interm); // squared vector
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "squared", interm, len);

            filtfilt(interm, len, b_second<T>, a_second<T>, zi_second<T>, filtered);
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "envelope", filtered, len);

            normalize_max(filtered, len, filtered); // divide each number by the maximum
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "normalized", filtered, len);

            res[i] = _find_peaks(filtered, len);
            RA_LOG_SCALAR("AUDIO_EEPD", "eepd", "n_peaks", (T)res[i]);
        }
    }

    free(interm);
    free(filtered);
}
