#include <stdlib.h>

#include <filters_parameters.h>
#include <helpers.h>
#include <range_analysis.h>
#include <types.h>

template <RealType T>
void linear_filer(T *sig, int len, const T *b, const T *a, T *zi, T *res) {
    RA_LOG_ARRAY("AUDIO_EEPD", "linear_filter", "input", sig, len);

    T sig_1 = 0.0;
    T y_1 = 0.0;

    T s_1 = zi[0];
    T s_2 = zi[1];

    // First one needs to be done outside the loop in order not to
    // update the states (s_1 and s_2)
    res[0] = b[0] * sig[0] + s_1;
    y_1 = res[0];
    sig_1 = sig[0];

    for (int i = 1; i < len; i++) {
        // s_1 is updated before and s_2 after. This is because at every
        // iteration we need the filter state 1 (s_1) of the current step and
        // the filter state 2 (s_2) of the previous step.
        // Also by doing so we avoid using 2 extra variables for the signal
        // and output values of 2 steps before

        s_1 = b[1] * sig_1 - a[1] * y_1 + s_2;
        res[i] = b[0] * sig[i] + s_1;
        s_2 = b[2] * sig_1 - a[2] * y_1;

        sig_1 = sig[i];
        y_1 = res[i];
    }

    RA_LOG_ARRAY("AUDIO_EEPD", "linear_filter", "output", res, len);
}

template <RealType T>
void filtfilt(const T *sig, int len, const T *b, const T *a, const T *zi,
              T *res) {
    RA_LOG_ARRAY("AUDIO_EEPD", "filtfilt", "input", sig, len);

    // PADDING //
    int padded_len = (2 * PADLEN) + len;
    T *pad = (T *)malloc(padded_len * sizeof(T));
    padding(sig, len, PADLEN, pad);

    // GET INITIAL STATE //
    T x0 = pad[0];

    // FILTER FORWARD //
    // intermediate output, cannot be res because it's not padded
    T *intermediate = (T *)malloc(padded_len * sizeof(T));

    T *initial = (T *)malloc(2 * sizeof(T));
    initial[0] = zi[0] * T(x0);
    initial[1] = zi[1] * T(x0);
    linear_filer(pad, padded_len, b, a, initial, intermediate);

    // FILTER BACKWARD //
    // for the reverse filtering
    T *reverse = (T *)malloc(padded_len * sizeof(T));
    for (int i = 0; i < padded_len; i++) {
        reverse[i] = intermediate[padded_len - 1 - i];
    }

    initial[0] = zi[0] * T(reverse[0]);
    initial[1] = zi[1] * T(reverse[0]);

    T *res_padded = (T *)malloc(padded_len * sizeof(T));
    linear_filer(reverse, padded_len, b, a, initial, res_padded);

    // CUT THE PADDING TO GET THE FINAL RESULT //
    for (int i = 0; i < len; i++) {
        // cut the padding and reverse
        res[i] = res_padded[PADLEN + len - 1 - i];
    }

    RA_LOG_ARRAY("AUDIO_EEPD", "filtfilt", "output", res, len);

    free(pad);
    free(intermediate);
    free(res_padded);
    free(initial);
    free(reverse);
}
