#include <math.h>
#include <stdlib.h>

#include <arithmetic.h>
#include <range_analysis.h>
#include <types.h>

// Costant used in the kurtosis computation using the
// Fisher definition
const sreal_t KURT_FISHER_CONST = sreal_t(3.0f);

// This serves as support for the argsort function
// It stores both values and indexes
typedef struct {
    uint16_t value;
    uint16_t idx;
} argsort_struct;

static void print_float(float f, int precision) {
    if (f < 0) {
        printf("-");
        f = -f;
    }

    int ipart = (int)f;
    float fpart = f - (float)ipart;
    int multiplier = 1;
    for (int i = 0; i < precision; i++) {
        multiplier *= 10;
    }
    int fpart_int = (int)(fpart * multiplier + 0.5f);
    if (fpart_int >= multiplier) {
        ipart += 1;
        fpart_int = 0;
    }
    printf("%d.", ipart);
    int temp = fpart_int;
    int digits_printed = 0;
    if (temp == 0) {
        digits_printed = 1;
    } else {
        while (temp > 0) {
            digits_printed++;
            temp /= 10;
        }
    }
    for (int i = 0; i < (precision - digits_printed); i++) {
        printf("0");
    }
    printf("%d", fpart_int);
}

/*
    Helper function for the maximum value and relative index computation
    It stores to max_value and max_index the maximum value and relative index
    found in the array x, respectively
*/

/// @brief Helper function for the maximum value and relative index computation
/// It stores to max_value and max_index the maximum value and relative index
/// found in the array x, respectively
/// @param *x   pointer to the inpug array
/// @param len  lenght of the array
/// @param *max_value   max value
/// @param *max_index   index of the max value
template <RealType T> static void _find_max(T *x, int16_t len, T *max_value, int16_t *max_index) {
    T max_v = x[0];
    int16_t max_i = 0;
    for (int16_t i = 0; i < len; i++) {
        if (x[i] > max_v) {
            max_v = x[i];
            max_i = i;
        }
    }

    *max_index = max_i;
    *max_value = max_v;
}

template <RealType T> static void vect_div_const(T *x, int16_t len, T divisor, T *res) {
    for (uint16_t i = 0; i < len; i++) {
        res[i] = x[i] / divisor;
    }
}

template <RealType T> static T vect_sum(const T *x, int16_t len) {
    quire_t q;
    q.clear();
    for (int16_t i = 0; i < len; i++) {
        q.add_mul(CONST_ONE, x[i]);
    }
    return q.round();
}

template <RealType T> static T vect_mean(const T *x, int16_t len) {
    return vect_sum(x, len) / len;
}

template <RealType T> static void vect_mult(T *x, const T *y, int16_t len, T *r) {
    for (int16_t i = 0; i < len; i++) {
        r[i] = x[i] * y[i];
    }
}

template <RealType T> static T vect_std(T *x, int16_t len) {
    T mean = vect_mean(x, len);
    quire_t q;
    q.clear();

    for (int16_t i = 0; i < len; i++) {
        T centered = x[i] - mean;
        RA_IMU_LOG_SCALAR("vect_std", "x_minus_mean", centered);
        // T sq_dev = centered * centered;
        // RA_IMU_LOG_SCALAR("vect_std", "sq_dev", sq_dev);
        q.add_mul(centered, centered);
    }
    T sum = q.round();

    RA_IMU_LOG_SCALAR("vect_std", "sum_sq_dev", sum);
    T variance = sum / len;
    RA_IMU_LOG_SCALAR("vect_std", "variance", variance);
    T result = sqrtreal(variance);
    RA_IMU_LOG_SCALAR("vect_std", "result", result);
    return result;
}

template <RealType T> static void vect_copy(const T *in, int16_t start, int16_t len, T *out) {
    for (int16_t i = 0; i < len; i++) {
        out[i] = in[i + start];
    }
}

template <RealType T> static void sub_constant(const T *x, int16_t len, T constant, T *res) {
    for (int16_t i = 0; i < len; i++) {
        res[i] = x[i] - constant;
    }
}

template <RealType T> static int16_t vect_max_index(T *x, int16_t len) {
    int16_t max_i = 0;
    T temp_v;
    _find_max(x, len, &temp_v, &max_i);
    return max_i;
}

template <RealType T> static T vect_max_value(T *x, int16_t len) {
    int16_t max_i = 0;
    T max_v;
    _find_max(x, len, &max_v, &max_i);
    return max_v;
}

template <RealType T> static T vect_max_abs_value(T *x, int16_t len) {
    T max_abs = x[0];
    T tmp = CONST_ZERO;
    for (int16_t i = 1; i < len; i++) {
        tmp = fabs(x[i]);
        if (tmp >= max_abs) {
            max_abs = tmp;
        }
    }

    return max_abs;
}

template <RealType T> static void normalize_max(T *x, int16_t len, T *res) {
    T max;
    int16_t max_i;
    _find_max(x, len, &max, &max_i);

    for (int16_t i = 0; i < len; i++) {
        res[i] = x[i] / max;
    }
}

/**
 * Comparator function to be passed to qsort() function.
 * It works with a structure having values and indexes, it has to be used for the argsort implementation
 */
static int _q_argsort__cmp(const void *e1, const void *e2) {
    argsort_struct *val_1 = (argsort_struct *)e1;
    argsort_struct *val_2 = (argsort_struct *)e2;

    if ((*val_1).value > (*val_2).value) {
        return 1;
    }
    if ((*val_1).value < (*val_2).value) {
        return -1;
    }
    return 0;
}

static void argsort(uint16_t *arr, uint16_t len, uint16_t *sort_idxs) {
    argsort_struct *elems = (argsort_struct *)malloc(len * sizeof(argsort_struct));

    for (uint16_t i = 0; i < len; i++) {
        elems[i].value = arr[i];
        elems[i].idx = i;
    }

    qsort(elems, len, sizeof(argsort_struct), _q_argsort__cmp);

    for (uint16_t i = 0; i < len; i++) {
        sort_idxs[i] = elems[i].idx;
    }

    free(elems);
}

template <typename T> void order_by_idxs(T *arr_in, uint16_t len, uint16_t *idxs) {
    T *tmp = (T *)malloc(len * sizeof(T));

    for (uint16_t i = 0; i < len; i++) {
        // printf(">  %d\n", i);
        tmp[i] = arr_in[idxs[i]];
    }

    // Copies the idex-ordered tmp array back in place into arr
    vect_copy(tmp, 0, len, arr_in);

    free(tmp);
}

/// @brief Helper function that implements the definition of the composite Simpson's integral.
/// This function is not callable externally.
/// @param *x       pointer to the input signal
/// @param spacing  spacing
/// @param start    start index
/// @param end      end index
/// @return
template <RealType T> T _simpson_step(T *x, T spacing, int16_t start, int16_t end) {
    int n_intervals = (end - start) / 2; // realber of intervals (h in the formula)

    T sum = CONST_ZERO;
    int interval_start = start;

    // computes the indexes
    for (int i = 0; i < n_intervals; i++) {
        sum += x[interval_start] + CONST_FOUR * x[interval_start + 1] + x[interval_start + 2];
        interval_start = interval_start + 2;
    }
    return (sum * (spacing / CONST_THREE));
}

template <RealType T> T simpson(T *x, int16_t len, T spacing) {
    T result = CONST_ZERO;

    if (len % 2 == 0) {
        T val = CONST_ZERO;
        val += spacing * (x[len - 1] + x[len - 2]) / CONST_TWO;
        result += _simpson_step(x, spacing, 0, len - 1);

        val += spacing * (x[0] + x[1]) / CONST_TWO;
        result += _simpson_step(x, spacing, 1, len);

        val /= CONST_TWO;
        result /= CONST_TWO;
        result = result + val;
    } else {
        result = _simpson_step(x, spacing, 0, len);
    }
    return result;
}

template <RealType T> void padding(const T *sig, int len, int padlen, T *res) {
    T left_end = sig[0];
    T right_end = sig[len - 1];

    T *left_ext = (T *)malloc(padlen * sizeof(T));
    T *right_ext = (T *)malloc(padlen * sizeof(T));

    // compute and append padding for the left side
    for (int i = 0; i < padlen; i++) {
        left_ext[i] = sig[padlen - i];
        right_ext[i] = sig[len - 2 - i];

        res[i] = (CONST_TWO * left_end) - left_ext[i];
    }

    // copy the original signal in the central part of the result
    vect_copy(sig, 0, len, &res[padlen]);

    // computes and append padding for the right side
    for (int i = padlen + len; i < (padlen * 2) + len; i++) {
        res[i] = (CONST_TWO * right_end) - right_ext[i - (padlen + len)];
    }

    free(left_ext);
    free(right_ext);
}

template <RealType T> void zero_padding(const T *x, int16_t len, int16_t side_pad_len, T *r) {
    // pad with zeros in front and in the end
    for (int16_t i = 0; i < side_pad_len; i++) {
        r[i] = CONST_ZERO;
        r[i + side_pad_len + len] = CONST_ZERO;
    }

    // copy input vector
    vect_copy(x, 0, len, &r[side_pad_len]);
}

template <RealType T> void reflect_padding(const T *x, uint16_t len, uint16_t side_pad_len, T *r) {
    // Adds the padding to head and tail of the array by reversing the original samples
    for (uint16_t i = 0; i < side_pad_len; i++) {
        r[i] = x[side_pad_len - i];                 // head
        r[side_pad_len + len + i] = x[len - 2 - i]; // tail
    }

    // Copies the original array in the center of the result
    vect_copy(x, 0, len, &r[side_pad_len]);
}

template <RealType T> T get_line_length(T *x, int16_t len) {
    T sum = CONST_ZERO;

    for (int16_t i = 0; i < len - 1; i++) {
        sum += fabs(x[i + 1] - x[i]);
        RA_IMU_LOG_SCALAR("get_line_length", "diff", fabs(x[i + 1] - x[i]));
    }

    RA_IMU_LOG_SCALAR("get_line_length", "accum", sum);
    T result = sum / (len - 1);
    RA_IMU_LOG_SCALAR("get_line_length", "result", result);
    return result;
}

template <RealType T> T get_kurtosis(T *x, int16_t len) {
    T std = vect_std(x, len);
    T mean = vect_mean(x, len);

    RA_IMU_LOG_SCALAR("get_kurtosis", "mean", mean);
    RA_IMU_LOG_SCALAR("get_kurtosis", "std", std);

    T sum = CONST_ZERO;
#ifdef RANGE_ANALYSIS
    T moment_max = CONST_ZERO;
#endif

    for (int16_t i = 0; i < len; i++) {
        T tmp = (x[i] - mean) * (x[i] - mean);
        T x4 = tmp * tmp;
#ifdef RANGE_ANALYSIS
        if (x4 > moment_max)
            moment_max = x4;
#endif
        sum += x4;
    }

#ifdef RANGE_ANALYSIS
    RA_IMU_LOG_SCALAR("get_kurtosis", "moment_max", moment_max);
#endif
    RA_IMU_LOG_SCALAR("get_kurtosis", "sum_x4", sum);

    T std4 = std * std * std * std;
    RA_IMU_LOG_SCALAR("get_kurtosis", "std4", std4);

    T result = (sum / (len * std4)) - KURT_FISHER_CONST;
    RA_IMU_LOG_SCALAR("get_kurtosis", "result", result);
    return result;
}

template <RealType T> T L2_norm(const T *x, int16_t len) {
    T sum = CONST_ZERO;

    for (int16_t i = 0; i < len; i++) {
        sum += x[i] * x[i];
    }

    RA_IMU_LOG_SCALAR("L2_norm", "sum_sq", sum);
    T result = sqrtreal(sum);
    RA_IMU_LOG_SCALAR("L2_norm", "result", result);
    return result;
}

static inline uint16_t min(uint16_t a, uint16_t b) {
    if (a < b)
        return a;
    else
        return b;
}

template <RealType T> void entropy_calc(T *x, int16_t len, uint8_t base) {
    for (int16_t i = 0; i < len; i++) {
        if (x[i] > CONST_ZERO) {
            x[i] = CONST_NEG_ONE * x[i] * logreal(x[i]);
        } else if (x[i] < CONST_ZERO) {
            x[i] = F_MIN;
        }
    }

    // If base needed, change the base of the logarithm result
    if (base != 1) {
        for (int16_t i = 0; i < len; i++) {
            if (x[i] > F_MIN) {
                x[i] /= logreal(base);
            }
        }
    }
}
