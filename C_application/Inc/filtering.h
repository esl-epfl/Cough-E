#ifndef _FILTERING_H_
#define _FILTERING_H_

#include <types.h>

/**
 * Applies a linear filter to the signal. The filter is defined by a, b, and zi coefficients
 *
 * @param *sig  : pointer to the signal to be filtered
 * @param len   : lenght of the signal
 * @param *b    : pointer to the b coefficients of the filter
 * @param *a    : pointer to the a coefficients of the filter
 * @param *zi   : pointer to the zi coefficients of the filter
 * @param *res  : pointer to the array where to store the resulting filtered signal
 */
template <RealType T>
void linear_filer(T *sig, int len, const T *b, const T *a, T *zi, T *res);


/**
 *  Applies a filter to the signal.
    The filter has transfer function with coefficients b and a, passed as parameters
    The way this filter is implemented is compliant with the filtfilt() function
    of the python package "scipy".

    b, a and zi are respectively the b, a coefficients of the filter and
    its initial state.

    The main steps are:
    - padding of the signal
    - forward filtering
    - backward filtering
 */
template <RealType T>
void filtfilt(const T *sig, int len, const T *b, const T* a, const T *zi, T *res);

#include <filtering.hpp>

#endif
