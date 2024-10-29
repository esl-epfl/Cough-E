#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <inttypes.h>


/**
 * Enums of the available types on which to call the `order_by_idxs()` function.
*/
typedef enum type_sort{
    FLOAT_SORT,
    UINT16_T_SORT
} type_sort_t;

/*
    Set of general functions to help the computations of features 
*/

/**
 * Computes the indexes that sort a given array.
 * Note that this function does't sort the initial array!
 * 
 * @param *arr          :   pointer to the array to sort
 * @param len           :   length of the array
 * @param *sort_idxs    :   pointer to the array where to store the indexes that would sort the `arr` array
*/
void argsort(uint16_t *arr, uint16_t len, uint16_t *sort_idxs);

/**
 * Orders an array based on some indexes specified as input parameters.
 * The array has to be of one of the availble types specified by the `type_sort_t` enum, otherwise umpredictable behaviour may occour.
 * Note that the input array is ordered in-place, so its original content will be lost!
 * 
 * @param *arr  :   pointer to the array to order
 * @param len   :   length of the array
 * @param *idxs :   pointer to the array of indexes that will order the input array
 * @param type  :   type of the input array pointed by `arr`
*/
void order_by_idxs(void *arr, uint16_t len, uint16_t *idxs, type_sort_t type);

void vect_div_const(float *x, int16_t len, float divisor, float *res);
float vect_sum(const float *x, int16_t len);
float vect_mean(const float *x, int16_t len);
void vect_mult(float *x, const float *y, int16_t len, float *r);
float vect_std(float *x, int16_t len);
void vect_copy(const float *in, int16_t start, int16_t len, float *out);
void vect_copy_uint16_t(uint16_t *in, int16_t start, int16_t len, uint16_t *out);
void sub_constant(const float *x, int16_t len, float constant, float *res);
int16_t vect_max_index(float *x, int16_t len);
float vect_max_value(float *x, int16_t len);
float vect_max_abs_value(float *x, int16_t len);
void normalize_max(float *x, int16_t len, float *res);
float simpson(float *x, int16_t len, float spacing);
void padding(const float *sig, int len, int padlen, float *res);
void zero_padding(const float *x, int16_t len, int16_t side_pad_len, float *r);
void reflect_padding(const float *x, uint16_t len, uint16_t side_pad_len, float *r);

float get_line_length(float *x, int16_t len);
float get_kurtosis(float *x, int16_t len);
float L2_norm(const float *x, int16_t len);

/**
 * Returns the minimum of two `uint16_t` numbers.
 * 
 * @param a     :   first number to compare    
 * @param b     :   second number to compare
*/
uint16_t min(uint16_t a, uint16_t b);

void entropy_calc(float *x, int16_t len, uint8_t base);

#endif