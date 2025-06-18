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

/**
 * Divides all the elements of the input array by the same divisor.
 * 
 * @param *x        : pointer to the input array
 * @param len       : length of the input array
 * @param divisor   : number to be used as a diviros for the elements of the array
*/
void vect_div_const(float *x, int16_t len, float divisor, float *res);


/**
 * Returns the sum of all the values of the input array.
 * 
 * @param *x    : pointer to the input array
 * @param len   : lenght of the input array
*/
float vect_sum(const float *x, int16_t len);



/**
 * Returns the mean of a sequence of numbers

* @param *x    : pointer to the input array
* @param len   : lenght of the input array
 */
float vect_mean(const float *x, int16_t len);


/// @brief Computes the element-wise multiplication between x and y arrays,
/// the multiplication is stored in r array.
/// @param *x   pointer to the first vector 
/// @param *y   pointer to the second vector
/// @param len  lenght of the vectors
/// @param *r   pointer to the resulting vectors
void vect_mult(float *x, const float *y, int16_t len, float *r);



/// @brief Returns the standard deviation of the given input array
/// @param *x       pointer to the signal
/// @param len      lenght of the signal
/// @return         the standard deviation
float vect_std(float *x, int16_t len);



/// @brief Copies "len" samples from "in" array of float starting at index "start"
/// Destination is "out"
/// Notice that the samples are taken from the input starting at 
/// index "start" but are placed in output from index 0!
/// @param *in      pointer to the input signal
/// @param start    start index
/// @param len      lenght to copy
/// @param *out     poitner to the output array
void vect_copy(const float *in, int16_t start, int16_t len, float *out);



/// @brief Copies "len" samples from "in" array of uint16_t starting at index "start"
/// Destination is "out"
/// Notice that the samples are taken from the input starting at 
/// index "start" but are placed in output from index 0!
/// @param *in      pointer to the input signal
/// @param start    start index
/// @param len      lenght to copy
/// @param *out     poitner to the output array
void vect_copy_uint16_t(uint16_t *in, int16_t start, int16_t len, uint16_t *out);



/// @brief Subtract a constant to each element of an input array and stores in into res array
/// @param *x           pointer to the input signal
/// @param len          lenght of the signal
/// @param constant     constant value to subtract
/// @param *res         pointer to the result
void sub_constant(const float *x, int16_t len, float constant, float *res);



/// @brief Returns the index at which the maximum value of array x is found
/// @param *x   pointer to the inpug array
/// @param len  lenght of the array
/// @return     index of the maximum value
int16_t vect_max_index(float *x, int16_t len);



/// @brief Returns the maximum value within an array
/// @param *x   pointer to the inpug array
/// @param len  lenght of the array
/// @return     max value
float vect_max_value(float *x, int16_t len);


/**
 * Returns the maximum absoulate value of the given array
 * 
 * @param *x    :   pointer to the input array
 * @param len   :   length of the input array
 * @return      :   maximum absolute value in the input array   
*/
float vect_max_abs_value(float *x, int16_t len);



/// @brief Divides every element in the specified "x" array by the maximum value.
/// @param *x   pointer to the inpug array
/// @param len  lenght of the array 
/// @param *res pointer to the resulting array
void normalize_max(float *x, int16_t len, float *res);



/// @brief Returns the integral of signal x coputed using the composite 
/// Simpson's rule. The spacing of the samples is defined by the
/// "spacing" parameter.
/// If the number of samples is even, the integral is averaged
/// @param *x   pointer to the inpug array
/// @param len  lenght of the array 
/// @param spacing  spacing for the integral
/// @return integral of the signal
float simpson(float *x, int16_t len, float spacing);



/// @brief Applies a padding to the specified signal, it appends 
/// "padlen" elements to the left and to the right.
/// The padded values are computed differently for the 
/// left and the right ends. 
/// @param *sig     pointer to the input signal
/// @param len      lenght of the signal
/// @param padlen   padding lenght
/// @param *res     pointer to the result
void padding(const float *sig, int len, int padlen, float *res);



/// @brief Adds a 0 padding to the left and right of the input x
/// The amount of 0 added to each side is specified by side_pad_len
/// @param *x           pointer to the inpug array
/// @param len          lenght of the array 
/// @param side_pad_len lenght of the side padding
/// @param r            pointer to the result
void zero_padding(const float *x, int16_t len, int16_t side_pad_len, float *r);



/**
 * Pads the input array at the start and at the end by reflecting the first `side_pad_len`
 * numbers of the array (the first one excluded).
 * 
 * @param *x: pointer to the input array to be padded
 * @param len: initial length of the input array
 * @param side_pad_len: length of the side paddings. Amount of numbers to add on each side
 * @param *r: pointer to the result array
*/
void reflect_padding(const float *x, uint16_t len, uint16_t side_pad_len, float *r);



/// @brief Computes the line length feature of the specified input array
/// @param *x           pointer to the inpug array
/// @param len          lenght of the array 
/// @return     the line lenght
float get_line_length(float *x, int16_t len);



/// @brief Returns the kurtosis of the given input array
/// @param *x           pointer to the inpug array
/// @param len          lenght of the array 
/// @return     the kurtosis
float get_kurtosis(float *x, int16_t len);



/// @brief Computes the L2 norm of a signal
/// @param *x           pointer to the inpug array
/// @param len          lenght of the array 
/// @return     the L2 norm
float L2_norm(const float *x, int16_t len);



/**
 * Returns the minimum of two `uint16_t` numbers.
 * 
 * @param a     :   first number to compare    
 * @param b     :   second number to compare
*/
uint16_t min(uint16_t a, uint16_t b);



/**
 * Computes the entropy of the input array with the following formula:
 *                   | -xlog(x)    if x > 0.0
 *      entropy(x) = | 0.0         if x = 0.0
 *                   | -inf        if x < 0.0
 * 
 * Notice that the input array is modified in-place, so its content will be
 * changed by the function.
 * 
 * @param *x    :   pointer to the input array
 * @param len   :   length of the input array
 * @param base  :   base of the logarithm. If base is 1, then it's the natural log
 * 
*/
void entropy_calc(float *x, int16_t len, uint8_t base);

#endif