#ifndef _AZC_H_
#define _AZC_H_

#include <inttypes.h>


/**
 * Computes the AZC (Approximate Zero Crossing) feature of the given signal
 * using the approximation of it with tolerance epsilon.
 * 
 * @param *sig: pointer to the signal
 * @param len: length of the signal
 * @param epsilon: epsiln tolerance values used to approximate the signal with the Douglas-Peucker algorithm
 * 
 * @return the number of times the differentaition of the approximated signal crosses the 0-axis
*/
int16_t azc_computation(float *sig, int16_t len, float epsilon);

#endif