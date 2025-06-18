#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <helpers.h>

// Minimum float available (supposing 32-bit float)
#define MIN_FLOAT 1.17549e-038

// Costant used in the kurtosis computation using the
// Fisher definition
#define KURT_FISHER_CONST   3  


// Internal functions to support some computations
void _find_max(float *x, int16_t len, float *max_value, int16_t *max_index);
float _simpson_step(float *x, float spacing, int16_t start, int16_t end);


// This serves as support for the argsort function
// It stores both values and indexes
typedef struct {
    uint16_t value;
    uint16_t idx;
} argsort_struct;

/**
 * Comparator function to be passed to qsort() function.
 * It works with a structure having values and indexes, it has to be used for the argsort implementation
*/
int _q_argsort__cmp(const void *e1, const void *e2){

    argsort_struct *val_1 = (argsort_struct*)e1;
    argsort_struct *val_2 = (argsort_struct*)e2;

    if( (*val_1).value > (*val_2).value ){
        return 1;
    }
    if( (*val_1).value < (*val_2).value ){
        return -1;
    }
    return 0;
}


void argsort(uint16_t *arr, uint16_t len, uint16_t *sort_idxs){

    argsort_struct *elems = (argsort_struct*)malloc(len * sizeof(argsort_struct));

    for(uint16_t i=0; i<len; i++){
        elems[i].value = arr[i];
        elems[i].idx = i;
    }

    qsort(elems, len, sizeof(argsort_struct), _q_argsort__cmp);

    for(uint16_t i=0; i<len; i++){
        sort_idxs[i] = elems[i].idx;
    }

    free(elems);

}


void order_by_idxs(void *arr_in, uint16_t len, uint16_t *idxs, type_sort_t type){

    if(type == FLOAT_SORT){

        float *arr = (float*)arr_in;
        float *tmp = (float*)malloc(len * sizeof(float));  
        
        for(uint16_t i=0; i<len; i++){
            // printf(">  %d\n", i);
            tmp[i] = arr[idxs[i]];
        }

        // Copies the idex-ordered tmp array back in place into arr
        vect_copy(tmp, 0, len, arr);
        
        free(tmp);

    } else if(type == UINT16_T_SORT) {

        uint16_t *arr = (uint16_t*)arr_in;
        uint16_t *tmp = (uint16_t*)malloc(len * sizeof(uint16_t));
        
        for(uint16_t i=0; i<len; i++){
            // printf(">  %d\n", i);
            tmp[i] = arr[idxs[i]];
        }

        // Copies the idex-ordered tmp array back in place into arr
        vect_copy_uint16_t(tmp, 0, len, arr);
        
        free(tmp);
    }
}



void vect_div_const(float *x, int16_t len, float divisor, float *res){
    for(uint16_t i=0; i<len; i++){
        res[i] = x[i] / divisor;
    }
}



float vect_sum(const float *x, int16_t len){
    float sum = 0.0;
    for(int16_t i=0; i<len; i++){
        sum += x[i];
    }
    return sum;
}


float vect_mean(const float *x, int16_t len){
    return vect_sum(x, len) / len;
}


void vect_mult(float *x, const float *y, int16_t len, float *r){
    for(int16_t i=0; i<len; i++){
        r[i] = x[i] * y[i];
    }
}



float vect_std(float *x, int16_t len){
    float mean = vect_mean(x, len);
    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        sum += (x[i] - mean) * (x[i] - mean);
    }

    return sqrtf(sum / len);
}



void vect_copy(const float *in, int16_t start, int16_t len, float *out){
    for(int16_t i=0; i<len; i++){
        out[i] = in[i + start];
    }
}




void vect_copy_uint16_t(uint16_t *in, int16_t start, int16_t len, uint16_t *out){
    for(int16_t i=0; i<len; i++){
        out[i] = in[i + start];
    }  
}



void sub_constant(const float *x, int16_t len, float constant, float *res){
    for(int16_t i=0; i<len; i++){
        res[i] = x[i] - constant;
    }
}



int16_t vect_max_index(float *x, int16_t len){
    int16_t max_i = 0.0;
    float temp_v;
    _find_max(x, len, &temp_v, &max_i);
    return max_i;
}



float vect_max_value(float *x, int16_t len){
    int16_t max_i = 0;
    float max_v;
    _find_max(x, len, &max_v, &max_i);
    return max_v;
}



float vect_max_abs_value(float *x, int16_t len){
    float max_abs = x[0];
    float tmp = 0.0;
    for(int16_t i=1; i<len; i++){
        tmp = fabs(x[i]);
        if(tmp >= max_abs){
            max_abs = tmp;
        }
    }

    return max_abs;
}




void normalize_max(float *x, int16_t len, float *res){
    float max;
    int16_t max_i;
    _find_max(x, len, &max, &max_i);

    for(int16_t i=0; i<len; i++){
        res[i] = x[i] / max;
    }
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
void _find_max(float *x, int16_t len, float *max_value, int16_t *max_index){
    float max_v = x[0];
    int16_t max_i = 0;
    for(int16_t i=0; i<len; i++){
        if(x[i] > max_v){
            max_v = x[i];
            max_i = i;
        }
    }

    *max_index = max_i;
    *max_value = max_v;
}



float simpson(float *x, int16_t len, float spacing){
    float result = 0.0;

    if(len % 2 == 0){
        float val = 0.0;
        val += spacing * (x[len - 1] + x[len - 2]) / 2;
        result += _simpson_step(x, spacing, 0, len-1);

        val += spacing * (x[0] + x[1]) / 2;
        result += _simpson_step(x, spacing, 1, len);

        val /= 2;
        result /= 2;
        result = result + val;
    } else {
        result = _simpson_step(x, spacing, 0, len);
    }
    return result;
}




/// @brief Helper function that implements the definition of the composite Simpson's integral.
/// This function is not callable externally.
/// @param *x       pointer to the input signal
/// @param spacing  spacing
/// @param start    start index
/// @param end      end index
/// @return         
float _simpson_step(float *x, float spacing, int16_t start, int16_t end){
    
    int n_intervals = (end - start) / 2; // number of intervals (h in the formula)

    float sum = 0.0;
    int interval_start = start;

    // computes the indexes
    for (int i = 0; i < n_intervals; i++)
    {
        sum += x[interval_start] + 4 * x[interval_start+1] + x[interval_start+2];
        interval_start = interval_start + 2;
    }
    return (sum * (spacing / 3));
}


void padding(const float *sig, int len, int padlen, float *res){

    float left_end = sig[0];
    float right_end = sig[len-1];

    float *left_ext = (float*)malloc(padlen * sizeof(float));
    float *right_ext = (float*)malloc(padlen * sizeof(float));

    // compute and append padding for the left side
    for(int i=0; i<padlen; i++){
        left_ext[i] = sig[padlen-i]; 
        right_ext[i] = sig[len-2-i];

        res[i] = (2 * left_end) - left_ext[i];
    }

    // copy the original signal in the central part of the result
    vect_copy(sig, 0, len, &res[padlen]);

    // computes and append padding for the right side
    for(int i=padlen+len; i<(padlen*2)+len; i++){
        res[i] = (2 * right_end) - right_ext[i - (padlen + len)];
    }

    free(left_ext);
    free(right_ext);
}




void zero_padding(const float *x, int16_t len, int16_t side_pad_len, float *r){

    // pad with zeros in front and in the end
    for(int16_t i=0; i<side_pad_len; i++){
        r[i] = 0.0;
        r[i+side_pad_len+len] = 0.0;
    }

    // copy input vector
    vect_copy(x, 0, len, &r[side_pad_len]);
}



void reflect_padding(const float *x, uint16_t len, uint16_t side_pad_len, float *r){


    // Adds the padding to head and tail of the array by reversing the original samples
    for(uint16_t i=0; i<side_pad_len; i++){
        r[i] = x[side_pad_len-i];           // head
        r[side_pad_len+len+i] = x[len-2-i]; // tail
    }

    // Copies the original array in the center of the result
    vect_copy(x, 0, len, &r[side_pad_len]);
}




float get_line_length(float *x, int16_t len){

    float sum = 0.0;

    for(int16_t i=0; i<len-1; i++){
        sum += fabs(x[i+1] - x[i]);
    }

    return sum / (len-1);
}




float get_kurtosis(float *x, int16_t len){

    float std = vect_std(x, len);
    float mean = vect_mean(x, len);

    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        register float tmp = (x[i] - mean) * (x[i] - mean);
        sum += tmp * tmp;
    }

    return (sum / (len * pow(std, 4))) - KURT_FISHER_CONST;

}


float L2_norm(const float *x, int16_t len){
    
    float sum = 0.0;

    for(int16_t i=0; i<len; i++){
        sum += x[i] * x[i];
    }

    return sqrtf(sum);
}



uint16_t min(uint16_t a, uint16_t b){
    if(a < b)
        return a;
    else
        return b;
}



void entropy_calc(float *x, int16_t len, uint8_t base){

    for(int16_t i=0; i<len; i++){
        if(x[i] > 0.0){
            x[i] = -1.0 * x[i] * logf(x[i]);
        }
        else if(x[i] < 0.0){
            x[i] = MIN_FLOAT;
        }
    }

    // If base needed, change the base of the logarithm result
    if(base != 1){
        for(int16_t i=0; i<len; i++){
            if(x[i] > MIN_FLOAT){
                x[i] /= logf(base);
            }
        }
    }
}