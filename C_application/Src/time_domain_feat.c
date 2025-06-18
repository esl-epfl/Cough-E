#include <stdio.h>
#include <math.h>

#include <feature_extraction.h>
#include <time_domain_feat.h>
#include <helpers.h>
#include <filters_parameters.h>
#include <filtering.h>

#include <audio_features.h>


// Here I put all the functions to compute time domain features

int16_t _find_peaks(float *x, int16_t len);




float get_max(float *sig, int16_t len){
    
    float max = sig[0];

    for(int16_t i=1; i<len; i++){
        if(sig[i] > max){
            max = sig[i];
        }
    }
    return max;
}



void sub_mean(const float *sig, float *res, int16_t len){

    float mean = vect_mean(sig, len);

    sub_constant(sig, len, mean, res);
}


float get_rms(float *sig, int16_t len){
    float sum = 0;
    for(int16_t i=0; i<len; i++){
        sum += sig[i] * sig[i];
    }
    return sqrtf(sum / len);
}




float compute_zrc(float *sig, int16_t len){

    int sum = 0;
    float interm_product = 0;

    for(int16_t i=0; i<len-1; i++){
        interm_product = sig[i] * sig[i + 1];
        if(interm_product < 0){
            sum++;
        }
    }
    return (float) sum / (len - 1);
}


/*
    Helper function to count the number of local maxima in a signal.
    This function mimics the _local_maxima_1d() function in the python
    "scipy" module.
*/
int16_t _find_peaks(float *x, int16_t len){

    int16_t i = 1; // points to the first considered sample (the second one)
    int16_t i_max = len-1;
    int16_t i_ahead = 0;

    int16_t npeaks = 0;

    while(i < i_max){

        if(x[i-1] < x[i]){
            i_ahead = i + 1;

            while(i_ahead < i_max && x[i_ahead] == x[i]){
                i_ahead++;
            }

            if(x[i_ahead] < x[i]){
                npeaks++;
                i = i_ahead;
            }
        }
        i++;
    }
    return npeaks;
}




void eepd(const float *sig, int16_t len, int16_t fs, const int8_t *select, int16_t *res){

    float *interm = (float*)malloc(len * sizeof(float));    // to store the intermediate result between the first and the second filter 
    float *filtered = (float*)malloc(len * sizeof(float));       // temporary to store the result of each filter

    const float *b, *a, *zi;

    for(int16_t i=0; i<N_EEPD; i++){
        if(select[i] == 1){

            b = filters_parameters.filters[i].b;
            a = filters_parameters.filters[i].a;
            zi = filters_parameters.filters[i].zi;  

            filtfilt(sig, len, b, a, zi, interm);
            vect_mult(interm, interm, len, interm);     // squared vector
            filtfilt(interm, len, b_second, a_second, zi_second, filtered);

            normalize_max(filtered, len, filtered);   // divide each number by the maximum

            res[i] = _find_peaks(filtered, len);
        }
    }

    free(interm);
    free(filtered);
}
