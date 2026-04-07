#include <stdio.h>
#include <math.h>

#include <feature_extraction.h>
#include <time_domain_feat.h>
#include <helpers.h>
#include <filters_parameters.h>
#include <filtering.h>

#include <audio_features.h>
#include <range_analysis.h>
#include <imu_features.h>

#ifdef FXP_MODE
#include <fxp.h>
#endif


// Here I put all the functions to compute time domain features

int16_t _find_peaks(float *x, int16_t len);




float get_max(float *sig, int16_t len){

    float max = sig[0];

    for(int16_t i=1; i<len; i++){
        if(sig[i] > max){
            max = sig[i];
        }
    }
    RA_IMU_LOG_SCALAR("get_max", "get_max", max);
    return max;
}



void sub_mean(const float *sig, float *res, int16_t len){

    float mean = vect_mean(sig, len);

    sub_constant(sig, len, mean, res);
}


float get_rms(float *sig, int16_t len){
    float sum = 0;
    for(int16_t i=0; i<len; i++){
        float sq = sig[i] * sig[i];
        RA_IMU_LOG_SCALAR("get_rms", "sig_sq", sq);
        sum += sq;
    }
    RA_IMU_LOG_SCALAR("get_rms", "sum_sq", sum);
    float result = sqrtf(sum / len);
    RA_IMU_LOG_SCALAR("get_rms", "result", result);
    return result;
}




float compute_zrc(float *sig, int16_t len){

    int sum = 0;
    float interm_product = 0;

    for(int16_t i=0; i<len-1; i++){
        interm_product = sig[i] * sig[i + 1];
        RA_IMU_LOG_SCALAR("compute_zrc", "multiplier", interm_product);
        if(interm_product < 0){
            sum++;
        }
    }
    float result = (float) sum / (len - 1);
    RA_IMU_LOG_SCALAR("compute_zrc", "result", result);
    return result;
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
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "bandpass_out", interm, len);

            vect_mult(interm, interm, len, interm);     // squared vector
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "squared", interm, len);

            filtfilt(interm, len, b_second, a_second, zi_second, filtered);
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "envelope", filtered, len);

            normalize_max(filtered, len, filtered);   // divide each number by the maximum
            RA_LOG_ARRAY("AUDIO_EEPD", "eepd", "normalized", filtered, len);

            res[i] = _find_peaks(filtered, len);
            RA_LOG_SCALAR("AUDIO_EEPD", "eepd", "n_peaks", (float)res[i]);
        }
    }

    free(interm);
    free(filtered);
}


// =============================================================================
// Fixed-point kernel instantiations (time_domain_feat.c owns: get_rms, get_max)
// =============================================================================
#ifdef FXP_MODE

// RMS: all three signal types
FXP_DEFINE_GET_RMS(raw, q11_5_t,  uq11_16_t, uint64_t,
                   fxp_rms_raw_sq, fxp_rms_raw_sq_to_accum, fxp_rms_raw_result)
FXP_DEFINE_GET_RMS(l2a, uq10_6_t, uq13_3_t,  uint32_t,
                   fxp_rms_l2a_sq, fxp_rms_l2a_sq_to_accum, fxp_rms_l2a_result)
FXP_DEFINE_GET_RMS(l2g, uq5_11_t, uq7_9_t,   uint32_t,
                   fxp_rms_l2g_sq, fxp_rms_l2g_sq_to_accum, fxp_rms_l2g_result)

// get_max: L2_G only (needed for crest factor)
FXP_DEFINE_GET_MAX_L2G()

#endif // FXP_MODE
