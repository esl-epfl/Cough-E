#include <stdio.h>
#include <stdlib.h>
#include <helpers.h>
#include <azc.h>
#include <math.h>



/**
 * Structure used for the polygonal approximation function.
 * It stores a pair of (first, last) indexes.
*/
typedef struct seg_idxs
{
    int16_t first;
    int16_t last;
} seg_idxs_t;


/**
 * Implements the comparison function to be used by the qsort algorithm
*/
int _qsort_cmp(const void *e1, const void *e2){
    if( *(int16_t*)e1 > *(int16_t*)e2 ){
        return 1;
    }
    if( *(int16_t*)e1 < *(int16_t*)e2 ){
        return -1;
    }
    return 0;
}

/**
 * Computes the interpolated linear segment of length len from (xf, yf) to (xl, yl).
 * Given the first and last points' coordinates, it computes all the intermediate
 * points of the linear interpolation fitting a linear segment. The number of points
 * will be equal to len
 * 
 * @param len: length in points of the final segment
 * @param xf: x coordinate of the first point
 * @param yf: y coordinate of the first point
 * @param xl: x coordinate of the last point
 * @param yf: y coordinate of the last point
 * @param *res: pointer to the array used to store the result. Notice that the result
 * will be made only of y coordinates
*/
void _interp(int16_t len, int16_t xf, float yf, int16_t xl, float yl, float *res){

    res[0] = yf;
    res[len-1] = yl;

    // Compute the segment parameters
    float m = (yl - yf) / (xl - xf);    // slope
    float q = yf - (m * xf);            // y-axis intercept

    for(int i=1; i<len-1; i++){
        res[i] = (m * (xf + i)) + q;
    }
}

/**
 * Computes the discrete-time differentiation of the given signal given integer timestamps
 * 
 * @param *sig: pointer to array storing the signal
 * @param timestamps: time instants of each signal sample, needs to be passed since
 * the timestamps might be non equally spaced
 * @param len: length of the input signal array
 * 
 * @return a float pointer containing the resulting differentiation
*/
float *_discrete_diff(float *sig, int16_t *timestamps, int16_t len){
        
    int16_t res_len = len-1;
    float *res = (float*)malloc((res_len) * sizeof(float));

    for(int16_t i=0; i<(res_len); i++){
        res[i] = (sig[i+1] - sig[i]) / (timestamps[i+1] - timestamps[i]);
    }
    return res;
}

/**
 * Computes the max vertical distance from the specified signal and the
 * linear segment delimited by the signal points indexes by first and last.
 * 
 * @param *sig: pointer to the signal
 * @param  first: index of the first signal sample of the linear segment
 * @param  last: index of the last signal sample of the linear segment
 * @param  *idx: pointer in which to store the index of the sample having max vertical distance
 *  
 * @return The max vertical distance found. Also the index of the signal sample having this max
 * distance from the linear segment will be stored in the idx parameter
*/
float _max_vdist(float *sig, int16_t first, int16_t last, int16_t *idx){
    
    // Check if the first and last indexes are the same
    if(first == last){
        *idx = first;
        return 0.0;
    }

    // Length - number of point for which to check the distance
    int16_t len = last - first + 1;

    // Interpolated segment
    float *intrp = (float*)malloc(len * sizeof(float));
    _interp(len, first, sig[first], last, sig[last], intrp);
    
    // To store the distances 
    float *dist = (float*)malloc(len * sizeof(float));
    for(int16_t i=0; i<len; i++){
        dist[i] = fabs(sig[first+i] - intrp[i]);
    }

    // Get the maximum distance
    *idx = vect_max_index(dist, len);
    
    // Have to take the result before adjusting the index
    float result  = dist[*idx];

    // Adjust the index to have the global one with respect to sig
    *idx += first;

    free(intrp);
    free(dist);

    return result;
}


/**
 * Computes the polygonal approximation of the given signal and returns the 
 * indexes of the result.
 * 
 * @param sig: pointer to the signal
 * @param len: length of the original signal
 * @param eps: epsilon value used as a tolerance for the Douglas-Peucker algorithm
 * @param res_len: pointer in which to store the lenght of the resulting approximated signal
 * 
 * @return pointer to the array storing the indexes of the point in the original signal
 * that form the result. 
*/
int16_t *polygonal_approx(float *sig, int16_t len, float eps, int16_t *res_len){

    // Array of the resulting indexes. For safety reasons it is allocated
    // at his maximum possible size (i.e. len)
    int16_t *res = (int16_t*)malloc(len * sizeof(int16_t));

    // Counts how many indexes has been found
    int16_t idxs_found = 0;

    // To check if it is possible to add fisrt and last idx (i.e. if they are not already in)
    int16_t add_first = 0;
    int16_t add_last = 0;

    // Array of structures to save the indexes pairs in the form (fist, last) to be processed
    // Each new pair will be added in the end.
    seg_idxs_t *stack = (seg_idxs_t*)malloc(len * sizeof(seg_idxs_t));

    // Initialize the first pair with the first and last indexes
    stack[0].first = 0;
    stack[0].last = len-1;

    // Always keeps track of the tail of the array, next element to process
    int16_t next_to_process = 0;

    float max_dist = 0.0;
    int16_t max_idx = 0;

    // Indexes of first and last element to consider at each iteration
    int16_t first, last = 0;

    // Loops until there are no more pairs to check (i.e. the index is >= 0)
    while(next_to_process >= 0){
        first = stack[next_to_process].first;
        last = stack[next_to_process].last;

        next_to_process -= 1;

        max_dist = _max_vdist(sig, first, last, &max_idx);

        // Only keep the sample if it's max distance is greater then the tolerance eps
        if(max_dist > eps){
            stack[next_to_process+1].first = first;
            stack[next_to_process+1].last = max_idx;

            stack[next_to_process+2].first = max_idx;
            stack[next_to_process+2].last = last;

            next_to_process += 2;
        } else {
            
            // To avoid adding duplicated indexes in the result
            add_first = 1;
            add_last = 1;
            for(int16_t i=0; i<idxs_found; i++){
                if(first == res[i]){
                    add_first = 0;
                }
                if(last != res[i]){
                    add_last = 0;
                }
            }

            if(add_first){
                res[idxs_found] = first;
                idxs_found++;
            }
            if(add_last){
                res[idxs_found] = last;
                idxs_found++;
            }
        }
    }

    free(stack);

    *res_len = idxs_found;
    return res;
}



int16_t azc_computation(float *sig, int16_t len, float epsilon){

    int16_t approx_len = 0;

    // Compute the approximation
    int16_t *approx_idxs = polygonal_approx(sig, len, epsilon, &approx_len);
    
    // Sort the resulting indexes in ascending way
    qsort(approx_idxs, approx_len, sizeof(int16_t), _qsort_cmp);

    // Extract the approximated signal
    float *approx_sig = (float*)malloc(approx_len * sizeof(float));
    int16_t *timestamps = (int16_t*)malloc(approx_len * sizeof(int16_t));
    for(int16_t i=0; i<approx_len; i++){
        approx_sig[i] = sig[approx_idxs[i]];
        timestamps[i] = approx_idxs[i];
    }
    
    float *diff = _discrete_diff(approx_sig, timestamps, approx_len);

    int16_t azc = 0;

    // Count the times the differentiation crosses the 0-axis
    if(approx_len - 1 > 1){
        for(int16_t i=0; i<(approx_len-2); i++){
            if(diff[i] * diff[i+1] < 0){
                azc++;
            }
        }
    }

    free(approx_idxs);
    free(approx_sig);
    free(timestamps);
    free(diff);

    return azc;
}