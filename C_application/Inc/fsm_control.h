#ifndef _FSM_CONTROL_H_
#define _FSM_CONTROL_H_

#include <main.h>
#include <inttypes.h>

/**
 * Enum to discriminate between one model and the other
 * 
 * IMU --> the current running model is IMU
 * AUDIO --> the current running model is AUDIO
*/
typedef enum Model {
    IMU_MODEL,
    AUDIO_MODEL
} model_t;


/**
 * Enum to discriminate between one output and the other
 * 
 * COUGH --> the current window is classified as cough
 * NON-COUGH --> the current window is classified as non-cough
*/
typedef enum Class {
    NON_COUGH_OUT,
    COUGH_OUT
} class_t;


/**
 * Struct that contains the state of the system.
 * This is used to implement the cooperation of the two models and allow 
 * the system to properly switch between them.
*/
typedef struct fsm
{
    model_t model;              // Model to process the current window
    class_t model_cls_out;      // Output of the model for the current window
    uint32_t last_output_tick;       // Audio-sample tick of when the last output was provided
    uint32_t ticks_from_last_output; // Audio-sample ticks elapsed from last output
    uint32_t window_start_tick;      // Audio-sample tick of the current window start
    uint32_t last_window_start_tick; // Audio-sample tick of the last processed window start
    uint8_t n_winds_aud;        // Number of consecutive windows processed with AUDIO model
} fsm_t;


/**
 * Extern declaration of the state variable, so to be modified from other files.
 * This variable is meant to be updated during the execution of the current window.
 * At the end of it, when also the classification outcome is available, it has to be 
 * updated by calling the update function
*/
extern fsm_t fsm_state;


/**
 * Initilize the state variable with the following default values
 * 
 * model                --> IMU / AUDIO depedning on the selected confguration (MIX / ONLY_AUDIO / ONLY_IMU)
 * model_cls_out        --> NON_COUGH
 * ticks_from_last_output --> 0
 * window_start_tick      --> 0
 * last_window_start_tick --> 0
 * n_winds_aud          --> 0
*/
void init_state();


/**
 * This function updates the state of the system according to the struct variables.
 * It is supposed to be called after executing a window and before checking if the postprocessing has to be executed
*/
void update();

/**
 * Checks if the postprocessing has to be executed
 * 
 * @return If the postprocessing routing has to be executed (1) or not (0)
*/
uint8_t check_postprocessing();

/**
 * Returns the index pointing to the first sample of the next window to process.
 * Depending on the current model to use, the index will be different.
*/
uint32_t get_idx_window();

#endif
