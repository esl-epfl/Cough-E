#ifndef _MAIN_H_
#define _MAIN_H_

#include <inttypes.h>

//////////////////////////////////////
/* Model to be used                 */
//////////////////////////////////////
#include <audio_model.h>            
#include <imu_model.h>              
//////////////////////////////////////


//////////////////////////////////////
/* Input data                       */
// ///////////////////////////////////
#include <input_data/audio_input_49393_w0_1wnds.h>
#include <input_data/imu_input_49393_w0_1wnds.h>
#include <input_data/bio_input_49393.h>
//////////////////////////////////////


#include <audio_features.h>
#include <imu_features.h>


/* Threshold for the audio model */
#define AUDIO_TH    0.3

/* Threshold for the imu model */
#define IMU_TH    0.05

// Defines (in seconds) how often to provide the final estimation (execute post-processing)
#define TIME_DEADLINE_OUTPUT    5.0

// Maximum number of consecutive windows to be run by AUDIO model
#define N_MAX_WIND_AUD  4

////////////////////////////////////////////////
/* Define if to run in multi or unimodal mode */
////////////////////////////////////////////////

// Execute in multimodal mode, using both modalities cooperating
#define RUN_MIXED

#ifndef RUN_MIXED
    #define RUN_ONLY_AUD        // Use only the audio modality

    #ifndef RUN_ONLY_AUD
        #define RUN_ONLY_IMU    // Use only the imu modality
    #endif
#endif
////////////////////////////////////////////////


#endif