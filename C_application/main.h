#ifndef _MAIN_H_
#define _MAIN_H_

#include <inttypes.h>
#include <core/cough_backend.h>

//////////////////////////////////////
/* Model to be used                 */
//////////////////////////////////////
#include <audio_model.h>            
#include <imu_model.h>              
#ifdef FXP_MODE
#include <model_fxp.h>
#endif
//////////////////////////////////////


//////////////////////////////////////
/* Input data                       */
// ///////////////////////////////////
#include <input_data/20724/audio_input_20724_t1_sit_traffic_deep_breathing.h>
#include <input_data/20724/imu_input_20724_t1_sit_traffic_deep_breathing.h>
#include <input_data/20724/bio_input_20724.h>
//////////////////////////////////////


#include <audio_features.h>
#include <imu_features.h>


/* Threshold for the audio model */
#ifndef FXP_MODE
#define AUDIO_TH    0.3

/* Threshold for the imu model */
#define IMU_TH    0.05
#endif

// Defines how often to provide the final estimation, in audio-sample ticks.
#define TIME_DEADLINE_OUTPUT_NUM    3U
#define TIME_DEADLINE_OUTPUT_DEN    2U
#define TIME_DEADLINE_OUTPUT_TICKS  ((uint32_t)(((uint64_t)TIME_DEADLINE_OUTPUT_NUM * AUDIO_FS) / TIME_DEADLINE_OUTPUT_DEN))

// Maximum number of consecutive windows to be run by AUDIO model
#define N_MAX_WIND_AUD  4

////////////////////////////////////////////////
/* Define if to run in multi or unimodal mode */
////////////////////////////////////////////////

// Execute in multimodal mode, using both modalities cooperating
#define RUN_MIXED

#ifndef RUN_MIXED
    // #define RUN_ONLY_AUD        // Use only the audio modality

    #ifndef RUN_ONLY_AUD
        #define RUN_ONLY_IMU    // Use only the imu modality
    #endif
#endif
////////////////////////////////////////////////


#endif
