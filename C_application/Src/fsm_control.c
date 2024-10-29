
#include <fsm_control.h>

fsm_t fsm_state;


void init_state(){
    
#ifndef RUN_ONLY_AUD
    fsm_state.model = IMU_MODEL;    // This is for both ONLY_IMU or MIXED
#else
    fsm_state.model = AUDIO_MODEL;
#endif

fsm_state.model_cls_out = NON_COUGH_OUT;
fsm_state.time_from_last_out = 0.0;
fsm_state.time_start_wind = 0.0;
fsm_state.time_start_last_wind = 0.0;
fsm_state.n_winds_aud = 0;

}



void update(){

    fsm_state.time_start_last_wind = fsm_state.time_start_wind;

    // COUGH found
    if(fsm_state.model_cls_out ==  COUGH_OUT){
        
        // IMU was used 
        if(fsm_state.model == IMU_MODEL){
            fsm_state.time_from_last_out = fsm_state.time_start_wind + WIND_LEN_IMU - fsm_state.timestamp_last_out;

            #ifdef RUN_MIXED
            fsm_state.model = AUDIO_MODEL;      // Switch to AUDIO model
            #else
            #ifdef RUN_ONLY_IMU
            fsm_state.time_start_wind += IMU_STEP_SEC;
            #endif
            #endif
        }
        else{   // AUDIO was used

            fsm_state.n_winds_aud++;
            fsm_state.time_from_last_out = fsm_state.time_start_wind + WIND_LEN_AUD - fsm_state.timestamp_last_out;
            
            #ifdef RUN_MIXED
            // Max number of windows to be processed by audio reached, switch back to IMU
            if(fsm_state.n_winds_aud >= N_MAX_WIND_AUD){
                fsm_state.n_winds_aud = 0;
                fsm_state.model = IMU_MODEL;
                fsm_state.time_start_wind += WIND_LEN_AUD;  // Next window start from the end of the current one
            }
            else{
                fsm_state.model = AUDIO_MODEL;
                fsm_state.time_start_wind += AUDIO_STEP_SEC;
            }
            #else
            
            #ifdef RUN_ONLY_AUD
            fsm_state.time_start_wind += AUDIO_STEP_SEC;
            #endif
            #endif

        }
    }
    else{   // NON COUGH

        // IMU was used 
        if(fsm_state.model == IMU_MODEL){
            fsm_state.time_from_last_out = fsm_state.time_start_wind + WIND_LEN_IMU - fsm_state.timestamp_last_out;
            fsm_state.model = IMU_MODEL;
            fsm_state.time_start_wind += IMU_STEP_SEC;
        }
        else{   // AUDIO was used

            #ifdef RUN_MIXED
            fsm_state.model = IMU_MODEL;
            fsm_state.time_from_last_out = fsm_state.time_start_wind + WIND_LEN_AUD - fsm_state.timestamp_last_out;
            fsm_state.time_start_wind += WIND_LEN_AUD;  // Next window start from the end of the current one
            #else
            #ifdef RUN_ONLY_AUD
            fsm_state.time_from_last_out = fsm_state.time_start_wind + WIND_LEN_AUD - fsm_state.timestamp_last_out;
            fsm_state.time_start_wind += AUDIO_STEP_SEC;
            #endif
            #endif
        }
    }

}

// #include <stdio.h>
uint8_t check_postprocessing(){

    // printf(">> Checking post\n");
    // printf("time from last: %f\n", fsm_state.time_from_last_out);
    // printf("last out time: %f\n", fsm_state.timestamp_last_out);

    if(fsm_state.time_from_last_out >= TIME_DEADLINE_OUTPUT){
        
        if(fsm_state.model == IMU_MODEL){
            fsm_state.timestamp_last_out = fsm_state.time_start_last_wind + WIND_LEN_IMU;
        } else {
            fsm_state.timestamp_last_out = fsm_state.time_start_last_wind + WIND_LEN_AUD;
        }

        fsm_state.time_from_last_out = 0.0;

        // printf("\n");
        // printf("time start: %f\n", fsm_state.time_start_last_wind);
        // printf("time from last: %f\n", fsm_state.time_from_last_out);
        // printf("last out time: %f\n", fsm_state.timestamp_last_out);


        return 1;
    }

    return 0;
}


uint32_t get_idx_window(){

    if(fsm_state.model == IMU_MODEL){
        return (uint32_t)(fsm_state.time_start_wind * IMU_FS);
    }
    else{
        return (uint32_t)(fsm_state.time_start_wind * AUDIO_FS);
    }
}




// #include <stdio.h>
// int main(){

//     init_state();


//     printf("Model :%d\n", fsm_state.model);
//     printf("Time: %f\n", fsm_state.time_start_wind);
//     printf("Idx wind: %d\n", get_idx_window());
//     printf("\n");

//     fsm_state.time_start_wind += IMU_STEP_SEC;

//     printf("Model :%d\n", fsm_state.model);
//     printf("Time: %f\n", fsm_state.time_start_wind);
//     printf("Idx wind: %d\n", get_idx_window());
//     printf("\n");

//     fsm_state.model = AUDIO_MODEL;
    

//     printf("Model :%d\n", fsm_state.model);
//     printf("Time: %f\n", fsm_state.time_start_wind);
//     printf("Idx wind: %d\n", get_idx_window());
//     printf("\n");


// }