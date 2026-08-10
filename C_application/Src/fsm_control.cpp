
#include <fsm_control.h>

fsm_t fsm_state;


void init_state(){
    
#ifndef RUN_ONLY_AUD
    fsm_state.model = IMU_MODEL;    // This is for both ONLY_IMU or MIXED
#else
    fsm_state.model = AUDIO_MODEL;
#endif

fsm_state.model_cls_out = NON_COUGH_OUT;
fsm_state.last_output_tick = 0U;
fsm_state.ticks_from_last_output = 0U;
fsm_state.window_start_tick = 0U;
fsm_state.last_window_start_tick = 0U;
fsm_state.n_winds_aud = 0;

}



void update(){

    fsm_state.last_window_start_tick = fsm_state.window_start_tick;

    // COUGH found
    if(fsm_state.model_cls_out ==  COUGH_OUT){
        
        // IMU was used 
        if(fsm_state.model == IMU_MODEL){
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + IMU_WINDOW_TICKS - fsm_state.last_output_tick;

            #ifdef RUN_MIXED
            fsm_state.model = AUDIO_MODEL;      // Switch to AUDIO model
            #else
            #ifdef RUN_ONLY_IMU
            fsm_state.window_start_tick += IMU_STEP_TICKS;
            #endif
            #endif
        }
        else{   // AUDIO was used

            fsm_state.n_winds_aud++;
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + AUDIO_WINDOW_TICKS - fsm_state.last_output_tick;

            #ifdef RUN_MIXED
            // Max number of windows to be processed by audio reached, switch back to IMU
            if(fsm_state.n_winds_aud >= N_MAX_WIND_AUD){
                fsm_state.n_winds_aud = 0;
                fsm_state.model = IMU_MODEL;
                fsm_state.window_start_tick += AUDIO_WINDOW_TICKS;  // Next window start from the end of the current one
            }
            else{
                fsm_state.model = AUDIO_MODEL;
                fsm_state.window_start_tick += AUDIO_STEP_TICKS;
            }
            #else
            
            #ifdef RUN_ONLY_AUD
            fsm_state.window_start_tick += AUDIO_STEP_TICKS;
            #endif
            #endif

        }
    }
    else{   // NON COUGH

        // IMU was used 
        if(fsm_state.model == IMU_MODEL){
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + IMU_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.model = IMU_MODEL;
            fsm_state.window_start_tick += IMU_STEP_TICKS;
        }
        else{   // AUDIO was used

            #ifdef RUN_MIXED
            fsm_state.model = IMU_MODEL;
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + AUDIO_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.window_start_tick += AUDIO_WINDOW_TICKS;  // Next window start from the end of the current one
            #else
            #ifdef RUN_ONLY_AUD
            fsm_state.ticks_from_last_output = fsm_state.window_start_tick + AUDIO_WINDOW_TICKS - fsm_state.last_output_tick;
            fsm_state.window_start_tick += AUDIO_STEP_TICKS;
            #endif
            #endif
        }
    }

}

// #include <stdio.h>
uint8_t check_postprocessing(){

    // printf(">> Checking post\n");
    // printf("ticks from last: %u\n", fsm_state.ticks_from_last_output);
    // printf("last out tick: %u\n", fsm_state.last_output_tick);

    if(fsm_state.ticks_from_last_output >= TIME_DEADLINE_OUTPUT_TICKS){
        
        if(fsm_state.model == IMU_MODEL){
            fsm_state.last_output_tick = fsm_state.last_window_start_tick + IMU_WINDOW_TICKS;
        } else {
            fsm_state.last_output_tick = fsm_state.last_window_start_tick + AUDIO_WINDOW_TICKS;
        }

        fsm_state.ticks_from_last_output = 0U;

        // printf("\n");
        // printf("start tick: %u\n", fsm_state.last_window_start_tick);
        // printf("ticks from last: %u\n", fsm_state.ticks_from_last_output);
        // printf("last out tick: %u\n", fsm_state.last_output_tick);


        return 1;
    }

    return 0;
}


uint32_t get_idx_window(){

    if(fsm_state.model == IMU_MODEL){
        return (uint32_t)(((uint64_t)fsm_state.window_start_tick * IMU_FS) / AUDIO_FS);
    }
    else{
        return fsm_state.window_start_tick;
    }
}
