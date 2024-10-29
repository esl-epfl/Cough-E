#ifndef _AUDIO_FEATURES_H_
#define _AUDIO_FEATURES_H_

#include <inttypes.h>

/* Defines of the hyperparameters of audio features extraction */


#define WIND_LEN_AUD    0.8
#define OVERLAP_AUD     50  

// audio samples in a window
#define WINDOW_SAMP_AUDIO   (int16_t)(WIND_LEN_AUD * AUDIO_FS)

#define OVERLAP_SAMP        (int16_t)(WINDOW_SAMP_AUDIO * OVERLAP_AUD / 100.0)
#define AUDIO_STEP          (int16_t)(WINDOW_SAMP_AUDIO * (1.0 - (OVERLAP_AUD / 100.0)))
#define N_OVERLAPS          (int16_t)(WINDOW_SAMP_AUDIO / AUDIO_STEP - 1)

#define AUDIO_OVERLAP_SEC   (float)(WIND_LEN_AUD * (OVERLAP_AUD / 100.0))
#define AUDIO_STEP_SEC      (float)(WIND_LEN_AUD  - AUDIO_OVERLAP_SEC)


// Number of MFCCs to extract, this number of feature per each family
#define N_MFCC 64
// Number of MFCC feature families to extract
#define N_MFCC_FAMILIES 4

// number of PSD bands
#define N_PSD 3

// Start and end frequencies for the PSD bands
#define PSD_BAND_1_START 0
#define PSD_BAND_1_END 50

#define PSD_BAND_2_START 50
#define PSD_BAND_2_END 140

#define PSD_BAND_3_START 160
#define PSD_BAND_3_END 350

// #define PSD_BAND_4_START 1700
// #define PSD_BAND_4_END 2100

// #define PSD_BAND_5_START 2100
// #define PSD_BAND_5_END 2400

// #define PSD_BAND_6_START 2500
// #define PSD_BAND_6_END 3000

// #define PSD_BAND_7_START 3000
// #define PSD_BAND_7_END 3500

// #define PSD_BAND_8_START 3500
// #define PSD_BAND_8_END 4200

// #define PSD_BAND_9_START 4900
// #define PSD_BAND_9_END 5300

// struct to store the limits of each PSD band
typedef struct band {
    int16_t start;
    int16_t end;
} band_t;

// array to access all the bands in a compact way
static const band_t psd_bands[N_PSD] = {
                                        {.start=PSD_BAND_1_START, .end=PSD_BAND_1_END},
                                        {.start=PSD_BAND_2_START, .end=PSD_BAND_2_END},
                                        {.start=PSD_BAND_3_START, .end=PSD_BAND_3_END}
                                        };


// Start, end and bandwidth frequencies for the EEPD bandpass filter
#define EEPD_START 50
#define EEPD_END 1000
#define EEPD_BANDWIDTH 50
#define N_EEPD (EEPD_END - EEPD_START) / EEPD_BANDWIDTH


/* All the families of features used. The values of the enum are adjusted 
   considering the amount of features for every family */
enum audio_features_families{
    SPECTRAL_DECREASE,
    SPECTRAL_SLOPE,
    SPECTRAL_ROLLOFF,
    SPECTRAL_CENTROID,
    SPECTRAL_SPREAD,
    SPECTRAL_KURTOSIS,
    SPECTRAL_SKEW,
    SPECTRAL_FLATNESS,
    SPECTRAL_STD,
    SPECTRAL_ENTROPY,
    DOMINANT_FREQUENCY,
    POWER_SPECTRAL_DENSITY,
    MEL_FREQUENCY_CEPSTRAL_COEFFICIENT = POWER_SPECTRAL_DENSITY + N_PSD,
    ZERO_CROSSING_RATE = MEL_FREQUENCY_CEPSTRAL_COEFFICIENT + (N_MFCC * N_MFCC_FAMILIES),
    ROOT_MEANS_SQUARED,
    CREST_FACTOR,
    ENERGY_ENVELOPE_PEAK_DETECT,
    Number_AUDIO_Features = ENERGY_ENVELOPE_PEAK_DETECT + N_EEPD// Hardcoded just to get the final length
};

#endif