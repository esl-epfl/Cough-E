#pragma once

#include <inttypes.h>

#include <audio_features.h>
#include <core/fxp_core.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

void audio_fft_features(const int8_t *features_selector,
                        const int16_t *sig,
                        int16_t len,
                        int16_t fs,
                        fxp_feat_t *feats);

#if defined(FXP_STAGE_PROBES)

typedef struct {
    int16_t n_frames;
    int16_t n_mels;
    uint8_t idxs_needed[N_MFCC];
    uq8_56_t *frame_power;
    uq8_56_t *mel_power;
    q23_9_t *mel_db;
    uint8_t frame_power_frac_bits;
    uint8_t mel_power_frac_bits;
    q7_9_t mean[N_MFCC];
    q7_9_t std[N_MFCC];
    q7_9_t max[N_MFCC];
    uq2_14_t entropy[N_MFCC];
    q23_9_t stft_db_offset;
    q23_9_t mel_db_offset;
} audio_mel_stage_probe_t;

int audio_fft_stage_probe(const int16_t *sig,
                          int16_t len,
                          int16_t fs,
                          uq4_28_t *mags,
                          uq12_20_t *freqs,
                          uq7_25_t *sum_mags);

int audio_psd_stage_probe(const int16_t *sig,
                          int16_t sig_len,
                          int16_t fs,
                          uq9_23_t *acc_power,
                          uq12_20_t *freqs,
                          int16_t *steps_out);

#endif

void audio_psd_features(const int8_t *features_selector,
                        const int16_t *sig,
                        int16_t sig_len,
                        int16_t fs,
                        fxp_feat_t *feats);

void audio_mel_features(const int8_t *features_selector,
                        const int16_t *sig,
                        int16_t len,
                        fxp_feat_t *feats);

void audio_crest_factor(const int8_t *features_selector,
                        const int16_t *sig,
                        int16_t len,
                        fxp_feat_t *feats);

#if defined(FXP_STAGE_PROBES)

int audio_mel_stage_probe(const int8_t *features_selector,
                          const int16_t *sig,
                          int16_t len,
                          audio_mel_stage_probe_t *probe);

void audio_mel_stage_probe_free(audio_mel_stage_probe_t *probe);

#endif

#endif
