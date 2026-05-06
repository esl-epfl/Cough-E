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
    uq20_44_t *frame_power;
    uq20_44_t *mel_power;
    int32_t *mel_db_q9; // Q?.9 widened to int32 to keep pre-clip headroom.
    q7_9_t mean_q9[N_MFCC];
    q7_9_t std_q9[N_MFCC];
    q7_9_t max_q9[N_MFCC];
    uint16_t entropy_q14[N_MFCC];
    int32_t stft_db_offset_q9;
    int32_t mel_db_offset_q9;
} audio_mel_stage_probe_t;

int audio_fft_stage_probe(const int16_t *sig,
                          int16_t len,
                          int16_t fs,
                          uq12_20_t *mags,
                          uq12_20_t *freqs,
                          uq15_17_t *sum_mags);

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
