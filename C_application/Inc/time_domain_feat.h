#ifndef _TIME_DOMAIN_FEAT_H
#define _TIME_DOMAIN_FEAT_H

#include <inttypes.h>

float get_max(float *sig, int16_t len);
void sub_mean(const float *sig, float *res, int16_t len);
float get_rms(float *sig, int16_t len);
float compute_zrc(float *sig, int16_t len);
void eepd(const float *sig, int16_t len, int16_t fs, const int8_t *select, int16_t *res);

#endif