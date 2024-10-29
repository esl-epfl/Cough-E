#ifndef _FILTERING_H_
#define _FILTERING_H_

void linear_filer(float *sig, int len, const float *b, const float *a, float *zi, float *res);
void filtfilt(const float *sig, int len, const float *b, const float* a, const float *zi, float *res);

#endif