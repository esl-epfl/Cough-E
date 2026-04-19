#pragma once

#include <stdint.h>

#include <core/fxp_qformats.h>

// 16-bit Q-format aliases.
typedef int16_t  q11_5_t;
typedef int16_t  q5_11_t;
typedef uint16_t uq13_3_t;
typedef uint16_t uq7_9_t;
typedef uint16_t uq5_11_t;
typedef uint16_t uq2_14_t;
typedef uint16_t uq10_6_t;
typedef uint16_t uq11_5_t;

// 32-bit Q-format aliases.
typedef int32_t  q12_20_t;
typedef int32_t  q13_19_t;
typedef uint32_t uq10_22_t;
typedef uint32_t uq9_23_t;
typedef uint32_t uq16_16_t;
typedef uint32_t uq12_20_t;
typedef uint32_t uq15_17_t;
typedef uint32_t uq10_21_t;
typedef uint32_t uq0_20_t;
typedef uint32_t uq9_22_t;
typedef uint32_t uq18_12_t;
typedef uint32_t uq7_15_t;

// 64-bit Q-format aliases.
typedef int64_t  q34_30_t;
typedef uint64_t uq20_44_t;
typedef uint64_t uq30_32_t;

// Width checks keep type aliases in sync with the manifest-generated constants.
_Static_assert(sizeof(q11_5_t) * 8 == FXP_BITS_IMU_RAW, "q11_5_t width mismatch");
_Static_assert(sizeof(uq10_6_t) * 8 == FXP_BITS_IMU_L2A, "uq10_6_t width mismatch");
_Static_assert(sizeof(uq5_11_t) * 8 == FXP_BITS_IMU_L2G, "uq5_11_t width mismatch");
_Static_assert(sizeof(uq16_16_t) * 8 == FXP_BITS_IMU_RMS_RAW, "uq16_16_t width mismatch");
_Static_assert(sizeof(uq13_3_t) * 8 == FXP_BITS_IMU_RMS_L2A, "uq13_3_t width mismatch");
_Static_assert(sizeof(uq7_9_t) * 8 == FXP_BITS_IMU_RMS_L2G, "uq7_9_t width mismatch");
_Static_assert(sizeof(uq9_23_t) * 8 == FXP_BITS_IMU_LINE_LENGTH_RAW, "uq9_23_t width mismatch");
_Static_assert(sizeof(q34_30_t) * 8 == FXP_BITS_IMU_KURTOSIS_RAW, "q34_30_t width mismatch");
_Static_assert(sizeof(uq2_14_t) * 8 == FXP_BITS_IMU_CREST_L2G, "uq2_14_t width mismatch");
_Static_assert(sizeof(uq10_22_t) * 8 == FXP_BITS_IMU_KURT_VARIANCE, "uq10_22_t width mismatch");
_Static_assert(sizeof(uq20_44_t) * 8 == FXP_BITS_IMU_KURT_STD4, "uq20_44_t width mismatch");
_Static_assert(sizeof(q12_20_t) * 8 == FXP_BITS_AUDIO_FFT_RE_IM, "q12_20_t width mismatch");
_Static_assert(sizeof(uq12_20_t) * 8 == FXP_BITS_AUDIO_FFT_MAGNITUDES, "uq12_20_t width mismatch");
_Static_assert(sizeof(uq15_17_t) * 8 == FXP_BITS_AUDIO_FFT_SUM_MAGS, "uq15_17_t width mismatch");
_Static_assert(sizeof(uq10_21_t) * 8 == FXP_BITS_AUDIO_FFT_CENTROID, "uq10_21_t width mismatch");
_Static_assert(sizeof(q13_19_t) * 8 == FXP_BITS_AUDIO_FFT_DEV, "q13_19_t width mismatch");
_Static_assert(sizeof(uq11_5_t) * 8 == FXP_BITS_AUDIO_FFT_SPREAD, "uq11_5_t width mismatch");
_Static_assert(sizeof(uq0_20_t) * 8 == FXP_BITS_AUDIO_FFT_INV_SPREAD, "uq0_20_t width mismatch");
_Static_assert(sizeof(q5_11_t) * 8 == FXP_BITS_AUDIO_FFT_NORM_DEV, "q5_11_t width mismatch");
_Static_assert(sizeof(uq9_22_t) * 8 == FXP_BITS_AUDIO_FFT_NORM_DEV2, "uq9_22_t width mismatch");
_Static_assert(sizeof(uq18_12_t) * 8 == FXP_BITS_AUDIO_FFT_NORM_DEV4, "uq18_12_t width mismatch");
_Static_assert(sizeof(uq30_32_t) * 8 == FXP_BITS_AUDIO_FFT_KURT_WEIGHT, "uq30_32_t width mismatch");
_Static_assert(sizeof(uq7_15_t) * 8 == FXP_BITS_AUDIO_FFT_KURTOSIS, "uq7_15_t width mismatch");
