#pragma once

#include <stdint.h>

#include <core/fxp_qformats.h>

// 16-bit Q-format aliases.
typedef int16_t  q11_5_t;
typedef uint16_t uq13_3_t;
typedef uint16_t uq7_9_t;
typedef uint16_t uq5_11_t;
typedef uint16_t uq2_14_t;
typedef uint16_t uq10_6_t;

// 32-bit Q-format aliases.
typedef uint32_t uq10_22_t;
typedef uint32_t uq9_23_t;
typedef uint32_t uq16_16_t;

// 64-bit Q-format aliases.
typedef int64_t  q34_30_t;
typedef uint64_t uq20_44_t;

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
