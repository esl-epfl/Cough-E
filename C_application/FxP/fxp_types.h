#pragma once

#include <stdint.h>

// =============================================================================
// fxp_types.h — Q-format type aliases and signal-entry converters
//
// Q-format notation:
//   Qm.f  : signed,   m integer bits (1 sign + m-1 magnitude), f fractional bits
//   UQm.f : unsigned, m integer bits, f fractional bits
//   Total register width = m + f bits  (always 16, 32, or 64)
// =============================================================================

// =============================================================================
// Section 1 — 16-bit types
// =============================================================================

typedef int16_t  q11_5_t;    // Q11.5   signed   16-bit
typedef uint16_t uq16_0_t;   // UQ16.0  unsigned 16-bit  (plain integer)
typedef uint16_t uq13_3_t;   // UQ13.3  unsigned 16-bit
typedef uint16_t uq7_9_t;    // UQ7.9   unsigned 16-bit
typedef uint16_t uq5_11_t;   // UQ5.11  unsigned 16-bit
typedef uint16_t uq2_14_t;   // UQ2.14  unsigned 16-bit
typedef uint16_t uq10_6_t;   // UQ10.6  unsigned 16-bit

// =============================================================================
// Section 2 — 32-bit types
// =============================================================================

typedef int32_t  q4_28_t;    // Q4.28   signed   32-bit
typedef uint32_t uq10_22_t;  // UQ10.22 unsigned 32-bit
typedef uint32_t uq9_23_t;   // UQ9.23  unsigned 32-bit
typedef uint32_t uq16_16_t;  // UQ16.16 unsigned 32-bit
typedef uint32_t uq2_30_t;   // UQ2.30  unsigned 32-bit

// =============================================================================
// Section 3 — 64-bit types
// =============================================================================

typedef int64_t  q34_30_t;   // Q34.30  signed   64-bit
typedef uint64_t uq20_44_t;  // UQ20.44 unsigned 64-bit

// =============================================================================
// Section 4 — Signal-entry converters
//
// Convert a float sample at the pipeline boundary into the correct Q-format.
// These are macros (not inline functions) because the fractional-bit count f
// must be an integer constant for the shift to be legal in C.
//
// =============================================================================

// Signed: rounds to nearest for both positive and negative floats
#define FXP_FROM_FLOAT(x, f)                                  \
    ((int32_t)(((float)(x) >= 0.0f)                           \
                   ? ((float)(x) * (float)(1 << (f)) + 0.5f)  \
                   : ((float)(x) * (float)(1 << (f)) - 0.5f)))

// Unsigned: rounds to nearest (non-negative values only)
#define FXP_FROM_FLOAT_U(x, f) \
    ((uint32_t)((float)(x) * (float)(1 << (f)) + 0.5f))

// Fixed-point back to float (for validation / printf)
#define FXP_TO_FLOAT(x, f) ((float)(x) / (float)(1 << (f)))

// IMU pipeline entry-point converters
#define FXP_IMU_RAW_FROM_FLOAT(x) ((q11_5_t) FXP_FROM_FLOAT  ((x),  5))
#define FXP_IMU_L2A_FROM_FLOAT(x) ((uq10_6_t)FXP_FROM_FLOAT_U((x),  6))
#define FXP_IMU_L2G_FROM_FLOAT(x) ((uq5_11_t)FXP_FROM_FLOAT_U((x), 11))

// Audio pipeline entry-point converter (all three audio paths: RFFT, periodogram, STFT)
#define FXP_AUDIO_FROM_FLOAT(x)   ((int16_t)  FXP_FROM_FLOAT  ((x), 14))
