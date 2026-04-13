#pragma once

#include <limits.h>
#include <stdint.h>

static inline int16_t fxp_sat_s16_from_s32(int32_t x)
{
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

static inline uint16_t fxp_sat_u16_from_u32(uint32_t x)
{
    if (x > UINT16_MAX) return UINT16_MAX;
    return (uint16_t)x;
}

static inline uint16_t fxp_sat_u16_from_s32(int32_t x)
{
    if (x < 0) return 0;
    if ((uint32_t)x > UINT16_MAX) return UINT16_MAX;
    return (uint16_t)x;
}

static inline int32_t fxp_sat_s32_from_s64(int64_t x)
{
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static inline uint32_t fxp_sat_u32_from_u64(uint64_t x)
{
    if (x > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)x;
}
