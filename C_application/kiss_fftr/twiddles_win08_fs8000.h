#ifndef _TWIDDLES_H_
#define _TWIDDLES_H_

#include <types.h>

template <typename T> struct twiddles_t {
    T cosine;
    T sine;
};

template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_3200[3200];
template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_450[450];
template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_1024[1024];
template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_1600[1600];
template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_225[225];
template <RealType T> extern const twiddles_t<T> FLASH1 twiddles_512[512];

#endif
