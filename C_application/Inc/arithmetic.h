#ifndef _ARITHMETIC_H
#define _ARITHMETIC_H

#include <math.h>
#include <types.h>

template <typename T> static inline T floorreal(T x) {
    return floorf((float)x);
}
template <typename T> static inline T expreal(T x) {
    return expf((float)x);
}
template <typename T, typename S> static inline T powreal(T base, S exp) {
    return powf((float)base, (float)exp);
}
template <typename T> static inline T sqrtreal(T x) {
    return sqrtf((float)x);
}
template <typename T> static inline T logreal(T x) {
    return logf((float)x);
}
template <typename T> static inline T log10real(T x) {
    return log10f((float)x);
}

#if defined(UPOS_MODE) && defined(DOUBLE_MATH)
template <Universal T> static inline T floorreal(T x) {
    return sw::universal::floor(x);
}
template <Universal T> static inline T expreal(T x) {
    return sw::universal::exp(x);
}
template <Universal T, typename S> static inline T powreal(T base, S exp) {
    return sw::universal::pow(base, static_cast<S>(exp));
}
template <Universal T> static inline T sqrtreal(T x) {
    return sw::universal::sqrt(x);
}
template <Universal T> static inline T logreal(T x) {
    return sw::universal::log(x);
}
template <Universal T> static inline T log10real(T x) {
    return sw::universal::log10(x);
}
#endif

#ifdef LPOS_MODE
static inline Posit sqrtreal(Posit x) { return libposit::sqrt(x); }
#endif

#endif
