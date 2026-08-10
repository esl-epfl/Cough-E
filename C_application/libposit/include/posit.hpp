// posit.hpp

#ifndef POSIT_HPP
#define POSIT_HPP

#include "posit.h"

#if POSIT_SIZE == 16
#include "softposit/posit16.h"
#else
#if POSIT_SIZE == 32
#include "softposit/posit32.h"
#else
#error "Invalid POSIT_SIZE"
#endif
#endif

#ifdef LOG_PERF_FLOAT
extern volatile int count_pos2float_conv;
extern volatile int count_float2pos_conv;
#define INC_POS2FLOAT() if (!__builtin_is_constant_evaluated()) count_pos2float_conv += 1
#define INC_FLOAT2POS() if (!__builtin_is_constant_evaluated()) count_float2pos_conv += 1
#else
#define INC_POS2FLOAT() ((void)0)
#define INC_FLOAT2POS() ((void)0)
#endif

static constexpr inline posit_t float2posit(float x) {
#if POSIT_SIZE == 16
    return convertFloatToP16(x);
#else
#if POSIT_SIZE == 32
    return convertFloatToP32(x);
#endif
#endif
}

static constexpr inline float posit2float(posit_t x) {
#if POSIT_SIZE == 16
    return convertP16ToFloat(x);
#else
#if POSIT_SIZE == 32
    return convertP32ToFloat(x);
#endif
#endif
}

class Posit {
    posit_t v;

public:
    constexpr Posit() : v(0) { }

    constexpr Posit(float x) : v(float2posit(x)) { INC_FLOAT2POS(); }
    template<typename T>
    constexpr Posit(T x) : v(float2posit(static_cast<float>(x))) { INC_FLOAT2POS(); }

    explicit operator float() const { INC_POS2FLOAT(); return posit2float(v); }
    template<typename T>
    explicit operator T() const { INC_POS2FLOAT(); return static_cast<T>(posit2float(v)); }

    inline posit_t bits() {
        return v;
    }

    // call `from_bits` to directly set posit numbers
    static constexpr Posit from_bits(posit_t x){
        Posit p;
        p.v = x;
        return p;
    }

    static inline Posit min(Posit x, Posit y){
        Posit p;
        p.v = pmin(x.v, y.v);
        return p;
    }

    static inline Posit max(Posit x, Posit y){
        Posit p;
        p.v = pmax(x.v, y.v);
        return p;
    }

    static inline Posit sqrt(Posit x){
        Posit p;
        p.v = psqrt(x.v);
        return p;
    }

    Posit& operator+=(Posit y) {
        v = padd(v, y.v);
        return *this;
    }

    Posit& operator-=(Posit y) {
        v = psub(v, y.v);
        return *this;
    }

    Posit& operator*=(Posit y) {
        v = pmul(v, y.v);
        return *this;
    }

    Posit& operator/=(Posit y) {
        v = pdiv(v, y.v);
        return *this;
    }

    friend inline Posit operator+(Posit x, Posit y) { return x += y; }
    friend inline Posit operator-(Posit x) { return Posit(0) -= x; }
    friend inline Posit operator-(Posit x, Posit y) { return x -= y; }
    friend inline Posit operator*(Posit x, Posit y) { return x *= y; }
    friend inline Posit operator/(Posit x, Posit y) { return x /= y; }

    friend inline bool operator==(Posit x, Posit y) { return peq(x.v, y.v); }
    friend inline bool operator< (Posit x, Posit y) { return plt(x.v, y.v); }
    friend inline bool operator<=(Posit x, Posit y) { return ple(x.v, y.v); }
    friend inline bool operator> (Posit x, Posit y) { return !ple(x.v, y.v); }
    friend inline bool operator>=(Posit x, Posit y) { return !plt(x.v, y.v); }

    template<typename T>
    friend inline Posit operator+(Posit x, T y) { return x += Posit(y); }
    template<typename T>
    friend inline Posit operator+(T x, Posit y) { return Posit(x) += y; }

    template<typename T>
    friend inline Posit operator-(Posit x, T y) { return x -= Posit(y); }
    template<typename T>
    friend inline Posit operator-(T x, Posit y) { return Posit(x) -= y; }

    template<typename T>
    friend inline Posit operator*(Posit x, T y) { return x *= Posit(y); }
    template<typename T>
    friend inline Posit operator*(T x, Posit y) { return Posit(x) *= y; }

    template<typename T>
    friend inline Posit operator/(Posit x, T y) { return x /= Posit(y); }
    template<typename T>
    friend inline Posit operator/(T x, Posit y) { return Posit(x) /= y; }

    template<typename T>
    friend inline bool operator==(Posit x, T y) { return peq(x.v, Posit(y).v); }
    template<typename T>
    friend inline bool operator==(T x, Posit y) { return peq(Posit(x).v, y.v); }

    template<typename T>
    friend inline bool operator< (Posit x, T y) { return plt(x.v, Posit(y).v); }
    template<typename T>
    friend inline bool operator< (T x, Posit y) { return plt(Posit(x).v, y.v); }

    template<typename T>
    friend inline bool operator<=(Posit x, T y) { return ple(x.v, Posit(y).v); }
    template<typename T>
    friend inline bool operator<=(T x, Posit y) { return ple(Posit(x).v, y.v); }

    template<typename T>
    friend inline bool operator> (Posit x, T y) { return !ple(x.v, Posit(y).v); }
    template<typename T>
    friend inline bool operator> (T x, Posit y) { return !ple(Posit(x).v, y.v); }

    template<typename T>
    friend inline bool operator>=(Posit x, T y) { return !plt(x.v, Posit(y).v); }
    template<typename T>
    friend inline bool operator>=(T x, Posit y) { return !plt(Posit(x).v, y.v); }
};

class Quire {
    [[maybe_unused]] char q; // dummy

public:
    constexpr Quire() : q(' ') { }

    inline void clear() {
        qclr();
    }

    inline Posit round() {
        return Posit::from_bits(qround());
    }

    inline void neg() {
        qneg();
    }

    inline void add_mul(Posit x, Posit y) {
        qmadd(x.bits(), y.bits());
    }

    inline void sub_mul(Posit x, Posit y) {
        qmsub(x.bits(), y.bits());
    }
};

namespace libposit {
    static inline Posit abs(Posit x) {
        return Posit::from_bits(psgnjxs(x.bits(), x.bits()));
    }
    static inline Posit fabs(Posit x) { return abs(x); }

    static inline Posit sqrt(Posit x) { return Posit::sqrt(x); }
    static inline Posit sqrtf(Posit x) { return sqrt(x); }
}

#endif
