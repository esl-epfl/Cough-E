#ifndef _QUIRE_WRAPPER_H_
#define _QUIRE_WRAPPER_H_

// this provides libposit-standard quire class for universal and float

// we use true quire in both libposit and universal posit, however, it is
// possible to use mixed precision (posit + float) by modifying this wrapper

// a generic quire that does regular accumulation
template <typename T>
class GenericQuire {
    T f;

public:
    constexpr GenericQuire() : f(0.0f) { }

    inline void clear() {
        f = 0.0f;
    }

    inline T round() {
        return f;
    }

    inline void neg() {
        f = -f;
    }

    inline void add_mul(T x, T y) {
        f += x * y;
    }

    inline void sub_mul(T x, T y) {
        f -= x * y;
    }
};

#if defined(UPOS_MODE) // universal posit
#ifndef POSIT_SIZE
#define POSIT_SIZE 16
#endif

#include <universal/number/posit/posit.hpp>
using namespace sw::universal;
typedef posit<POSIT_SIZE, 2> Posit;

#ifdef NO_UPOS_QUIRE
typedef GenericQuire<Posit> Quire;
#else
class Quire {
    quire<Posit> q;

public:
    Quire() : q() { }

    inline void clear() {
        q.clear();
    }

    inline Posit round() {
        return quire_resolve(q);
    }

    inline void neg() {
        q.set_sign(!q.isneg());
    }

    inline void add_mul(Posit x, Posit y) {
        q += quire_mul(x, y);
    }

    inline void sub_mul(Posit x, Posit y) {
        q -= quire_mul(x, y);
    }
};
#endif
#else
#if defined(LPOS_MODE) // libposit
// do nothing - we use libposit standard
#else
// NOTE: Quire for float is a simple accumulator for code consistency.
//       It does NOT convert float to double to reduce rounding error.
typedef GenericQuire<float> Quire;
#endif
#endif

#endif
