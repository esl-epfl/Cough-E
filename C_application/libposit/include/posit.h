// posit.h

#ifndef POSIT_H
#define POSIT_H

#include <stdint.h>
#include <stdbool.h>

#if POSIT_SIZE == 16
typedef uint16_t posit_t;
#else
#if POSIT_SIZE == 32
typedef uint32_t posit_t;
#else
#error "Invalid POSIT_SIZE"
#endif
#endif

static inline posit_t padd(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "padd.s %0,%1,%2"

        : "=p" (result)
        : "%p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t psub(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "psub.s %0,%1,%2"

        : "=p" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t pmul(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "pmul.s %0,%1,%2"

        : "=p" (result)
        : "%p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t pdiv(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "pdiv.s %0,%1,%2"

        : "=p" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t psqrt(posit_t x) {
    posit_t result;

    asm inline volatile (
        "psqrt.s %0,%1"

        : "=p" (result)
        : "p" (x)
        :
    );

    return result;
}

static inline posit_t pmin(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "pmin.s %0,%1,%2"

        : "=p" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t pmax(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "pmax.s %0,%1,%2"

        : "=p" (result)
        : "%p" (x), "p" (y)
        :
    );

    return result;
}

static inline bool peq(posit_t x, posit_t y) {
    bool result;

    asm inline volatile (
        "peq.s %0,%1,%2"

        : "=r" (result)
        : "%p" (x), "p" (y)
        :
    );

    return result;
}

static inline bool plt(posit_t x, posit_t y) {
    bool result;

    asm inline volatile (
        "plt.s %0,%1,%2"

        : "=r" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline bool ple(posit_t x, posit_t y) {
    bool result;

    asm inline volatile (
        "ple.s %0,%1,%2"

        : "=r" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline void qclr() {
    asm inline volatile (
        "qclr.s"

        :::
    );
}

static inline void qneg() {
    asm inline volatile (
        "qneg.s"

        :::
    );
}

static inline void qmadd(posit_t x, posit_t y) {
    asm inline volatile (
        "qmadd.s %0,%1"

        :
        : "%p" (x), "p" (y)
        :
    );
}

static inline void qmsub(posit_t x, posit_t y) {
    asm inline volatile (
        "qmsub.s %0,%1"

        :
        : "%p" (x), "p" (y)
        :
    );
}

static inline posit_t qround() {
    posit_t result;

    asm inline volatile (
        "qround.s %0"

        : "=p" (result)
        :
        :
    );

    return result;
}

static inline posit_t psgnjxs(posit_t x, posit_t y) {
    posit_t result;

    asm inline volatile (
        "psgnjx.s %0,%1,%2"

        : "=p" (result)
        : "p" (x), "p" (y)
        :
    );

    return result;
}

static inline posit_t plw(posit_t x) {
    posit_t result;

    asm inline volatile (
        "plw %0,%1"

        : "=p" (result)
        : "m" (x)
        :
    );

    return result;
}

static inline posit_t psw(posit_t x) {
    posit_t result;

    asm inline volatile (
        "psw %1,%0"

        : "=m" (result)
        : "p" (x)
        :
    );

    return result;
}

static inline posit_t pmvwx(posit_t x) {
    posit_t result;

    asm inline volatile (
        "pmv.w.x %0,%1"

        : "=p" (result)
        : "r" (x)
        :
    );

    return result;
}

static inline posit_t pmvxw(posit_t x) {
    posit_t result;

    asm inline volatile (
        "pmv.x.w %0,%1"

        : "=r" (result)
        : "p" (x)
        :
    );

    return result;
}

#endif
