/*============================================================================

This source file is part of the SoftPosit Posit Arithmetic Package
by S. H. Leong (Cerlane).

Copyright 2017, 2018 A*STAR.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#define signP32UI(a) ((bool)((uint32_t)(a) >> 31))
#define signregP32UI(a) ((bool)(((uint32_t)(a) >> 30) & 0x1))

constexpr void checkExtraP32TwoBits(float f32, float temp,
                                        bool *bitsNPlusOne, bool *bitsMore) {
    temp /= 2;
    if (temp <= f32) {
        *bitsNPlusOne = 1;
        f32 -= temp;
    }
    if (f32 > 0)
        *bitsMore = 1;
}

constexpr uint_fast32_t convertFractionP32(float f32,
                                               uint_fast16_t fracLength,
                                               bool *bitsNPlusOne,
                                               bool *bitsMore) {
    uint_fast32_t frac = 0;

    if (f32 == 0)
        return 0;
    else if (f32 == INFINITY)
        return 0x80000000;

    f32 -= 1; // remove hidden bit
    if (fracLength == 0)
        checkExtraP32TwoBits(f32, 1.0, bitsNPlusOne, bitsMore);
    else {
        float temp = 1;
        while (true) {
            temp /= 2;
            if (temp <= f32) {
                f32 -= temp;
                fracLength--;
                frac = (frac << 1) + 1; // shift in one
                if (f32 == 0) {
                    frac <<= (uint_fast16_t)fracLength;
                    break;
                }

                if (fracLength == 0) {
                    checkExtraP32TwoBits(f32, temp, bitsNPlusOne, bitsMore);
                    break;
                }
            } else {
                frac <<= 1; // shift in a zero
                fracLength--;
                if (fracLength == 0) {
                    checkExtraP32TwoBits(f32, temp, bitsNPlusOne, bitsMore);
                    break;
                }
            }
        }
    }

    return frac;
}

constexpr uint32_t convertFloatToP32(float f32) {
    uint32_t result = 0;
    bool sign, regS;
    uint_fast32_t reg, frac = 0;
    int_fast32_t exp = 0;
    bool bitNPlusOne = 0, bitsMore = 0;

    (f32 >= 0) ? (sign = 0) : (sign = 1);

    if (f32 == 0) {
        return 0;
    } else if (f32 == INFINITY || f32 == -INFINITY || f32 == NAN) {
        return 0x80000000;
    } else if (f32 == 1) {
        return 0x40000000;
    } else if (f32 == -1) {
        return 0xC0000000;
    } else if (f32 >= 1.329227995784916e+36) {
        // maxpos
        return 0x7FFFFFFF;
    } else if (f32 <= -1.329227995784916e+36) {
        // -maxpos
        return 0x80000001;
    } else if (f32 <= 7.52316384526264e-37 && !sign) {
        // minpos
        return 0x1;
    } else if (f32 >= -7.52316384526264e-37 && sign) {
        //-minpos
        return 0xFFFFFFFF;
    } else if (f32 > 1 || f32 < -1) {
        if (sign) {
            // Make negative numbers positive for easier computation
            f32 = -f32;
        }

        regS = 1;
        reg = 1; // because k = m-1; so need to add back 1
        // minpos
        if (f32 <= 7.52316384526264e-37) {
            result = 1;
        } else {
            // regime
            while (f32 >= 16) {
                f32 *= 0.0625; // f32/=16;
                reg++;
            }
            while (f32 >= 2) {
                f32 *= 0.5;
                exp++;
            }

            int8_t fracLength = 28 - reg;

            if (fracLength < 0) {
                // in both cases, reg=29 and 30, e is n+1 bit and
                // frac are sticky bits
                if (reg == 29) {
                    bitNPlusOne = exp & 0x1;
                    exp >>= 1; // taken care of by the pack
                               // algo
                } else {       // reg=30
                    bitNPlusOne = exp >> 1;
                    bitsMore = exp & 0x1;
                    exp = 0;
                }
                if (f32 != 1) { // because of hidden bit
                    bitsMore = 1;
                    frac = 0;
                }
            } else
                frac = convertFractionP32(f32, fracLength, &bitNPlusOne,
                                          &bitsMore);

            if (reg > 30) {
                (regS) ? (result = 0x7FFFFFFF) : (result = 0x1);
            }
            // rounding off fraction bits
            else {

                uint_fast32_t regime = 1;
                if (regS)
                    regime = ((1 << reg) - 1) << 1;
                if (reg <= 28)
                    exp <<= (28 - reg);
                result = ((uint32_t)(regime) << (30 - reg)) + ((uint32_t)exp) +
                         ((uint32_t)(frac));
                result +=
                    (bitNPlusOne & (result & 1)) | (bitNPlusOne & bitsMore);
            }
            if (sign)
                result = -result & 0xFFFFFFFF;
        }
    } else if (f32 < 1 || f32 > -1) {
        if (sign) {
            // Make negative numbers positive for easier computation
            f32 = -f32;
        }
        regS = 0;
        reg = 0;

        // regime
        while (f32 < 1) {
            f32 *= 16;
            reg++;
        }

        while (f32 >= 2) {
            f32 *= 0.5;
            exp++;
        }

        // only possible combination for reg=15 to reach here is 7FFF
        // (maxpos) and FFFF (-minpos) but since it should be caught on
        // top, so no need to handle
        int_fast8_t fracLength = 28 - reg;
        if (fracLength < 0) {
            // in both cases, reg=29 and 30, e is n+1 bit and frac are
            // sticky bits
            if (reg == 29) {
                bitNPlusOne = exp & 0x1;
                exp >>= 1; // taken care of by the pack algo
            } else {       // reg=30
                bitNPlusOne = exp >> 1;
                bitsMore = exp & 0x1;
                exp = 0;
            }
            if (f32 != 1) { // because of hidden bit
                bitsMore = 1;
                frac = 0;
            }
        } else
            frac = convertFractionP32(f32, fracLength, &bitNPlusOne, &bitsMore);

        if (reg > 30) {
            (regS) ? (result = 0x7FFFFFFF) : (result = 0x1);
        }
        // rounding off fraction bits
        else {

            uint_fast32_t regime = 1;
            if (regS)
                regime = ((1 << reg) - 1) << 1;
            if (reg <= 28)
                exp <<= (28 - reg);
            result = ((uint32_t)(regime) << (30 - reg)) + ((uint32_t)exp) +
                     ((uint32_t)(frac));
            result += (bitNPlusOne & (result & 1)) | (bitNPlusOne & bitsMore);
        }
        if (sign)
            result = -result & 0xFFFFFFFF;
    } else {
        // NaR - for NaN, INF and all other combinations
        result = 0x80000000;
    }
    return result;
}

constexpr float convertP32ToFloat(uint32_t a) {
    float d32;

    if (a == 0) {
        return 0;
    } else if (a == 0x7FFFFFFF) { // maxpos
        return 1.329227995784916e+36;
    } else if (a == 0x80000001) { //-maxpos
        return -1.329227995784916e+36;
    } else if (a == 0x80000000) {
        return INFINITY;
    }

    bool regS, sign;
    uint_fast32_t reg, shift = 2, frac, tmp;
    int_fast32_t k = 0;
    int_fast8_t exp;
    float fraction_max;

    sign = signP32UI(a);
    if (sign)
        a = -a & 0xFFFFFFFF;
    regS = signregP32UI(a);

    tmp = (a << 2) & 0xFFFFFFFF;
    if (regS) {
        while (tmp >> 31) {
            k++;
            shift++;
            tmp = (tmp << 1) & 0xFFFFFFFF;
        }
        reg = k + 1;
    } else {
        k = -1;
        while (!(tmp >> 31)) {
            k--;
            shift++;
            tmp = (tmp << 1) & 0xFFFFFFFF;
        }
        tmp &= 0x7FFFFFFF;
        reg = -k;
    }
    exp = tmp >> 29;
    frac = (tmp & 0x1FFFFFFF) >> shift;

    fraction_max = (reg > 28) ? 1 : powf(2, 28 - reg);
    d32 = (float)(powf(16, k) * powf(2, exp) *
                   (1 + ((float)frac / fraction_max)));
    if (sign)
        d32 = -d32;

    return d32;
}
