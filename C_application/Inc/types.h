#pragma once

#ifdef FXP_MODE
#include <FxP/core/fxp_core.h>
#endif

#ifdef UPOS_MODE // universal posit
    #ifndef POSIT_SIZE
    #define POSIT_SIZE 16
    #endif

    #include <universal/number/posit/posit.hpp>
    #include <quire_wrapper.h>
    using namespace sw::universal;

    typedef float real_t;
    typedef Posit sreal_t;
    typedef Quire quire_t;

    #define POS(hexval) ([]{ \
        sw::universal::posit<POSIT_SIZE,2> p; \
        p.setbits(hexval); \
        return p; \
    }())

    template <typename T>
    concept Universal = requires(T x) { T::nbits; };
#else
#ifdef LPOS_MODE // libposit
    #ifndef POSIT_SIZE
    #define POSIT_SIZE 16
    #endif

    #include <posit.hpp>
    using namespace libposit;

    typedef float real_t;
    typedef Posit sreal_t;
    typedef Quire quire_t;

    #define POS Posit::from_bits
#else
    #include <quire_wrapper.h>
    typedef float real_t;
    typedef float sreal_t;
    typedef Quire quire_t;
#endif
#endif

const sreal_t CONST_ZERO = sreal_t(0.0f);
const sreal_t CONST_ONE = sreal_t(1.0f);
const sreal_t CONST_TWO = sreal_t(2.0f);
const sreal_t CONST_THREE = sreal_t(3.0f);
const sreal_t CONST_FOUR = sreal_t(4.0f);
const sreal_t CONST_TEN = sreal_t(10.0f);
const sreal_t CONST_NEG_ONE = sreal_t(-1.0f);
const sreal_t CONST_095 = sreal_t(0.95f);

#if defined(UPOS_MODE) || defined(LPOS_MODE) // posit16
const sreal_t F_MIN = POS(0x0001);
#else // float
const sreal_t F_MIN = sreal_t(1.17549e-038);
#endif

#ifndef FXP_MODE
typedef sreal_t audio_data_t;
typedef sreal_t imu_data_t;

typedef sreal_t feat_t;
typedef sreal_t audio_feat_t;
typedef sreal_t imu_feat_t;

typedef sreal_t audio_score_t;
typedef sreal_t imu_score_t;
#endif // for FXP_MODE, these are defined in fxp_core.h

#ifdef UPOS_MODE
#include <concepts>
template <typename T>
concept RealType = std::same_as<T, float> || std::integral<T> || Universal<T>;
#else
// use generic typename for non-upos (including lpos) because concept is c++20
#define RealType typename
#endif

#ifdef HEEPATIA_MODE
#define DINTL0 __attribute__((section(".xheep_data_interleaved_sec0"))) __attribute__((aligned(4)))
#define DINTL1 __attribute__((section(".xheep_data_interleaved_sec1"))) __attribute__((aligned(4)))
#define FLASH0 __attribute__((section(".xheep_data_flash_only_sec0"))) __attribute__((aligned(4)))
#define FLASH1 __attribute__((section(".xheep_data_flash_only_sec1"))) __attribute__((aligned(4)))
#define FLASH2 __attribute__((section(".xheep_data_flash_only_sec2"))) __attribute__((aligned(4)))
#define FLASH3 __attribute__((section(".xheep_data_flash_only_sec3"))) __attribute__((aligned(4)))
#define CARUS00 __attribute__((section(".xheep_data_carus0_sec0"))) __attribute__((aligned(4)))
#define CARUS01 __attribute__((section(".xheep_data_carus0_sec1"))) __attribute__((aligned(4)))
#define CARUS10 __attribute__((section(".xheep_data_carus1_sec0"))) __attribute__((aligned(4)))
#define CARUS11 __attribute__((section(".xheep_data_carus1_sec1"))) __attribute__((aligned(4)))
#else
#define DINTL0
#define DINTL1
#define FLASH0
#define FLASH1
#define FLASH2
#define FLASH3
#define CARUS00
#define CARUS01
#define CARUS10
#define CARUS11
#endif
