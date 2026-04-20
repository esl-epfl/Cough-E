#include <audio/audio_periodogram_kernels.h>
#include <audio/audio_periodogram_lut.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

#define FXP_LN2_Q24 ((int32_t)11629080) /* round(ln(2) * 2^24) */
#define FXP_PROXY_TO_INT_SHIFT (FXP_FRAC_AUDIO_PSD_PROXY - FXP_FRAC_AUDIO_PSD_INTEGRAL)

static inline int32_t _round_div_s64(int64_t num, int32_t den)
{
    if (den <= 0) return 0;
    if (num >= 0) return (int32_t)((num + (den / 2)) / den);
    return -(int32_t)(((-num) + (den / 2)) / den);
}

static inline uint32_t _round_div_u64(uint64_t num, uint32_t den)
{
    if (den == 0U) return 0U;
    return (uint32_t)((num + ((uint64_t)den >> 1)) / (uint64_t)den);
}

/* Floor division for signed numerator / positive denominator. */
static inline int32_t _floor_div_s64(int64_t num, int32_t den)
{
    int64_t q = num / (int64_t)den;
    int64_t r = num - q * (int64_t)den;
    if (r != 0 && num < 0) q -= 1;
    return (int32_t)q;
}

/* Natural logarithm on UQ18.14 input, result in Q5.11.
 * Piecewise-linear LUT on normalized mantissa m in [1,2).
 */
static q5_11_t _fxp_ln_proxy_q11(uq18_14_t x_q14)
{
    if (x_q14 == 0U) x_q14 = 1U; /* clamp to 1 LSB to avoid ln(0) */

    uint32_t msb = 31U - (uint32_t)__builtin_clz(x_q14);
    uint32_t base = (uint32_t)1U << msb;
    uint32_t frac_q24 = (uint32_t)((((uint64_t)(x_q14 - base)) << 24) / (uint64_t)base);

    uint32_t idx = frac_q24 >> 16; /* 8-bit index */
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = frac_q24 & 0xFFFFU;

    int32_t y0 = fxp_ln_lut_q24[idx];
    int32_t y1 = fxp_ln_lut_q24[idx + 1];
    int32_t y = y0 + (int32_t)((((int64_t)(y1 - y0) * (int64_t)alpha) + (1LL << 15)) >> 16);

    int32_t exp2 = (int32_t)msb - FXP_FRAC_AUDIO_PSD_PROXY;
    int64_t ln_x_q24 = (int64_t)exp2 * (int64_t)FXP_LN2_Q24 + (int64_t)y;
    int64_t ln_x_q11 = (ln_x_q24 >= 0) ? ((ln_x_q24 + (1LL << 12)) >> 13) : -(((-ln_x_q24) + (1LL << 12)) >> 13);

    return fxp_sat_s16_from_s32((int32_t)ln_x_q11);
}

/* Exponential on Q5.11 input, output UQ0.16.
 * Range reduction via exp(x) = 2^k * exp(r), r in [0, ln(2)),
 * with piecewise-linear LUT for exp(r).
 */
static uq0_16_t _fxp_exp_q16_from_q11(q5_11_t x_q11)
{
    /* Avoid UB: left-shifting negative signed values is undefined in C. */
    int64_t x_q24 = (int64_t)x_q11 * (int64_t)(1U << 13);
    int32_t k = _floor_div_s64(x_q24, FXP_LN2_Q24);
    int64_t rem_q24 = x_q24 - (int64_t)k * (int64_t)FXP_LN2_Q24;
    if (rem_q24 < 0) rem_q24 = 0;

    uint32_t z_q24 = (uint32_t)(((uint64_t)rem_q24 << 24) / (uint32_t)FXP_LN2_Q24); /* z = rem/ln2 in Q24 */
    uint32_t idx = z_q24 >> 16;
    if (idx >= FXP_LN_LUT_SIZE) idx = FXP_LN_LUT_SIZE - 1;
    uint32_t alpha = z_q24 & 0xFFFFU;

    uint32_t y0 = fxp_exp_lut_q24[idx];
    uint32_t y1 = fxp_exp_lut_q24[idx + 1];
    uint32_t er_q24 = y0 + (uint32_t)((((int64_t)((int32_t)y1 - (int32_t)y0) * (int64_t)alpha) + (1LL << 15)) >> 16);
    uint32_t er_q16 = (er_q24 + (1U << 7)) >> 8;

    uint64_t out_q16;
    if (k >= 0) {
        if (k >= 16) return UINT16_MAX;
        out_q16 = ((uint64_t)er_q16) << (uint32_t)k;
    } else {
        uint32_t shift = (uint32_t)(-k);
        if (shift >= 32) out_q16 = 0;
        else out_q16 = (((uint64_t)er_q16) + ((uint64_t)1U << (shift - 1U))) >> shift;
    }

    return fxp_sat_u16_from_u32(fxp_sat_u32_from_u64(out_q16));
}

static uint64_t _fxp_simpson_step_q8(const uq18_14_t *x_q14, int16_t start, int16_t end)
{
    int n_intervals = (end - start) / 2;
    int16_t idx = start;
    uint64_t sum_q8 = 0;

    for (int i = 0; i < n_intervals; i++) {
        uint64_t x0_q8 = (uint64_t)(x_q14[idx] >> FXP_PROXY_TO_INT_SHIFT);
        uint64_t x1_q8 = (uint64_t)(x_q14[idx + 1] >> FXP_PROXY_TO_INT_SHIFT);
        uint64_t x2_q8 = (uint64_t)(x_q14[idx + 2] >> FXP_PROXY_TO_INT_SHIFT);
        sum_q8 += x0_q8 + (x1_q8 << 2) + x2_q8;
        idx += 2;
    }

    return (sum_q8 + 1ULL) / 3ULL;
}

/* Composite Simpson integral with unit spacing on UQ18.14 input.
 * Output is kept as UQ24.8-equivalent integer.
 */
static uint64_t _fxp_simpson_q8(const uq18_14_t *x_q14, int16_t len)
{
    if (!x_q14 || len <= 1) return 0ULL;

    if ((len & 1) == 0) {
        uint64_t val_q8 = (((uint64_t)(x_q14[len - 1] >> FXP_PROXY_TO_INT_SHIFT) +
                             (uint64_t)(x_q14[len - 2] >> FXP_PROXY_TO_INT_SHIFT)) + 1ULL) >> 1;
        uint64_t result_q8 = _fxp_simpson_step_q8(x_q14, 0, len - 1);

        val_q8 += ((((uint64_t)(x_q14[0] >> FXP_PROXY_TO_INT_SHIFT) +
                     (uint64_t)(x_q14[1] >> FXP_PROXY_TO_INT_SHIFT)) + 1ULL) >> 1);
        result_q8 += _fxp_simpson_step_q8(x_q14, 1, len);

        val_q8 = (val_q8 + 1ULL) >> 1;
        result_q8 = (result_q8 + 1ULL) >> 1;
        return result_q8 + val_q8;
    }

    return _fxp_simpson_step_q8(x_q14, 0, len);
}

uq12_20_t fxp_audio_psd_dominant_freq_q20(const fxp_audio_psd_view_t *view)
{
    if (!view || !view->proxy_q14 || !view->freqs_q20 || view->len <= 0) return 0;

    int16_t max_idx = 0;
    uq18_14_t max_val = view->proxy_q14[0];
    for (int16_t i = 1; i < view->len; i++) {
        if (view->proxy_q14[i] > max_val) {
            max_val = view->proxy_q14[i];
            max_idx = i;
        }
    }
    return view->freqs_q20[max_idx];
}

uq0_16_t fxp_audio_psd_flatness_q16(const fxp_audio_psd_view_t *view)
{
    if (!view || view->len <= 0) return 0;

    /* log_proxy_q11[i] = ln(acc_power[i]) - ln(mean_power), so
     * mean(log_proxy) = mean(ln acc) - ln(mean acc), which is exactly
     * log(gmean) - log(amean). Preferred path: full dynamic range on the
     * geometric mean comes directly from the unnormalized accumulator.
     */
    if (view->log_proxy_q11) {
        int64_t sum_logs_q11 = 0;
        for (int16_t i = 0; i < view->len; i++) {
            sum_logs_q11 += (int64_t)view->log_proxy_q11[i];
        }
        int32_t diff_q11 = _round_div_s64(sum_logs_q11, view->len);
        if (diff_q11 > 0) diff_q11 = 0; /* spectral flatness is bounded by 1 */
        return _fxp_exp_q16_from_q11(fxp_sat_s16_from_s32(diff_q11));
    }

    /* Fallback: derive both means from the normalized Q14 proxy (hybrid path). */
    if (!view->proxy_q14) return 0;

    int64_t sum_logs_q11 = 0;
    uint64_t sum_proxy_q14 = 0;

    for (int16_t i = 0; i < view->len; i++) {
        uq18_14_t x_q14 = view->proxy_q14[i];
        if (x_q14 == 0U) x_q14 = 1U;
        sum_logs_q11 += (int64_t)_fxp_ln_proxy_q11(x_q14);
        sum_proxy_q14 += (uint64_t)x_q14;
    }

    if (sum_proxy_q14 == 0ULL) return 0;

    int32_t mean_log_q11 = _round_div_s64(sum_logs_q11, view->len);
    uq18_14_t mean_proxy_q14 = fxp_sat_u32_from_u64((uint64_t)_round_div_u64(sum_proxy_q14, (uint32_t)view->len));
    if (mean_proxy_q14 == 0U) mean_proxy_q14 = 1U;

    q5_11_t log_mean_q11 = _fxp_ln_proxy_q11(mean_proxy_q14);
    int32_t diff_q11 = mean_log_q11 - (int32_t)log_mean_q11;
    if (diff_q11 > 0) diff_q11 = 0; /* spectral flatness is bounded by 1 */
    return _fxp_exp_q16_from_q11(fxp_sat_s16_from_s32(diff_q11));
}

void fxp_audio_psd_bandpowers_q16(const fxp_audio_psd_view_t *view,
                                  const int8_t *psd_selector,
                                  uq0_16_t *band_powers_q16)
{
    if (!band_powers_q16) return;
    for (int8_t i = 0; i < N_PSD; i++) band_powers_q16[i] = 0;

    if (!view || !view->proxy_q14 || !view->freqs_q20 || !psd_selector || view->len <= 2) return;

    uint64_t total_power_q8 = _fxp_simpson_q8(view->proxy_q14, view->len);
    if (total_power_q8 == 0ULL) return;

    for (int8_t i = 0; i < N_PSD; i++) {
        if (!psd_selector[i]) continue;

        uq12_20_t band_start_q20 = (uq12_20_t)((uint32_t)psd_bands[i].start << FXP_FRAC_AUDIO_FFT_FREQUENCIES);
        uq12_20_t band_end_q20 = (uq12_20_t)((uint32_t)psd_bands[i].end << FXP_FRAC_AUDIO_FFT_FREQUENCIES);

        int16_t start_idx = 0;
        int16_t n_bins = 0;
        int found = 0;

        for (int16_t j = 0; j < view->len; j++) {
            uq12_20_t f_q20 = view->freqs_q20[j];
            if (!found && f_q20 >= band_start_q20) {
                start_idx = j;
                found = 1;
            }
            if (found && f_q20 <= band_end_q20) {
                n_bins++;
            } else if (found) {
                break;
            }
        }

        if (!found || n_bins <= 1) {
            band_powers_q16[i] = 0;
            continue;
        }

        uint64_t band_power_q8 = _fxp_simpson_q8(&view->proxy_q14[start_idx], n_bins);
        uint64_t ratio_q16 = ((band_power_q8 << 16) + (total_power_q8 >> 1)) / total_power_q8;
        band_powers_q16[i] = fxp_sat_u16_from_u32(fxp_sat_u32_from_u64(ratio_q16));
    }
}

#endif
