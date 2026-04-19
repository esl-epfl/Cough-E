#include <audio/audio_fft_kernels.h>

#if defined(FXP_MODE) && defined(FIXED_POINT)

#define FXP_ROLLOFF_95_Q16 ((uint32_t)62259U) /* round(0.95 * 2^16) */

static inline q13_19_t fxp_audio_fft_dev_q19(uq12_20_t freq_q20, uq10_21_t centroid_q21)
{
    uint32_t freq_q19 = freq_q20 >> 1;
    uint32_t cent_q19 = centroid_q21 >> 2;
    return (q13_19_t)((int32_t)freq_q19 - (int32_t)cent_q19);
}

uq12_20_t fxp_audio_fft_rolloff_q20(const fxp_audio_fft_view_t *view)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0)
        return 0;
    if (view->sum_mags_q17 == 0)
        return 0;

    uq15_17_t rolloff_energy_q17 =
        (uq15_17_t)((((uint64_t)view->sum_mags_q17 * (uint64_t)FXP_ROLLOFF_95_Q16) + (1ULL << 15)) >> 16);

    uq15_17_t running_sum_q17 = 0;
    for (int16_t i = 0; i < view->len; i++)
    {
        running_sum_q17 += (view->mags_q20[i] >> 3); /* UQ12.20 -> UQ15.17 */
        if (running_sum_q17 >= rolloff_energy_q17)
        {
            return view->freqs_q20[i];
        }
    }
    return view->freqs_q20[view->len - 1];
}

uq10_21_t fxp_audio_fft_centroid_q21(const fxp_audio_fft_view_t *view)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0)
        return 0;
    if (view->sum_mags_q17 == 0)
        return 0;

    /* 64-bit formulation:
     * centroid = sum(freq * mag) / sum(mag)
     * per-term contribution in UQ?.23: (freq_q20 * mag_q20) / sum_q17
     * final conversion to UQ?.21 is SR(2).
     */
    uint64_t accum_q23 = 0;
    for (int16_t i = 0; i < view->len; i++)
    {
        uint64_t prod_q24_40 = (uint64_t)view->freqs_q20[i] * (uint64_t)view->mags_q20[i];
        uint64_t term_q23 = (prod_q24_40 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        accum_q23 += term_q23;
    }

    /* UQ?.23 -> UQ11.21 */
    uint64_t centroid_raw = (accum_q23 + 2ULL) >> 2;
    return (uq10_21_t)fxp_sat_u32_from_u64(centroid_raw);
}

uq11_5_t fxp_audio_fft_spread_q5(const fxp_audio_fft_view_t *view, uq10_21_t centroid_q21)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0)
        return 0;
    if (view->sum_mags_q17 == 0)
        return 0;

    /* 64-bit formulation:
     * mean_dev2_q10 = sum( (dev2_q7 * mag_q20) / sum_q17 )
     * where dev2_q7 = (dev_q19^2) >> 31.
     */
    uint64_t mean_q22_10 = 0;
    for (int16_t i = 0; i < view->len; i++)
    {
        q13_19_t dev_q19 = fxp_audio_fft_dev_q19(view->freqs_q20[i], centroid_q21);
        uint64_t dev2_q25_7 = ((uint64_t)((int64_t)dev_q19 * (int64_t)dev_q19)) >> 31;
        uint64_t weighted_q37_27 = dev2_q25_7 * (uint64_t)view->mags_q20[i];
        uint64_t term_q10 = (weighted_q37_27 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        mean_q22_10 += term_q10;
    }

    return (uq11_5_t)fxp_sat_u16_from_u32((uint32_t)fxp_sqrt64(mean_q22_10));
}

uq7_15_t fxp_audio_fft_kurtosis_q15(const fxp_audio_fft_view_t *view, uq10_21_t centroid_q21, uq11_5_t spread_q5)
{
    if (!view || !view->mags_q20 || !view->freqs_q20 || view->len <= 0)
        return 0;
    if (view->sum_mags_q17 == 0 || spread_q5 == 0)
        return 0;

    uint64_t kurt_q15 = 0;
    for (int16_t i = 0; i < view->len; i++)
    {
        q13_19_t dev_q19 = fxp_audio_fft_dev_q19(view->freqs_q20[i], centroid_q21);

        /* norm_dev in Q11 from dev_q19 / spread_q5:
         * norm_q11 = (dev_q19 / 8) / spread_q5
         */
        int64_t num = (dev_q19 >= 0) ? ((int64_t)dev_q19 >> 3) : -(((int64_t)(-dev_q19)) >> 3);
        int64_t norm_q11 = (num >= 0)
                               ? ((num + ((int64_t)spread_q5 >> 1)) / (int64_t)spread_q5)
                               : -(((-num) + ((int64_t)spread_q5 >> 1)) / (int64_t)spread_q5);

        uint64_t abs_norm_q11 = (norm_q11 < 0) ? (uint64_t)(-norm_q11) : (uint64_t)norm_q11;
        if (abs_norm_q11 > 65535ULL)
            abs_norm_q11 = 65535ULL; /* keeps square-of-square in uint64_t */

        uint64_t norm2_q22 = abs_norm_q11 * abs_norm_q11;
        uint64_t norm4_q12 = (norm2_q22 * norm2_q22 + (1ULL << 31)) >> 32;

        /* term_q15 = norm4_q12 * mag_q20 / sum_q17 */
        uint64_t weighted_q32 = norm4_q12 * (uint64_t)view->mags_q20[i];
        uint64_t term_q15 = (weighted_q32 + ((uint64_t)view->sum_mags_q17 >> 1)) / (uint64_t)view->sum_mags_q17;
        kurt_q15 += term_q15;
    }

    return (uq7_15_t)fxp_sat_u32_from_u64(kurt_q15);
}

#endif
