#include <imu/imu_kernels.h>

#ifdef FXP_MODE

uq5_11_t fxp_get_max_l2g(const uq5_11_t *sig, int16_t len)
{
    if (len <= 0) return 0;

    uq5_11_t max = sig[0];
    for (int16_t i = 1; i < len; i++) {
        if (sig[i] > max) max = sig[i];
    }
    return max;
}

#endif
