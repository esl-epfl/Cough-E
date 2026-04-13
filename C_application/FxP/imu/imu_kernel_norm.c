#include <imu/imu_kernels.h>

#ifdef FXP_MODE

uq10_6_t fxp_l2_norm_accel_from_raw(q11_5_t ax, q11_5_t ay, q11_5_t az)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(ax, ax)
                 + (uint32_t)fxp_mul_s32(ay, ay)
                 + (uint32_t)fxp_mul_s32(az, az);

    return (uq10_6_t)fxp_sat_u16_from_u32(fxp_sqrt32(sum << 2));
}

uq5_11_t fxp_l2_norm_gyro_from_raw(q11_5_t gx, q11_5_t gy, q11_5_t gz)
{
    uint32_t sum = (uint32_t)fxp_mul_s32(gx, gx)
                 + (uint32_t)fxp_mul_s32(gy, gy)
                 + (uint32_t)fxp_mul_s32(gz, gz);

    return (uq5_11_t)fxp_sat_u16_from_u32((uint32_t)fxp_sqrt64((uint64_t)sum << 12));
}

uq10_6_t fxp_l2_norm_accel(float ax, float ay, float az)
{
    q11_5_t qx = FXP_IMU_RAW_FROM_FLOAT(ax);
    q11_5_t qy = FXP_IMU_RAW_FROM_FLOAT(ay);
    q11_5_t qz = FXP_IMU_RAW_FROM_FLOAT(az);
    return fxp_l2_norm_accel_from_raw(qx, qy, qz);
}

uq5_11_t fxp_l2_norm_gyro(float gx, float gy, float gz)
{
    q11_5_t qx = FXP_IMU_RAW_FROM_FLOAT(gx);
    q11_5_t qy = FXP_IMU_RAW_FROM_FLOAT(gy);
    q11_5_t qz = FXP_IMU_RAW_FROM_FLOAT(gz);
    return fxp_l2_norm_gyro_from_raw(qx, qy, qz);
}

#endif
