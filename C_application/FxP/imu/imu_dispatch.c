#include <stdio.h>
#include <stdlib.h>

#include <imu/imu_dispatch.h>
#include <helpers.h>
#include <time_domain_feat.h>
#include <azc.h>

typedef void (*imu_kernel_fn)(const imu_sig_view_t *sig, float *out, float param);

typedef struct {
    uint8_t     feature_idx;
    imu_kernel_fn fn;
    float       param;
} imu_kernel_desc_t;

// -----------------------------------------------------------------------------
// Float kernels
// -----------------------------------------------------------------------------

static void kern_float_line_length(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_line_length((float *)sig->data.float_data, sig->len);
}

static void kern_float_zcr(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = compute_zrc((float *)sig->data.float_data, sig->len);
}

static void kern_float_kurtosis(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_kurtosis((float *)sig->data.float_data, sig->len);
}

static void kern_float_rms(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = get_rms((float *)sig->data.float_data, sig->len);
}

static void kern_float_crest(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    float rms  = get_rms((float *)sig->data.float_data, sig->len);
    float peak = get_max((float *)sig->data.float_data, sig->len);
    *out = (rms > 0.0f) ? (peak / rms) : 0.0f;
}

static void kern_float_azc(const imu_sig_view_t *sig, float *out, float param)
{
    *out = (float)azc_computation((float *)sig->data.float_data, sig->len, param);
}

#ifdef FXP_MODE
// -----------------------------------------------------------------------------
// FxP kernels
// -----------------------------------------------------------------------------

static void kern_raw_line_length(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_line_length_raw(sig->data.raw_data, sig->len),
                        FXP_FRAC_IMU_LINE_LENGTH_RAW);
}

static void kern_raw_zcr(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = fxp_compute_zcr_raw(sig->data.raw_data, sig->len);
}

static void kern_raw_kurtosis(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_kurtosis_raw(sig->data.raw_data, sig->len),
                        FXP_FRAC_IMU_KURTOSIS_RAW);
}

static void kern_raw_rms(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_rms_raw(sig->data.raw_data, sig->len),
                        FXP_FRAC_IMU_RMS_RAW);
}

static void kern_raw_azc(const imu_sig_view_t *sig, float *out, float param)
{
    *out = (float)fxp_azc_computation_raw(sig->data.raw_data, sig->len, param);
}

static void kern_l2a_rms(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_rms_l2a(sig->data.l2a_data, sig->len),
                        FXP_FRAC_IMU_RMS_L2A);
}

static void kern_l2a_azc(const imu_sig_view_t *sig, float *out, float param)
{
    *out = (float)fxp_azc_computation_l2a(sig->data.l2a_data, sig->len, param);
}

static void kern_l2g_line_length(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_line_length_l2g(sig->data.l2g_data, sig->len),
                        FXP_FRAC_IMU_LINE_LENGTH_L2G);
}

static void kern_l2g_rms(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    *out = FXP_TO_FLOAT(fxp_get_rms_l2g(sig->data.l2g_data, sig->len),
                        FXP_FRAC_IMU_RMS_L2G);
}

static void kern_l2g_crest(const imu_sig_view_t *sig, float *out, float param)
{
    (void)param;
    uq7_9_t  rms  = fxp_get_rms_l2g(sig->data.l2g_data, sig->len);
    uq5_11_t peak = fxp_get_max_l2g(sig->data.l2g_data, sig->len);
    uq2_14_t cf   = (rms > 0) ? fxp_cf_l2g_result(peak, rms) : 0;
    *out = FXP_TO_FLOAT(cf, FXP_FRAC_IMU_CREST_L2G);
}

static void kern_l2g_azc(const imu_sig_view_t *sig, float *out, float param)
{
    *out = (float)fxp_azc_computation_l2g(sig->data.l2g_data, sig->len, param);
}
#endif

#define AZC_IDX(i) ((uint8_t)(APPROXIMATE_ZERO_CROSSING + (i)))
#define AZC_EPS(i) ((float)(EPSILON_START + (EPSILON_STEP * (i))))

static const imu_kernel_desc_t k_float_table[] = {
    { LINE_LENGTH,             kern_float_line_length, 0.0f        },
    { ZERO_CROSSING_RATE_IMU,  kern_float_zcr,         0.0f        },
    { KURTOSIS,                kern_float_kurtosis,    0.0f        },
    { ROOT_MEANS_SQUARED_IMU,  kern_float_rms,         0.0f        },
    { CREST_FACTOR_IMU,        kern_float_crest,       0.0f        },
    { AZC_IDX(0),              kern_float_azc,         AZC_EPS(0)  },
    { AZC_IDX(1),              kern_float_azc,         AZC_EPS(1)  },
    { AZC_IDX(2),              kern_float_azc,         AZC_EPS(2)  },
    { AZC_IDX(3),              kern_float_azc,         AZC_EPS(3)  },
    { AZC_IDX(4),              kern_float_azc,         AZC_EPS(4)  },
    { AZC_IDX(5),              kern_float_azc,         AZC_EPS(5)  },
    { AZC_IDX(6),              kern_float_azc,         AZC_EPS(6)  },
    { AZC_IDX(7),              kern_float_azc,         AZC_EPS(7)  },
};

#ifdef FXP_MODE
static const imu_kernel_desc_t k_raw_table[] = {
    { LINE_LENGTH,             kern_raw_line_length,   0.0f        },
    { ZERO_CROSSING_RATE_IMU,  kern_raw_zcr,           0.0f        },
    { KURTOSIS,                kern_raw_kurtosis,      0.0f        },
    { ROOT_MEANS_SQUARED_IMU,  kern_raw_rms,           0.0f        },
    { AZC_IDX(0),              kern_raw_azc,           AZC_EPS(0)  },
    { AZC_IDX(1),              kern_raw_azc,           AZC_EPS(1)  },
    { AZC_IDX(2),              kern_raw_azc,           AZC_EPS(2)  },
    { AZC_IDX(3),              kern_raw_azc,           AZC_EPS(3)  },
    { AZC_IDX(4),              kern_raw_azc,           AZC_EPS(4)  },
    { AZC_IDX(5),              kern_raw_azc,           AZC_EPS(5)  },
    { AZC_IDX(6),              kern_raw_azc,           AZC_EPS(6)  },
    { AZC_IDX(7),              kern_raw_azc,           AZC_EPS(7)  },
};

static const imu_kernel_desc_t k_l2a_table[] = {
    { ROOT_MEANS_SQUARED_IMU,  kern_l2a_rms,           0.0f        },
    { AZC_IDX(0),              kern_l2a_azc,           AZC_EPS(0)  },
    { AZC_IDX(1),              kern_l2a_azc,           AZC_EPS(1)  },
    { AZC_IDX(2),              kern_l2a_azc,           AZC_EPS(2)  },
    { AZC_IDX(3),              kern_l2a_azc,           AZC_EPS(3)  },
    { AZC_IDX(4),              kern_l2a_azc,           AZC_EPS(4)  },
    { AZC_IDX(5),              kern_l2a_azc,           AZC_EPS(5)  },
    { AZC_IDX(6),              kern_l2a_azc,           AZC_EPS(6)  },
    { AZC_IDX(7),              kern_l2a_azc,           AZC_EPS(7)  },
};

static const imu_kernel_desc_t k_l2g_table[] = {
    { LINE_LENGTH,             kern_l2g_line_length,   0.0f        },
    { ROOT_MEANS_SQUARED_IMU,  kern_l2g_rms,           0.0f        },
    { CREST_FACTOR_IMU,        kern_l2g_crest,         0.0f        },
    { AZC_IDX(0),              kern_l2g_azc,           AZC_EPS(0)  },
    { AZC_IDX(1),              kern_l2g_azc,           AZC_EPS(1)  },
    { AZC_IDX(2),              kern_l2g_azc,           AZC_EPS(2)  },
    { AZC_IDX(3),              kern_l2g_azc,           AZC_EPS(3)  },
    { AZC_IDX(4),              kern_l2g_azc,           AZC_EPS(4)  },
    { AZC_IDX(5),              kern_l2g_azc,           AZC_EPS(5)  },
    { AZC_IDX(6),              kern_l2g_azc,           AZC_EPS(6)  },
    { AZC_IDX(7),              kern_l2g_azc,           AZC_EPS(7)  },
};
#endif

static void imu_get_table(imu_sig_kind_t kind,
                          const imu_kernel_desc_t **table,
                          size_t *table_len)
{
    switch (kind) {
        case IMU_SIG_KIND_FLOAT:
            *table = k_float_table;
            *table_len = sizeof(k_float_table) / sizeof(k_float_table[0]);
            return;
#ifdef FXP_MODE
        case IMU_SIG_KIND_RAW:
            *table = k_raw_table;
            *table_len = sizeof(k_raw_table) / sizeof(k_raw_table[0]);
            return;
        case IMU_SIG_KIND_L2A:
            *table = k_l2a_table;
            *table_len = sizeof(k_l2a_table) / sizeof(k_l2a_table[0]);
            return;
        case IMU_SIG_KIND_L2G:
            *table = k_l2g_table;
            *table_len = sizeof(k_l2g_table) / sizeof(k_l2g_table[0]);
            return;
#endif
        default:
            fprintf(stderr, "IMU dispatch: unknown signal kind %d.\n", (int)kind);
            abort();
    }
}

void imu_run_feature_table(const int8_t *features_selector, imu_sig_view_t sig, float *feats)
{
    const imu_kernel_desc_t *table = NULL;
    size_t table_len = 0;
    imu_get_table(sig.kind, &table, &table_len);

    for (size_t i = 0; i < table_len; i++) {
        const imu_kernel_desc_t *row = &table[i];
        if (features_selector[row->feature_idx] != 1) continue;
        row->fn(&sig, &feats[row->feature_idx], row->param);
    }
}
