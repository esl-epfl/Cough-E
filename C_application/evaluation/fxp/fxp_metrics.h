#pragma once

#include <math.h>
#include <stdio.h>

typedef struct {
    int n;
    double sum_sq_err;
    double sum_sq_ref;
    double sum_abs_err;
    double sum_abs_ref;
    double max_abs_err;
    double max_abs_ref;
    unsigned long saturation_count;
    unsigned long overflow_count;
} fxp_metric_acc_t;

static inline void fxp_metric_init(fxp_metric_acc_t *m)
{
    m->n = 0;
    m->sum_sq_err = 0.0;
    m->sum_sq_ref = 0.0;
    m->sum_abs_err = 0.0;
    m->sum_abs_ref = 0.0;
    m->max_abs_err = 0.0;
    m->max_abs_ref = 0.0;
    m->saturation_count = 0UL;
    m->overflow_count = 0UL;
}

static inline void fxp_metric_add(fxp_metric_acc_t *m, double ref, double fxp)
{
    double err = fxp - ref;
    double abs_err = fabs(err);
    double abs_ref = fabs(ref);

    m->n += 1;
    m->sum_sq_err += err * err;
    m->sum_sq_ref += ref * ref;
    m->sum_abs_err += abs_err;
    m->sum_abs_ref += abs_ref;
    if (abs_err > m->max_abs_err) m->max_abs_err = abs_err;
    if (abs_ref > m->max_abs_ref) m->max_abs_ref = abs_ref;
}

static inline void fxp_metric_count_saturation(fxp_metric_acc_t *m)
{
    m->saturation_count += 1UL;
}

static inline void fxp_metric_count_overflow(fxp_metric_acc_t *m)
{
    m->overflow_count += 1UL;
}

static inline double fxp_metric_rmse(const fxp_metric_acc_t *m)
{
    return (m->n > 0) ? sqrt(m->sum_sq_err / (double)m->n) : 0.0;
}

static inline double fxp_metric_rel_rmse_pct(const fxp_metric_acc_t *m)
{
    if (m->n <= 0 || m->sum_sq_ref <= 0.0) return 0.0;
    return 100.0 * sqrt(m->sum_sq_err / m->sum_sq_ref);
}

static inline double fxp_metric_wape_pct(const fxp_metric_acc_t *m)
{
    return (m->sum_abs_ref > 0.0) ? (100.0 * m->sum_abs_err / m->sum_abs_ref) : 0.0;
}

static inline double fxp_metric_max_abs_pct(const fxp_metric_acc_t *m)
{
    return (m->max_abs_ref > 0.0) ? (100.0 * m->max_abs_err / m->max_abs_ref) : 0.0;
}

static inline double fxp_metric_sample_pct(double ref, double fxp)
{
    double denom = fabs(ref);
    return (denom > 0.0) ? (100.0 * fabs(fxp - ref) / denom) : 0.0;
}

static inline void fxp_metric_print_stage(const char *prefix,
                                          const char *mode,
                                          const char *block,
                                          const char *stage,
                                          const char *qformat,
                                          const fxp_metric_acc_t *m)
{
    printf("%s,mode=%s,block=%s,kernel=%s,stage=%s,backend=fxp,qformat=%s,n=%d,rmse=%.17g,rel_rmse_pct=%.9g,wape_pct=%.9g,max_abs=%.17g,max_abs_pct=%.9g,saturation_count=%lu,overflow_count=%lu\n",
           prefix,
           mode,
           block,
           stage,
           stage,
           qformat,
           m->n,
           fxp_metric_rmse(m),
           fxp_metric_rel_rmse_pct(m),
           fxp_metric_wape_pct(m),
           m->max_abs_err,
           fxp_metric_max_abs_pct(m),
           m->saturation_count,
           m->overflow_count);
}

static inline void fxp_metric_print_stage_ex(const char *prefix,
                                             const char *mode,
                                             const char *block,
                                             const char *kernel,
                                             const char *stage,
                                             const char *backend,
                                             const char *qformat,
                                             const fxp_metric_acc_t *m)
{
    printf("%s,mode=%s,block=%s,kernel=%s,stage=%s,backend=%s,qformat=%s,n=%d,rmse=%.17g,rel_rmse_pct=%.9g,wape_pct=%.9g,max_abs=%.17g,max_abs_pct=%.9g,saturation_count=%lu,overflow_count=%lu\n",
           prefix,
           mode,
           block,
           kernel,
           stage,
           backend,
           qformat,
           m->n,
           fxp_metric_rmse(m),
           fxp_metric_rel_rmse_pct(m),
           fxp_metric_wape_pct(m),
           m->max_abs_err,
           fxp_metric_max_abs_pct(m),
           m->saturation_count,
           m->overflow_count);
}
