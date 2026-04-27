#pragma once

#include <math.h>
#include <stdio.h>

typedef struct {
    int n;
    double sum_sq_err;
    double sum_sq_ref;
    double max_abs_err;
    double max_abs_ref;
} fxp_metric_acc_t;

static inline void fxp_metric_init(fxp_metric_acc_t *m)
{
    m->n = 0;
    m->sum_sq_err = 0.0;
    m->sum_sq_ref = 0.0;
    m->max_abs_err = 0.0;
    m->max_abs_ref = 0.0;
}

static inline void fxp_metric_add(fxp_metric_acc_t *m, double ref, double fxp)
{
    double err = fxp - ref;
    double abs_err = fabs(err);
    double abs_ref = fabs(ref);

    m->n += 1;
    m->sum_sq_err += err * err;
    m->sum_sq_ref += ref * ref;
    if (abs_err > m->max_abs_err) m->max_abs_err = abs_err;
    if (abs_ref > m->max_abs_ref) m->max_abs_ref = abs_ref;
}

static inline double fxp_metric_rel_rmse_pct(const fxp_metric_acc_t *m)
{
    if (m->n <= 0 || m->sum_sq_ref <= 0.0) return 0.0;
    return 100.0 * sqrt(m->sum_sq_err / m->sum_sq_ref);
}

static inline double fxp_metric_max_abs_pct(const fxp_metric_acc_t *m)
{
    return (m->max_abs_ref > 0.0) ? (100.0 * m->max_abs_err / m->max_abs_ref) : 0.0;
}

static inline void fxp_metric_print_stage(const char *prefix,
                                          const char *mode,
                                          const char *block,
                                          const char *stage,
                                          const char *qformat,
                                          const fxp_metric_acc_t *m)
{
    printf("%s,mode=%s,block=%s,kernel=%s,stage=%s,qformat=%s,n=%d,rel_rmse_pct=%.9g,max_abs_pct=%.9g\n",
           prefix,
           mode,
           block,
           stage,
           stage,
           qformat,
           m->n,
           fxp_metric_rel_rmse_pct(m),
           fxp_metric_max_abs_pct(m));
}
