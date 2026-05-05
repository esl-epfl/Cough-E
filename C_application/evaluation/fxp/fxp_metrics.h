#pragma once

#include <math.h>
#include <stdio.h>

typedef struct {
    int n;
    double sum_sq_err;
    double sum_sq_ref;
    double max_abs_err;
    double max_abs_ref;
    int nonfinite;
} fxp_metric_acc_t;

static inline void fxp_metric_init(fxp_metric_acc_t *m)
{
    m->n = 0;
    m->sum_sq_err = 0.0;
    m->sum_sq_ref = 0.0;
    m->max_abs_err = 0.0;
    m->max_abs_ref = 0.0;
    m->nonfinite = 0;
}

static inline void fxp_metric_add(fxp_metric_acc_t *m, double ref, double fxp)
{
    if (!isfinite(ref) || !isfinite(fxp)) {
        m->nonfinite += 1;
        return;
    }

    double err = fxp - ref;
    double abs_err = fabs(err);
    double abs_ref = fabs(ref);

    m->n += 1;
    m->sum_sq_err += err * err;
    m->sum_sq_ref += ref * ref;
    if (abs_err > m->max_abs_err) m->max_abs_err = abs_err;
    if (abs_ref > m->max_abs_ref) m->max_abs_ref = abs_ref;
}

static inline void fxp_metric_print_kernel_acc(const char *block,
                                               const char *kernel,
                                               const fxp_metric_acc_t *m)
{
    printf("FXP_KERNEL_ACC,block=%s,kernel=%s,n=%d,sum_sq_err=%.17g,sum_sq_ref=%.17g,max_abs_err=%.17g,max_abs_ref=%.17g,nonfinite=%d\n",
           block,
           kernel,
           m->n,
           m->sum_sq_err,
           m->sum_sq_ref,
           m->max_abs_err,
           m->max_abs_ref,
           m->nonfinite);
}
