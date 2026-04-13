#include <stdlib.h>

#include <imu/imu_kernels.h>

#ifdef FXP_MODE

typedef struct {
    int16_t first;
    int16_t last;
} fxp_azc_segment_t;

static inline int32_t fxp_azc_diff(int32_t a, int32_t b, int16_t gap)
{
    return (gap == 0) ? 0 : (b - a) / (int32_t)gap;
}

static void fxp_azc_interp(int16_t len, int16_t xf, int32_t yf,
                           int16_t xl, int32_t yl, int32_t *res)
{
    res[0] = yf;
    res[len - 1] = yl;

    int32_t dy = yl - yf;
    int16_t dx = xl - xf;
    for (int16_t i = 1; i < len - 1; i++) {
        res[i] = yf + (dy * i) / dx;
    }
}

static uint32_t fxp_azc_max_vdist(const int32_t *sig, int16_t first,
                                  int16_t last, int16_t *idx)
{
    if (first == last) {
        *idx = first;
        return 0;
    }

    int16_t len = last - first + 1;
    int32_t *intrp = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (intrp == NULL) {
        *idx = first;
        return 0;
    }

    fxp_azc_interp(len, first, sig[first], last, sig[last], intrp);

    uint32_t max_dist = 0;
    *idx = first;
    for (int16_t i = 0; i < len; i++) {
        uint32_t d = (uint32_t)fxp_abs_s32(sig[first + i] - intrp[i]);
        if (d > max_dist) {
            max_dist = d;
            *idx = first + i;
        }
    }

    free(intrp);
    return max_dist;
}

static int16_t *fxp_azc_polygonal_approx(const int32_t *sig, int16_t len,
                                         uint32_t eps_fxp, int16_t *res_len)
{
    int16_t *res = (int16_t *)malloc((size_t)len * sizeof(int16_t));
    fxp_azc_segment_t *stack = (fxp_azc_segment_t *)malloc((size_t)len * sizeof(fxp_azc_segment_t));
    if (res == NULL || stack == NULL) {
        free(res);
        free(stack);
        *res_len = 0;
        return NULL;
    }

    int16_t found = 0;
    stack[0].first = 0;
    stack[0].last = len - 1;
    int16_t next = 0;

    while (next >= 0) {
        int16_t first = stack[next].first;
        int16_t last = stack[next].last;
        next--;

        int16_t mid;
        uint32_t max_dist = fxp_azc_max_vdist(sig, first, last, &mid);

        if (max_dist > eps_fxp) {
            stack[next + 1].first = first;
            stack[next + 1].last = mid;
            stack[next + 2].first = mid;
            stack[next + 2].last = last;
            next += 2;
        } else {
            int16_t add_first = 1;
            int16_t add_last = 1;

            for (int16_t j = 0; j < found; j++) {
                if (first == res[j]) add_first = 0;
                if (last == res[j]) add_last = 0;
            }

            if (add_first) res[found++] = first;
            if (add_last) res[found++] = last;
        }
    }

    free(stack);
    *res_len = found;
    return res;
}

static int fxp_azc_qsort_cmp(const void *a, const void *b)
{
    return (int)(*(const int16_t *)a) - (int)(*(const int16_t *)b);
}

static int16_t fxp_azc_impl(const int32_t *sig, int16_t len, uint32_t eps_fxp)
{
    if (len <= 1) return 0;

    int16_t approx_len = 0;
    int16_t *idxs = fxp_azc_polygonal_approx(sig, len, eps_fxp, &approx_len);
    if (idxs == NULL || approx_len <= 0) {
        free(idxs);
        return 0;
    }

    qsort(idxs, (size_t)approx_len, sizeof(int16_t), fxp_azc_qsort_cmp);

    int16_t azc = 0;
    if (approx_len > 2) {
        int32_t prev = fxp_azc_diff(sig[idxs[0]], sig[idxs[1]], (int16_t)(idxs[1] - idxs[0]));
        for (int16_t i = 1; i < approx_len - 1; i++) {
            int32_t cur = fxp_azc_diff(sig[idxs[i]], sig[idxs[i + 1]], (int16_t)(idxs[i + 1] - idxs[i]));
            if ((prev > 0 && cur < 0) || (prev < 0 && cur > 0)) azc++;
            prev = cur;
        }
    }

    free(idxs);
    return azc;
}

int16_t fxp_azc_computation_raw(const q11_5_t *sig, int16_t len, float epsilon)
{
    if (len <= 0) return 0;

    uint32_t eps = fxp_azc_eps_raw(epsilon);
    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, eps);
    free(wide);
    return result;
}

int16_t fxp_azc_computation_l2a(const uq10_6_t *sig, int16_t len, float epsilon)
{
    if (len <= 0) return 0;

    uint32_t eps = fxp_azc_eps_l2a(epsilon);
    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, eps);
    free(wide);
    return result;
}

int16_t fxp_azc_computation_l2g(const uq5_11_t *sig, int16_t len, float epsilon)
{
    if (len <= 0) return 0;

    uint32_t eps = fxp_azc_eps_l2g(epsilon);
    int32_t *wide = (int32_t *)malloc((size_t)len * sizeof(int32_t));
    if (wide == NULL) return 0;

    for (int16_t i = 0; i < len; i++) wide[i] = (int32_t)sig[i];
    int16_t result = fxp_azc_impl(wide, len, eps);
    free(wide);
    return result;
}

#endif
