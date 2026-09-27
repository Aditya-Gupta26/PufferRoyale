/*
 * pr_math.h -- integer math for the simulation (SPEC §1: no floating point).
 *
 * Everything here is exact and platform independent:
 *   - pr_isqrt64: exact floor(sqrt(n)) on int64 (no libm)
 *   - floor / ceil / truncating division helpers (C's '/' truncates toward zero;
 *     these make the intent explicit at every call site)
 *   - PCG32 (O'Neill, pcg32_random_r) with an unbiased bounded draw
 *   - FNV-1a 64 for the state hash
 */
#ifndef PR_MATH_H
#define PR_MATH_H

#include <stddef.h>
#include <stdint.h>

#define PR_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PR_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PR_CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

static inline int32_t pr_abs32(int32_t v) { return v < 0 ? -v : v; }
static inline int64_t pr_abs64(int64_t v) { return v < 0 ? -v : v; }

/* floor(a / b) for b > 0. */
static inline int64_t pr_floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

/* ceil(a / b) for b > 0. */
static inline int64_t pr_ceildiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) == (b < 0))) q += 1;
    return q;
}

/* a / b truncated toward zero (C semantics), named so the choice is visible. */
static inline int64_t pr_truncdiv(int64_t a, int64_t b) { return a / b; }

/* a * b / c in 64-bit, truncated toward zero. */
static inline int32_t pr_muldiv(int32_t a, int32_t b, int32_t c) {
    return (int32_t)(((int64_t)a * (int64_t)b) / (int64_t)c);
}

/* Exact integer square root: the largest r with r*r <= n (n >= 0). Starts from a
 * power-of-two upper bound so Newton's iteration converges in a few steps. */
static inline int64_t pr_isqrt64(int64_t n) {
    if (n <= 0) return 0;
    if (n < 4) return 1;
    uint64_t un = (uint64_t)n;
    int bits = 64 - __builtin_clzll(un);
    uint64_t x = (uint64_t)1 << ((bits + 1) / 2); /* x >= sqrt(n) */
    for (;;) {
        uint64_t y = (x + un / x) >> 1;
        if (y >= x) break;
        x = y;
    }
    /* guard against any off-by-one at the boundary */
    while (x * x > un) x--;
    while ((x + 1) * (x + 1) <= un) x++;
    return (int64_t)x;
}

static inline int64_t pr_dist2(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    int64_t dx = (int64_t)ax - bx, dy = (int64_t)ay - by;
    return dx * dx + dy * dy;
}

static inline int32_t pr_dist(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    return (int32_t)pr_isqrt64(pr_dist2(ax, ay, bx, by));
}

/* Is the centre distance <= r? (no square root) */
static inline int pr_within(int32_t ax, int32_t ay, int32_t bx, int32_t by, int64_t r) {
    if (r < 0) return 0;
    return pr_dist2(ax, ay, bx, by) <= r * r;
}

/* Step from (x, y) toward (tx, ty) by at most `step`, never overshooting: if the
 * remaining distance is <= step the result is exactly the target. Direction
 * components truncate toward zero (SPEC §6.3). Returns the distance moved. */
static inline int32_t pr_step_toward(int32_t *x, int32_t *y, int32_t tx, int32_t ty, int32_t step) {
    if (step <= 0) return 0;
    int64_t dx = (int64_t)tx - *x, dy = (int64_t)ty - *y;
    int64_t d = pr_isqrt64(dx * dx + dy * dy);
    if (d == 0) return 0;
    if (d <= step) {
        *x = tx;
        *y = ty;
        return (int32_t)d;
    }
    *x += (int32_t)((dx * step) / d);
    *y += (int32_t)((dy * step) / d);
    return step;
}

/* ------------------------------------------------------------------ PCG32 */

typedef struct PrRng {
    uint64_t state;
    uint64_t inc;
} PrRng;

static inline uint32_t pr_rng_next(PrRng *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

/* pcg32_srandom_r(seed, stream) */
static inline void pr_rng_seed(PrRng *r, uint64_t seed, uint64_t stream) {
    r->state = 0u;
    r->inc = (stream << 1u) | 1u;
    pr_rng_next(r);
    r->state += seed;
    pr_rng_next(r);
}

/* Uniform integer in [0, bound), unbiased (rejection, pcg32_boundedrand_r). */
static inline uint32_t pr_rng_below(PrRng *r, uint32_t bound) {
    if (bound <= 1) return 0;
    uint32_t threshold = (uint32_t)(-bound) % bound;
    for (;;) {
        uint32_t v = pr_rng_next(r);
        if (v >= threshold) return v % bound;
    }
}

/* ------------------------------------------------------------------ FNV-1a 64 */

#define PR_FNV_OFFSET 0xcbf29ce484222325ULL
#define PR_FNV_PRIME 0x100000001b3ULL

static inline uint64_t pr_fnv1a(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= PR_FNV_PRIME;
    }
    return h;
}

#endif /* PR_MATH_H */
