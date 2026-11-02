// Standalone checks for the sum32 (pure-xlns32 accumulator) reductions.
//
// These invariants cannot be reached through test-lns-backend: that binary
// links only ggml, its MUL_MAT case is K=2 and its MUL_MAT_ID cases are K<=8,
// so nothing there can see an accumulator at work.
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -I xlnscpp -I lns-backend/src \
//       lns-backend/tests/test-lns16-sum32.cpp -o /tmp/test-lns16-sum32 && /tmp/test-lns16-sum32
//
// Exit code 0 = all checks pass.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#define xlns16_table
#define xlns16_alt
#include "xlns16.cpp"
#include "lns16-sum32.h"   // bootstraps xlns32.cpp with xlns32_alt

// Deterministic LCG (no rand(): the result must not depend on libc).
static uint64_t g_rng = 0x2545F4914F6CDD1DULL;
static inline double rnd_uniform() {         // [0,1)
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((g_rng >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
}

static int g_fail = 0;
static void report(const char * name, bool ok) {
    printf("%-46s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) g_fail++;
}

// Correctly-rounded xlns16 code for a real value (round-to-nearest in the log
// domain) - the yardstick both accumulators are measured against.
static xlns16 ideal_xlns16(double v) {
    if (v == 0.0) return xlns16_zero;
    const double l = log2(fabs(v)) * 128.0;
    long code = lround(l) + 0x4000;
    if (code < 0)      code = 0;
    if (code > 0x7FFF) code = 0x7FFF;
    return (xlns16)((v < 0 ? 0x8000 : 0) | (unsigned)code);
}

// Distance between two xlns16 values in log-field ULPs (1 ULP = 2^(1/128)).
static int ulp_diff(xlns16 a, xlns16 b) {
    if (xlns16_sign(a) != xlns16_sign(b)) return 0x7FFF;
    return abs((int)(a & xlns16_logmask) - (int)(b & xlns16_logmask));
}

// ------------------------------------------------------------------
// 1. narrow(widen(x)) == x for every xlns16 code
// ------------------------------------------------------------------
static void test_roundtrip() {
    bool ok = true;
    for (int i = 0; i < 65536; i++) {
        const xlns16 x = (xlns16)i;
        const xlns16 y = xlns16_narrow_sum32(xlns16_widen_sum32(x));
        if (x != y) {
            if (ok) printf("  first mismatch: 0x%04X -> 0x%04X\n", x, y);
            ok = false;
        }
    }
    report("roundtrip narrow(widen(x)) over all 65536", ok);
}

// ------------------------------------------------------------------
// 2. Two-term sums: sum32 should be at least as close to the correctly
//    rounded result as the 16-bit table adder.
// ------------------------------------------------------------------
static void test_two_term() {
    const int N = 200000;
    int worst_base = 0, worst_sum32 = 0;
    long long tot_base = 0, tot_sum32 = 0, counted = 0;

    for (int i = 0; i < N; i++) {
        // magnitudes spanning +-20 binades, both sign combinations,
        // including near-cancellation (ratio close to 1, opposite signs).
        const double ea = (rnd_uniform() * 40.0) - 20.0;
        const double eb = ea + (rnd_uniform() * 40.0) - 20.0;
        double va = pow(2.0, ea);
        double vb = pow(2.0, eb);
        if (rnd_uniform() < 0.5) va = -va;
        if (rnd_uniform() < 0.5) vb = -vb;

        const xlns16 a = fp2xlns16((float)va);
        const xlns16 b = fp2xlns16((float)vb);
        const double exact = (double)xlns162fp(a) + (double)xlns162fp(b);
        if (exact == 0.0) continue;                 // exact cancellation

        const xlns16 want  = ideal_xlns16(exact);
        const xlns16 pair[2] = { a, b };

        const xlns16 got_base  = xlns16_add(a, b);
        const xlns16 got_sum32 = xlns16_sum_sum32(pair, 2);

        const int db = ulp_diff(got_base,  want);
        const int ds = ulp_diff(got_sum32, want);
        if (db > worst_base)  worst_base  = db;
        if (ds > worst_sum32) worst_sum32 = ds;
        tot_base += db; tot_sum32 += ds; counted++;
    }

    printf("  two-term vs correctly-rounded (n=%lld): base max %d ULP (mean %.3f), "
           "sum32 max %d ULP (mean %.3f)\n",
           counted, worst_base, (double)tot_base / (double)counted,
           worst_sum32, (double)tot_sum32 / (double)counted);
    report("two-term sum32 no worse than xlns16_add", worst_sum32 <= worst_base &&
                                                      tot_sum32  <= tot_base);
}

// ------------------------------------------------------------------
// 3. Accumulator stagnation: sum of N copies of 1.0.
//    base saturates at ~370 (half-ULP absorption at F=7); sum32 must not.
// ------------------------------------------------------------------
static void test_stagnation() {
    const size_t sizes[] = { 128, 370, 512, 1536, 8960, 65536 };
    bool ok = true;
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); k++) {
        const size_t n = sizes[k];
        std::vector<xlns16> v(n, (xlns16)xlns16_one);
        const double base  = xlns162fp(xlns16_sum(v.data(), n));
        const double sum32 = xlns162fp(xlns16_sum_sum32(v.data(), n));
        const double rel_b = fabs(base  - (double)n) / (double)n;
        const double rel_s = fabs(sum32 - (double)n) / (double)n;
        printf("  N=%-6zu exact %-8zu base %12.4f (%6.2f%%)  sum32 %12.4f (%5.2f%%)\n",
               n, n, base, rel_b * 100.0, sum32, rel_s * 100.0);
        if (rel_s > 0.01) ok = false;               // 1 xlns16 ULP is 0.54%
    }
    report("sum32 does not stagnate (rel err < 1%)", ok);
}

// ------------------------------------------------------------------
// 4. Long mixed-sign dot products - the mat-mul case.
// ------------------------------------------------------------------
static void test_long_dot() {
    const size_t Ks[] = { 1536, 8960 };
    bool ok = true;
    for (size_t k = 0; k < 2; k++) {
        const size_t K = Ks[k];
        const int trials = 64;
        double sum_rel_b = 0.0, sum_rel_s = 0.0, max_rel_s = 0.0;
        int wins = 0;
        for (int t = 0; t < trials; t++) {
            std::vector<xlns16> a(K), b(K);
            double exact = 0.0;
            for (size_t i = 0; i < K; i++) {
                const float fa = (float)(rnd_uniform() * 2.0 - 1.0);
                const float fb = (float)(rnd_uniform() * 2.0 - 1.0);
                a[i] = fp2xlns16(fa);
                b[i] = fp2xlns16(fb);
                // reference over the xlns16-grid operands: isolates the
                // accumulator from the 16-bit operand rounding.
                exact += (double)xlns162fp(a[i]) * (double)xlns162fp(b[i]);
            }
            const double got_b = xlns162fp(xlns16_vec_dot(a.data(), b.data(), K));
            const double got_s = xlns162fp(xlns16_vec_dot_sum32(a.data(), b.data(), K));
            const double rel_b = fabs(got_b - exact) / fabs(exact);
            const double rel_s = fabs(got_s - exact) / fabs(exact);
            sum_rel_b += rel_b; sum_rel_s += rel_s;
            if (rel_s > max_rel_s) max_rel_s = rel_s;
            if (rel_s < rel_b) wins++;
        }
        printf("  K=%-5zu mean rel err: base %7.3f%%  sum32 %6.3f%%  (sum32 better in %d/%d)\n",
               K, sum_rel_b / trials * 100.0, sum_rel_s / trials * 100.0, wins, trials);
        if (!(sum_rel_s * 4.0 < sum_rel_b)) ok = false;   // expect a large margin
        if (max_rel_s > 0.02) ok = false;
    }
    report("sum32 beats base on long mixed-sign dots", ok);
}

// ------------------------------------------------------------------
// 5. RMS norm - the same-sign reduction (sum of squares).
// ------------------------------------------------------------------
static void test_rms_norm() {
    const size_t sizes[] = { 576, 1536 };
    bool ok = true;
    for (size_t k = 0; k < 2; k++) {
        const size_t n = sizes[k];
        std::vector<xlns16> x(n), db(n), ds(n);
        double sq = 0.0;
        for (size_t i = 0; i < n; i++) {
            x[i] = fp2xlns16((float)(rnd_uniform() * 2.0 - 1.0));
            const double v = xlns162fp(x[i]);
            sq += v * v;
        }
        const double inv_rms = 1.0 / sqrt(sq / (double)n + 1e-6);
        const xlns16 eps = fp2xlns16(1e-6f);
        xlns16_rms_norm(x.data(), db.data(), n, eps);
        xlns16_rms_norm_sum32(x.data(), ds.data(), n, eps);

        double max_b = 0.0, max_s = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double want = (double)xlns162fp(x[i]) * inv_rms;
            const double eb = fabs(xlns162fp(db[i]) - want) / fmax(fabs(want), 1e-6);
            const double es = fabs(xlns162fp(ds[i]) - want) / fmax(fabs(want), 1e-6);
            if (eb > max_b) max_b = eb;
            if (es > max_s) max_s = es;
        }
        printf("  n=%-5zu max rel err: base %6.3f%%  sum32 %5.3f%%\n",
               n, max_b * 100.0, max_s * 100.0);
        if (!(max_s < max_b)) ok = false;
        if (max_s > 0.02) ok = false;
    }
    report("sum32 rms_norm beats base", ok);
}

int main() {
    printf("sum32 standalone checks (xlns16_table + xlns16_alt, xlns32_alt)\n\n");
    test_roundtrip();
    test_two_term();
    test_stagnation();
    test_long_dot();
    test_rms_norm();
    printf("\n%s\n", g_fail == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fail == 0 ? 0 : 1;
}
