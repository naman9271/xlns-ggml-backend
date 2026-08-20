// xlns16 reductions with a pure xlns32 accumulator ("sum32").
//
// Long xlns16 summations stagnate: xlns16_add(S,t) returns S unchanged once
// t/S drops below half an ULP (2^-8.53 at F=7), so a sequential sum of equal
// terms saturates after ~370 of them.  xlnscpp/summation16bit.MD names summing
// through an xlns32 accumulator as the way to avoid this ("but that is very
// expensive"), and ships the cheaper approximations instead (_lpvip32, _monte).
// This file provides the expensive-but-honest version: the accuracy ceiling
// the 16-bit schemes can be measured against.
//
// Each xlns16 term is widened to xlns32 (exact — the formats are bit-aligned),
// accumulated with xlns32_add, and the final accumulator is rounded back to
// xlns16.  Everything outside the reduction (products, scale, mask, max
// subtraction, exp, sqrt, recip, division) stays in xlns16, exactly as in the
// _lpvip32 and _monte families.
//
// INVARIANT: xlns32_add is the ONLY xlns32 arithmetic in this file.  Operands,
// results and storage are xlns16 throughout; the accumulator is the single
// widened value, and it is narrowed exactly once per reduction.
//
// Include order: xlns16.cpp first (mandatory).  If xlns32.cpp has not been
// included yet, this file bootstraps it with xlns32_alt (the same configuration
// the LNS32 backend uses: table + linear-interpolation sb, cotransformation db;
// xlns32_ideal deliberately NOT defined).  If xlns32lpvip.cpp shares the
// translation unit, include xlns32.cpp yourself with xlns32_alt and
// xlns32_aicasb before both files — xlns32lpvip.cpp only defines xlns32_aicasb
// inside its own bootstrap, and that bootstrap forces xlns32_ideal.
//
// Thread safety: the non-ideal xlns32_sb is a macro that writes the globals
// xlns32_z / xlns32_zh.  Safe while the caller is single-threaded; those
// globals must become thread_local if the reductions are ever parallelized.

#ifndef xlns16_sum32
#define xlns16_sum32 1

#ifndef xlns16_zero
#error "lns16-sum32.h: include xlns16.cpp first (it defines xlns16_zero)"
#endif

#ifndef xlns32_zero
  #ifndef xlns32_alt
  #define xlns32_alt
  #endif
  #include "xlns32.cpp"
  // If the includer defined xlns32_ideal, the accumulator becomes the libm
  // (log/pow) adder instead of the table path.  That is a different, more
  // expensive experiment — deliberately allowed, but it is not what the
  // GGML_LNS16_SUM=sum32 backend builds.
#endif

// ------------------------------------------------------------------
// xlns16 <-> xlns32 conversion
// ------------------------------------------------------------------
// The two formats are bit-aligned: sign bit 15 -> 31, the log offset
// 0x4000 -> 0x40000000, and the 7 fraction bits land in bits 22..16 of the
// 23-bit xlns32 fraction.  Widening is therefore exact for every code,
// including xlns16_zero (0x0000), signed zero (0x8000) and the saturated
// values 0x7FFF / 0xFFFF.
static inline xlns32 xlns16_widen_sum32(xlns16 x) {
    return ((xlns32)x) << 16;
}

// Narrow with round-to-nearest on the log field ("round-half-up": a tie rounds
// toward the larger magnitude; ties are rare, they need the sb/db table to
// leave the low 16 bits at exactly 0x8000).  The _lpvip32 family truncates
// (sum>>16) instead, which is biased low by up to one xlns16 ULP.
//
//   * for magnitudes below 0x7FFF8000 the +0x8000 cannot carry into bit 31,
//     so the sign survives;
//   * at or above it the result saturates to +/- the largest xlns16 code,
//     which is what xlns16 arithmetic produces on overflow;
//   * +/-0 map to 0x0000 / 0x8000 (xlns16 reads both as zero via xlns16_abs),
//     and magnitudes below code 0x8000 round to xlns16 zero -- the nearest
//     representable value.
// narrow(widen(x)) == x for all 65536 xlns16 codes.
static inline xlns16 xlns16_narrow_sum32(xlns32 s) {
    if ((s & xlns32_logmask) >= (xlns32)0x7FFF8000u) {
        return (xlns16)((s >> 16) | xlns16_logmask);
    }
    return (xlns16)((s + 0x8000u) >> 16);
}

// ------------------------------------------------------------------
// Reductions
// ------------------------------------------------------------------

// Sum of array elements: result = sum a[i]
inline xlns16 xlns16_sum_sum32(const xlns16 *a, size_t n) {
    if (n == 0) return xlns16_zero;
    xlns32 sum = xlns16_widen_sum32(a[0]);
    for (size_t i = 1; i < n; i++) {
        sum = xlns32_add(sum, xlns16_widen_sum32(a[i]));
    }
    return xlns16_narrow_sum32(sum);
}

// Vector dot product: result = sum(a[i] * b[i]).  The products are xlns16
// (an integer add of the log fields) and widen exactly, so only the
// accumulation differs from xlns16_vec_dot.
inline xlns16 xlns16_vec_dot_sum32(const xlns16 *a, const xlns16 *b, size_t n) {
    if (n == 0) return xlns16_zero;
    xlns32 sum = xlns16_widen_sum32(xlns16_mul(a[0], b[0]));
    for (size_t i = 1; i < n; i++) {
        sum = xlns32_add(sum, xlns16_widen_sum32(xlns16_mul(a[i], b[i])));
    }
    return xlns16_narrow_sum32(sum);
}

// Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
// Identical to xlns16_layernorm except that both reductions accumulate in
// xlns32.  Unused by the ggml backend (no GGML_OP_NORM yet).
inline void xlns16_layernorm_sum32(const xlns16 *x, xlns16 *out,
                       const xlns16 *gamma, const xlns16 *beta,
                       size_t n, float eps) {
    // compute mean
    xlns16 mean = xlns16_sum_sum32(x, n);
    mean = xlns16_div(mean, fp2xlns16((float)n));
    // compute variance
    xlns32 var32 = xlns32_zero;
    for (size_t i = 0; i < n; i++) {
        xlns16 diff = xlns16_sub(x[i], mean);
        var32 = xlns32_add(var32, xlns16_widen_sum32(xlns16_mul(diff, diff)));
    }
    xlns16 var = xlns16_div(xlns16_narrow_sum32(var32), fp2xlns16((float)n));
    // normalize (xlns16_layernorm also takes inv_std through float)
    xlns16 inv_std = fp2xlns16(1.0f / sqrt(xlns162fp(var) + eps));
    for (size_t i = 0; i < n; i++) {
        out[i] = xlns16_mul(xlns16_sub(x[i], mean), inv_std);
        if (gamma) out[i] = xlns16_mul(out[i], gamma[i]);
        if (beta)  out[i] = xlns16_add(out[i], beta[i]);
    }
}

// RMSNorm: dst[i] = x[i] / sqrt(mean(x^2) + eps).
// Identical to xlns16_rms_norm except that the sum of squares accumulates in
// xlns32; the 1/n scale, the eps add, the sqrt and the reciprocal are all
// xlns16, so the only xlns32 arithmetic is the reduction itself.
// (Scaling by 1/n in xlns32 before narrowing was measured and is a wash --
// bit-identical for power-of-two n and it goes either way otherwise -- while
// costing a libm log() per call, since fp2xlns32 has no lookup table.)
// dst may alias x (in-place).
inline void xlns16_rms_norm_sum32(const xlns16 *x, xlns16 *dst, size_t n, xlns16 eps) {
    if (n == 0) return;
    xlns32 sum_sq = xlns32_zero;
    for (size_t i = 0; i < n; i++)
        sum_sq = xlns32_add(sum_sq, xlns16_widen_sum32(xlns16_square(x[i])));
    xlns16 mean = xlns16_mul(xlns16_narrow_sum32(sum_sq), fp2xlns16(1.0f / (float)n));
    xlns16 inv_rms = xlns16_recip(xlns16_sqrt(xlns16_add(mean, eps)));
    for (size_t i = 0; i < n; i++)
        dst[i] = xlns16_mul(x[i], inv_rms);
}

// Softmax: exp(scale*a[i] - max) / sum(exp(scale*a[j] - max)).
// Same control flow as xlns16_softmax.  Scale, mask bias, and max-sub use
// plain xlns16 ops; only the normalization sum uses xlns16_sum_sum32.
// a[i] == xlns16_neg_inf is treated as already-excluded and skips the scale
// multiply.  c may alias a (in-place).
inline void xlns16_softmax_sum32(const xlns16 *a, xlns16 *c, size_t n,
                                  xlns16 scale = xlns16_one) {
    if (n == 0) return;
    xlns16 maxval = xlns16_neg_inf;
    for (size_t i = 0; i < n; i++) {
        xlns16 v = a[i];
        if (v != xlns16_neg_inf) v = xlns16_mul(v, scale);
        c[i] = v;
        if (xlns16_gt(c[i], maxval)) maxval = c[i];
    }
    for (size_t i = 0; i < n; i++)
        c[i] = xlns16_exp(xlns16_sub(c[i], maxval));
    xlns16 total = xlns16_sum_sum32(c, n);
    for (size_t i = 0; i < n; i++)
        c[i] = xlns16_div(c[i], total);
}

// Masked variant of xlns16_softmax_sum32. mask[i] is a pre-converted xlns16
// value: xlns16_neg_inf marks a masked-out position, xlns16_zero means "no
// mask", anything else is an additive bias (e.g. ALiBi).
inline void xlns16_softmax_masked_sum32(const xlns16 *a, const xlns16 *mask,
                                         xlns16 *c, size_t n,
                                         xlns16 scale = xlns16_one) {
    if (n == 0) return;
    xlns16 maxval = xlns16_neg_inf;
    for (size_t i = 0; i < n; i++) {
        xlns16 v = a[i];
        if (v != xlns16_neg_inf) {
            v = xlns16_mul(v, scale);
            if (mask[i] == xlns16_neg_inf) v = xlns16_neg_inf;
            else if (!xlns16_is_zero(mask[i])) v = xlns16_add(v, mask[i]);
        }
        c[i] = v;
        if (xlns16_gt(c[i], maxval)) maxval = c[i];
    }
    for (size_t i = 0; i < n; i++)
        c[i] = xlns16_exp(xlns16_sub(c[i], maxval));
    xlns16 total = xlns16_sum_sum32(c, n);
    for (size_t i = 0; i < n; i++)
        c[i] = xlns16_div(c[i], total);
}

#endif // xlns16_sum32
