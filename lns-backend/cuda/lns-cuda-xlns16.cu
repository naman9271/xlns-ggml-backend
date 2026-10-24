// xlns16 kernels for the LNSCUDA16 device.
//
// Configured exactly like the LNS16 CPU backend (xlns16_table + xlns16_alt,
// no xlns16_ideal), with xlnscuda's xlns16d_table mirroring the host tables
// on the device. Every conversion, addition and lookup therefore returns the
// same bits on both sides; xlnscuda/tests/chkxlns16d_table.cu checks that.

// System headers first: the library includes below sit in a namespace, where
// their own nested includes of these headers must be no-ops.
#include <iostream>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "lns-cuda-kernels.h"

// The xlnscpp/xlnscuda sources carry non-inline definitions and their own
// globals, so each may appear in one translation unit per link unit; the
// anonymous namespace keeps this copy private to this file (the LNS16 CPU
// backend has another one).
namespace {

// Both libraries declare xlns16_float / xlns16d_float overloads of these libm
// names in this namespace, which would hide the global ones from the
// libraries' own calls. Import the global overload sets.
using ::abs; using ::atan; using ::cos; using ::exp; using ::fabs; using ::log;
using ::pow; using ::sin; using ::sqrt; using ::tanh;

#define xlns16_table
#define xlns16_alt
#define xlns16d_table
#include "xlns16.cpp"
#include "xlns16d.cu"

struct xlns16_arith {
    using value = xlns16;
    static constexpr const char * name = "xlns16";
    static constexpr value zero    = xlns16_zero;
    static constexpr value neg_inf = xlns16_neg_inf;

    static cudaError_t upload_tables() { return xlns16d_table_init(); }

    __device__ static value from_f32(float x)        { return fp2xlns16d(x); }
    __device__ static float to_f32(value v)          { return xlns16d2fp(v); }
    __device__ static value add(value a, value b)    { return xlns16d_add(a, b); }
    __device__ static value sub(value a, value b)    { return xlns16d_sub(a, b); }
    __device__ static value mul(value a, value b)    { return xlns16d_mul(a, b); }
    __device__ static value div(value a, value b)    { return xlns16d_div(a, b); }
    __device__ static bool  gt(value a, value b)     { return xlns16d_gt(a, b); }
    __device__ static bool  is_zero(value v)         { return xlns16d_is_zero(v); }
    __device__ static value exp(value v)             { return xlns16d_exp(v); }
    __device__ static value sqrt(value v)            { return xlns16_sqrt(v); }
    __device__ static value recip(value v)           { return xlns16_recip(v); }
    __device__ static value silu(value v)            { return xlns16d_silu(v); }
    __device__ static value gelu(value v)            { return xlns16d_gelu(v); }
    __device__ static value relu(value v)            { return xlns16d_relu(v); }
    __device__ static value tanh(value v)            { return xlns16d_tanh(v); }

    // ggml marks masked-out positions with a huge negative float rather than
    // xlns16's own -inf encoding (lns-ops.cpp: f32_mask_to_lns16).
    __device__ static value mask_from_f32(float m)   { return m <= -3.0e38f ? neg_inf : from_f32(m); }
};

} // namespace

#include "lns-cuda-kernels.cuh"

const lns_cuda::kernel_set & lns_cuda::xlns16_kernels() {
    static const detail::kernel_set_impl<xlns16_arith> kernels{};
    return kernels;
}
