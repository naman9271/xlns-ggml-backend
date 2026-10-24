// LNSCUDA16 / LNSCUDA32 backends: LNS arithmetic on a CUDA GPU.
//
// Both devices are the same backend (lns-cuda-context + lns-cuda-ops) over a
// different kernel_set, i.e. LNS format. Everything ggml-facing comes from
// the shared framework (lns-backend-framework.h); this file only names the
// two devices.

#include "ggml-lns-cuda.h"

#include "lns-backend-framework.h"
#include "lns-cuda-context.h"
#include "lns-cuda-ops.h"

#include <cuda_runtime_api.h>

#include <string>

namespace {

// What distinguishes the two devices.
struct xlns16_format {
    static constexpr const char * name    = "LNSCUDA16";
    // LNSCUDA16 reproduces LNS16's base summation; test-lns-backend checks
    // tolerance profiles against the "sum=" part of the description.
    static constexpr const char * summary = "xlns16, sum=base";
    static constexpr const char * audit   = "\"sum_mode\":\"base\",";
    static const lns_cuda::kernel_set & kernels() { return lns_cuda::xlns16_kernels(); }
    static ggml_guid_t guid() {
        // ASCII "LNSCUDA16-bkend-"
        static ggml_guid guid = { 0x4c, 0x4e, 0x53, 0x43, 0x55, 0x44, 0x41, 0x31,
                                  0x36, 0x2d, 0x62, 0x6b, 0x65, 0x6e, 0x64, 0x2d };
        return &guid;
    }
};

struct xlns32_format {
    static constexpr const char * name    = "LNSCUDA32";
    static constexpr const char * summary = "xlns32";
    static constexpr const char * audit   = "";
    static const lns_cuda::kernel_set & kernels() { return lns_cuda::xlns32_kernels(); }
    static ggml_guid_t guid() {
        // ASCII "LNSCUDA32-bkend-"
        static ggml_guid guid = { 0x4c, 0x4e, 0x53, 0x43, 0x55, 0x44, 0x41, 0x33,
                                  0x32, 0x2d, 0x62, 0x6b, 0x65, 0x6e, 0x64, 0x2d };
        return &guid;
    }
};

template <class Format>
struct lns_cuda_traits : lns::backend_traits_base {
    using context = lns_cuda::context;

    static constexpr const char * name = Format::name;

    static const char * description() {
        static const std::string desc = []() {
            std::string d = std::string("Logarithmic Number System on CUDA (") + Format::summary + ")";
            cudaDeviceProp prop;
            const int device = lns_cuda::selected_device();
            if (device >= 0 && cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
                d += std::string(" on ") + prop.name;
            }
            return d;
        }();
        return desc.c_str();
    }

    static ggml_guid_t guid() { return Format::guid(); }

    static lns::op_table<context> kernels() { return lns_cuda::op_table(); }

    static context * create_context() { return new context(Format::kernels(), Format::name); }

    static size_t device_count() { return lns_cuda::selected_device() >= 0 ? 1 : 0; }

    static const char * audit_fields() { return Format::audit; }
};

using lns_cuda16_backend = lns::host_backend<lns_cuda_traits<xlns16_format>>;
using lns_cuda32_backend = lns::host_backend<lns_cuda_traits<xlns32_format>>;

} // namespace

ggml_backend_t ggml_backend_lns_cuda16_init(void) {
    return lns_cuda16_backend::init();
}

ggml_backend_t ggml_backend_lns_cuda32_init(void) {
    return lns_cuda32_backend::init();
}

bool ggml_backend_is_lns_cuda(ggml_backend_t backend) {
    return lns_cuda16_backend::is(backend) || lns_cuda32_backend::is(backend);
}

ggml_backend_reg_t ggml_backend_lns_cuda16_reg(void) {
    return lns_cuda16_backend::reg();
}

ggml_backend_reg_t ggml_backend_lns_cuda32_reg(void) {
    return lns_cuda32_backend::reg();
}
