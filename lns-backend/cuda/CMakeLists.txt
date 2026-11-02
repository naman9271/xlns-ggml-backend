# LNSCUDA backend: LNS arithmetic on CUDA GPUs through xlnscuda.
#
# Needs only the CUDA toolkit (nvcc + cudart), not ggml's own CUDA backend.
# Useful cache variables:
#   XLNSCUDA_DIR               xlnscuda checkout (default: next to xlnscpp)
#   CMAKE_CUDA_ARCHITECTURES   default "native" (the GPUs in this machine)
#   CMAKE_CUDA_HOST_COMPILER   a host compiler your nvcc accepts, e.g. g++-12
#                              for CUDA 12.0 (or set CUDAHOSTCXX)

if (GGML_BACKEND_DL)
    # GGML_BACKEND_DL_IMPL exports one registry per library; this one has two.
    message(FATAL_ERROR "GGML_LNSCUDA does not support GGML_BACKEND_DL=ON")
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
    endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if (NOT CMAKE_CUDA_COMPILER)
    # check_language caches the failure; drop it so a fixed environment
    # (e.g. a supported host compiler) is picked up on the next configure.
    unset(CMAKE_CUDA_COMPILER CACHE)
    message(FATAL_ERROR "GGML_LNSCUDA=ON but nvcc is missing or rejects the host compiler "
                        "'${CMAKE_CUDA_HOST_COMPILER}'. Install the CUDA toolkit, or pass a host "
                        "compiler it supports: -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 (or set CUDAHOSTCXX)")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

ggml_add_backend_library(ggml-lnscuda
    ggml-lns-cuda.cpp
    lns-cuda-context.cpp
    lns-cuda-ops.cpp
    lns-cuda-xlns16.cu
    lns-cuda-xlns32.cu
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-lns-common/lns-common.cmake)
lns_backend_common_setup(ggml-lnscuda)

if (NOT DEFINED XLNSCUDA_DIR)
    set(XLNSCUDA_DIR "${XLNSCPP_DIR}/../xlnscuda")
endif()
if (NOT EXISTS "${XLNSCUDA_DIR}/xlns16d.cu")
    message(FATAL_ERROR "GGML_LNSCUDA: XLNSCUDA_DIR='${XLNSCUDA_DIR}' is not an xlnscuda checkout; "
                        "pass -DXLNSCUDA_DIR=/path/to/xlnscuda (run 'git submodule update --init' first)")
endif()
# The backend relies on the device lookup tables added to xlnscuda for exact
# CPU parity (xlns16d_table_init / xlns32d_table_init); fail at configure time,
# not with a wall of compile errors.
file(READ "${XLNSCUDA_DIR}/xlns16d.cu" LNS_XLNS16D_SOURCE)
if (NOT LNS_XLNS16D_SOURCE MATCHES "xlns16d_table_init")
    message(FATAL_ERROR "GGML_LNSCUDA: ${XLNSCUDA_DIR} predates device table support "
                        "(xlns16d_table_init); update the xlnscuda submodule")
endif()
target_include_directories(ggml-lnscuda PRIVATE "${XLNSCUDA_DIR}")

# Static cudart: binaries built with LNSCUDA still start on machines without
# the CUDA toolkit (the LNSCUDA devices then report no GPU and stay hidden).
target_link_libraries(ggml-lnscuda PRIVATE CUDA::cudart_static)

# The xlnscpp/xlnscuda sources are written as C and trip many warnings that
# are not ours to fix; keep nvcc quiet about them.
target_compile_options(ggml-lnscuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=declared_but_not_referenced>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=set_but_not_used>)
