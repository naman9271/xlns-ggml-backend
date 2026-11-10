# Build settings shared by the LNS backends; each backend's CMakeLists.txt
# include()s this file from the ggml-lns-common directory next to it.

set(LNS_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}")

# xlnscpp uses a header-inclusion model (#include "xlns16.cpp" after the
# configuration defines), so it is an include directory, not a library.
# The default matches this repository's layout (build/ggml, build/llama.cpp).
if (NOT DEFINED XLNSCPP_DIR)
    set(XLNSCPP_DIR "${CMAKE_SOURCE_DIR}/../../xlnscpp")
endif()
if (NOT EXISTS "${XLNSCPP_DIR}/xlns16.cpp")
    message(FATAL_ERROR "LNS backends: XLNSCPP_DIR='${XLNSCPP_DIR}' is not an xlnscpp checkout; "
                        "pass -DXLNSCPP_DIR=/path/to/xlnscpp (run 'git submodule update --init' first)")
endif()

# Adds the shared LNS headers and xlnscpp to a backend target.
function(lns_backend_common_setup target)
    target_include_directories(${target} PRIVATE "${LNS_COMMON_DIR}" "${XLNSCPP_DIR}")
endfunction()

# xlns16 summation mode, shared by LNS16 and LNSCUDA16 (declared here so that
# whichever backend configures first sees it): base | monte | lpvip | sum32.
set(GGML_LNS16_SUM "base" CACHE STRING "xlns16 summation variant: base|monte|lpvip|sum32")
set_property(CACHE GGML_LNS16_SUM PROPERTY STRINGS base monte lpvip sum32)
option(GGML_LNS16_MONTE_BYLPVIP "LNS16/LNSCUDA16: implement monte adds via xlns32_add_lpvip" OFF)
if (NOT GGML_LNS16_SUM MATCHES "^(base|monte|lpvip|sum32)$")
    message(FATAL_ERROR "GGML_LNS16_SUM must be one of: base, monte, lpvip, sum32 (got '${GGML_LNS16_SUM}')")
endif()

# Compile definitions of the mode; kind is CPU or CUDA (no GPU sum32: base).
function(lns16_sum_mode_defines out_var kind)
    set(mode "${GGML_LNS16_SUM}")
    if (kind STREQUAL "CUDA" AND mode STREQUAL "sum32")
        message(WARNING "GGML_LNS16_SUM=sum32 has no CUDA implementation; LNSCUDA16 runs base in this build")
        set(mode "base")
    endif()
    set(defs "")
    if (mode STREQUAL "monte")
        list(APPEND defs GGML_LNS16_SUM_MONTE)
        if (GGML_LNS16_MONTE_BYLPVIP)
            list(APPEND defs GGML_LNS16_MONTE_BYLPVIP)
        endif()
    elseif (mode STREQUAL "lpvip")
        list(APPEND defs GGML_LNS16_SUM_LPVIP)
    elseif (mode STREQUAL "sum32")
        list(APPEND defs GGML_LNS16_SUM_SUM32 xlns32_alt)
    endif()
    set(${out_var} "${defs}" PARENT_SCOPE)
endfunction()
