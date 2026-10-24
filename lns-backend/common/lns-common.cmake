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
