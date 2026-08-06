ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
    lns-ops.cpp
)

# xlnscpp uses a header-inclusion model: #include "xlns16.cpp" with defines.
# xlns16_table enables the 65536-entry O(1) float<->xlns16 lookup table.
# xlns16_alt enables the reduced-branching sb/db addition path.
# xlns16_ideal is intentionally NOT set (table approximation, consistent with a realistic hardware target).
target_compile_definitions(ggml-lns PRIVATE xlns16_table xlns16_alt)

# Summation variant for softmax / rms_norm / mat-mul vec_dot (compile-time):
#   base  — plain sequential xlns16_add accumulation (xlns16.cpp)
#   monte — Monte-Carlo LNS dithered accumulation (xlns16monte*.cpp)
#   lpvip — 32-bit LPVIP accumulator, returned as xlns16 (xlns32lpvip.cpp)
set(GGML_LNS16_SUM "base" CACHE STRING "xlns16 summation variant: base|monte|lpvip")
set_property(CACHE GGML_LNS16_SUM PROPERTY STRINGS base monte lpvip)
if(GGML_LNS16_SUM STREQUAL "monte")
    target_compile_definitions(ggml-lns PRIVATE GGML_LNS16_SUM_MONTE)
elseif(GGML_LNS16_SUM STREQUAL "lpvip")
    target_compile_definitions(ggml-lns PRIVATE GGML_LNS16_SUM_LPVIP)
elseif(NOT GGML_LNS16_SUM STREQUAL "base")
    message(FATAL_ERROR "GGML_LNS16_SUM must be one of: base, monte, lpvip (got '${GGML_LNS16_SUM}')")
endif()

# Which Monte-Carlo implementation backs GGML_LNS16_SUM=monte. The two files
# define identical symbols and are mutually exclusive, so this must be a
# compile-time choice:
#   OFF — xlns16monte.cpp (pure 16-bit MCLNS with LFSR dither; default)
#   ON  — xlns16monte_bylpvip.cpp (same API via xlns32_add_lpvip)
option(GGML_LNS16_MONTE_BYLPVIP "ggml-lns: implement monte adds via xlns32_add_lpvip" OFF)
if(GGML_LNS16_MONTE_BYLPVIP)
    target_compile_definitions(ggml-lns PRIVATE GGML_LNS16_MONTE_BYLPVIP)
endif()

# Point to the xlnscpp submodule so #include "xlns16.cpp" resolves.
# When building inside ggml/llama.cpp, xlnscpp must be available at the path
# below, or you can override with -DXLNSCPP_DIR=/path/to/xlnscpp.
if(NOT DEFINED XLNSCPP_DIR)
    set(XLNSCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xlnscpp")
endif()
target_include_directories(ggml-lns PRIVATE ${XLNSCPP_DIR})
