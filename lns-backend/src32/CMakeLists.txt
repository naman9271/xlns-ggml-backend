ggml_add_backend_library(ggml-lns32
    ggml-lns32.cpp
    lns32-ops.cpp
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-lns-common/lns-common.cmake)
lns_backend_common_setup(ggml-lns32)

# xlnscpp uses a header-inclusion model: #include "xlns32.cpp" with defines.
# xlns32_alt enables the reduced-branching sb/db addition path.
# xlns32_ideal is intentionally NOT set (interpolation/cotransformation tables,
# consistent with a realistic hardware target — same stance as the xlns16
# backend's no-xlns16_ideal choice).
# NOTE: xlns32 has no xlns32_table analogue — float<->xlns32 conversion goes
# through libm log()/pow() per element, so this backend is much slower than
# the xlns16 one (see README; weight caching is documented future work).
target_compile_definitions(ggml-lns32 PRIVATE xlns32_alt)
