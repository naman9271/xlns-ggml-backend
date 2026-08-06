#!/usr/bin/env bash
# integrate-lns.sh — apply LNS backend hooks to a ggml source tree via regex.
#
# Usage (sourced from setup.sh or validate-latest.sh):
#   integrate_lns_regex "$GGML_ROOT"          # standalone ggml (paths relative to root)
#   integrate_lns_regex "$LLAMA_ROOT/ggml"    # embedded ggml inside llama.cpp
#
# Integrates BOTH backends: LNS16 (option GGML_LNS, dir ggml-lns) and
# LNS32 (option GGML_LNS32, dir ggml-lns32). Each section is independently
# idempotent, so a tree that already has the LNS16 hooks still receives the
# LNS32 ones.
#
# Returns 0 on success, 1 on failure.

integrate_lns_regex() {
    local root="$1"
    local cmake_file="$root/CMakeLists.txt"
    local header_file="$root/include/ggml.h"
    local src_cmake="$root/src/CMakeLists.txt"
    local reg_file="$root/src/ggml-backend-reg.cpp"
    local ggml_c="$root/src/ggml.c"

    for f in "$cmake_file" "$header_file" "$src_cmake" "$reg_file" "$ggml_c"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: integrate-lns: missing file: $f" >&2
            return 1
        fi
    done

    if grep -q 'GGML_TYPE_LNS16' "$header_file"; then
        echo "    LNS16 already integrated in $root (GGML_TYPE_LNS16 present)"
    else

    # ── CMakeLists.txt: GGML_LNS option ──────────────────────────────────────
    if ! grep -q 'option(GGML_LNS ' "$cmake_file"; then
        perl -i -0pe '
            s/(option\(GGML_BLAS[^\n]+\n)/$1option(GGML_LNS                             "ggml: use LNS (Logarithmic Number System)"       OFF)\n/
                or die "integrate-lns: GGML_BLAS anchor not found in CMakeLists.txt\n";
        ' "$cmake_file" || return 1
    fi

    # ── ggml.h: dynamic enum slot before GGML_TYPE_COUNT ─────────────────────
    perl -i -pe '
        if (/^(\s+)GGML_TYPE_COUNT\s*=\s*(\d+)\s*,/) {
            my $indent = $1;
            my $n = $2;
            $_ = "${indent}GGML_TYPE_LNS16   = $n, // 16-bit Logarithmic Number System (xlns16)\n"
               . "${indent}GGML_TYPE_COUNT   = " . ($n + 1) . ",\n";
        }
    ' "$header_file"

    if ! grep -q 'GGML_TYPE_LNS16' "$header_file"; then
        echo "ERROR: integrate-lns: failed to insert GGML_TYPE_LNS16 in ggml.h" >&2
        return 1
    fi

    # ── src/CMakeLists.txt: ggml_add_backend(LNS) ────────────────────────────
    if ! grep -q 'ggml_add_backend(LNS)' "$src_cmake"; then
        perl -i -pe '
            BEGIN { $done = 0 }
            if (/^ggml_add_backend\(BLAS\)/ && !$done++) {
                $_ .= "ggml_add_backend(LNS)\n";
            }
        ' "$src_cmake" || return 1
    fi

    # ── ggml-backend-reg.cpp: include + registration ─────────────────────────
    if ! grep -q 'GGML_USE_LNS' "$reg_file"; then
        perl -i -0pe '
            s/(#ifdef GGML_USE_BLAS\n#include "ggml-blas.h"\n#endif\n)/
                $1
#ifdef GGML_USE_LNS
#include "ggml-lns.h"
#endif

/
                or die "integrate-lns: BLAS include anchor not found in ggml-backend-reg.cpp\n";

            s/(#ifdef GGML_USE_BLAS\n\s+register_backend\(ggml_backend_blas_reg\(\)\);\n#endif\n)/
                $1
#ifdef GGML_USE_LNS
        register_backend(ggml_backend_lns_reg());
#endif

/
                or die "integrate-lns: BLAS register anchor not found in ggml-backend-reg.cpp\n";
        ' "$reg_file" || return 1
    fi

    # ── ggml.c: LNS16 type traits after NVFP4 ────────────────────────────────
    if ! grep -q 'GGML_TYPE_LNS16' "$ggml_c"; then
        perl -i -0pe '
            s/(\[GGML_TYPE_NVFP4\] = \{.*?\},)\n(\s+\[GGML_TYPE_Q2_K\] = \{)/$1
    [GGML_TYPE_LNS16] = {
        .type_name                = "lns16",
        .blck_size                = 1,
        .type_size                = sizeof(uint16_t),
        .is_quantized             = false,
    },
$2/s
                or die "integrate-lns: NVFP4/Q2_K anchor not found in ggml.c\n";
        ' "$ggml_c" || return 1
    fi

    fi # end LNS16 section

    integrate_lns32_regex "$root" || return 1

    integrate_lns_validate "$root"
}

# LNS32 hooks. No ggml.h/ggml.c edits: the LNS32 backend defines no new ggml
# type (all tensors flow as F32), so only the build option, the backend
# subdirectory, and the registration are needed.
integrate_lns32_regex() {
    local root="$1"
    local cmake_file="$root/CMakeLists.txt"
    local src_cmake="$root/src/CMakeLists.txt"
    local reg_file="$root/src/ggml-backend-reg.cpp"

    # ── CMakeLists.txt: GGML_LNS32 option (after the GGML_LNS option) ────────
    if ! grep -q 'option(GGML_LNS32' "$cmake_file"; then
        perl -i -0pe '
            s/(option\(GGML_LNS \s*[^\n]+\n)/$1option(GGML_LNS32                           "ggml: use LNS32 (32-bit Logarithmic Number System)" OFF)\n/
                or die "integrate-lns: GGML_LNS option anchor not found in CMakeLists.txt\n";
        ' "$cmake_file" || return 1
    fi

    # ── src/CMakeLists.txt: ggml_add_backend(LNS32) ──────────────────────────
    if ! grep -q 'ggml_add_backend(LNS32)' "$src_cmake"; then
        perl -i -pe '
            BEGIN { $done = 0 }
            if (/^ggml_add_backend\(LNS\)/ && !$done++) {
                $_ .= "ggml_add_backend(LNS32)\n";
            }
        ' "$src_cmake" || return 1
        if ! grep -q 'ggml_add_backend(LNS32)' "$src_cmake"; then
            echo "ERROR: integrate-lns: ggml_add_backend(LNS) anchor not found in src/CMakeLists.txt" >&2
            return 1
        fi
    fi

    # ── ggml-backend-reg.cpp: include + registration ─────────────────────────
    if ! grep -q 'GGML_USE_LNS32' "$reg_file"; then
        perl -i -0pe '
            s/(#ifdef GGML_USE_LNS\n#include "ggml-lns.h"\n#endif\n)/
                $1
#ifdef GGML_USE_LNS32
#include "ggml-lns32.h"
#endif

/
                or die "integrate-lns: LNS include anchor not found in ggml-backend-reg.cpp\n";

            s/(#ifdef GGML_USE_LNS\n\s+register_backend\(ggml_backend_lns_reg\(\)\);\n#endif\n)/
                $1
#ifdef GGML_USE_LNS32
        register_backend(ggml_backend_lns32_reg());
#endif

/
                or die "integrate-lns: LNS register anchor not found in ggml-backend-reg.cpp\n";
        ' "$reg_file" || return 1
    fi
}

integrate_lns_validate() {
    local root="$1"
    local errors=0

    grep -q 'GGML_TYPE_LNS16' "$root/include/ggml.h" \
        || { echo "ERROR: validate: GGML_TYPE_LNS16 missing from ggml.h" >&2; errors=1; }
    grep -q 'option(GGML_LNS ' "$root/CMakeLists.txt" \
        || { echo "ERROR: validate: GGML_LNS option missing from CMakeLists.txt" >&2; errors=1; }
    grep -q 'option(GGML_LNS32' "$root/CMakeLists.txt" \
        || { echo "ERROR: validate: GGML_LNS32 option missing from CMakeLists.txt" >&2; errors=1; }
    grep -q 'ggml_add_backend(LNS)' "$root/src/CMakeLists.txt" \
        || { echo "ERROR: validate: ggml_add_backend(LNS) missing" >&2; errors=1; }
    grep -q 'ggml_add_backend(LNS32)' "$root/src/CMakeLists.txt" \
        || { echo "ERROR: validate: ggml_add_backend(LNS32) missing" >&2; errors=1; }
    grep -q 'GGML_USE_LNS' "$root/src/ggml-backend-reg.cpp" \
        || { echo "ERROR: validate: GGML_USE_LNS missing from ggml-backend-reg.cpp" >&2; errors=1; }
    grep -q 'GGML_USE_LNS32' "$root/src/ggml-backend-reg.cpp" \
        || { echo "ERROR: validate: GGML_USE_LNS32 missing from ggml-backend-reg.cpp" >&2; errors=1; }
    grep -q 'GGML_TYPE_LNS16' "$root/src/ggml.c" \
        || { echo "ERROR: validate: LNS16 traits missing from ggml.c" >&2; errors=1; }

    # Detect duplicate insertions
    local count
    count=$(grep -c 'GGML_TYPE_LNS16' "$root/include/ggml.h" || true)
    if [ "$count" -ne 1 ]; then
        echo "ERROR: validate: expected 1 GGML_TYPE_LNS16 in ggml.h, found $count" >&2
        errors=1
    fi
    # NOTE: 'ggml_add_backend(LNS)' does not match 'ggml_add_backend(LNS32)'
    # (the closing paren terminates the pattern), so both counts must be 1.
    count=$(grep -c 'ggml_add_backend(LNS)' "$root/src/CMakeLists.txt" || true)
    if [ "$count" -ne 1 ]; then
        echo "ERROR: validate: expected 1 ggml_add_backend(LNS), found $count" >&2
        errors=1
    fi
    count=$(grep -c 'ggml_add_backend(LNS32)' "$root/src/CMakeLists.txt" || true)
    if [ "$count" -ne 1 ]; then
        echo "ERROR: validate: expected 1 ggml_add_backend(LNS32), found $count" >&2
        errors=1
    fi
    count=$(grep -c 'option(GGML_LNS32' "$root/CMakeLists.txt" || true)
    if [ "$count" -ne 1 ]; then
        echo "ERROR: validate: expected 1 option(GGML_LNS32, found $count" >&2
        errors=1
    fi

    return "$errors"
}

integrate_lns_or_fallback() {
    local repo_dir="$1"       # ggml or llama.cpp checkout root
    local patch_file="$2"
    local base_commit="$3"
    local ggml_root

    if [ -f "$repo_dir/ggml/include/ggml.h" ]; then
        ggml_root="$repo_dir/ggml"
    else
        ggml_root="$repo_dir"
    fi

    if integrate_lns_regex "$ggml_root"; then
        echo "    Regex integration succeeded"
        return 0
    fi

    echo "    Regex integration failed; using patch fallback"
    cd "$repo_dir"
    git reset --hard "$base_commit"
    if ! git apply "$patch_file"; then
        echo "ERROR: patch fallback also failed" >&2
        return 1
    fi
    # The pinned patch only carries the LNS16 hooks; apply the LNS32 ones on top.
    integrate_lns32_regex "$ggml_root" || return 1
    integrate_lns_validate "$ggml_root"
}
