#pragma once

// Runtime configuration shared by every LNS backend:
//
//   GGML_LNS_BACKEND  which LNS backends claim ops (see resolve_backend_selection)
//   LNS_DISABLE_OPS   force named ops / tensors back to the CPU backend
//   LNS_OP_AUDIT      log one JSON line per supports_op query
//
// Header-only on purpose: each backend is its own library, so each compiles a
// private copy, and every copy reads the same environment and therefore makes
// the same decisions.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>
#include <vector>

namespace lns {

// True when the environment variable is set to a non-empty value other than "0".
inline bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

// Splits "a, b,,c" into {"a", "b", "c"}: blanks around tokens are trimmed and
// empty tokens dropped (an empty token must never match anything).
inline std::vector<std::string> split_list(const std::string & list) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t comma = list.find(',', pos);
        std::string tok = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t first = tok.find_first_not_of(" \t");
        if (first != std::string::npos) {
            tokens.push_back(tok.substr(first, tok.find_last_not_of(" \t") - first + 1));
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return tokens;
}

// ============================================================
// Backend selection (GGML_LNS_BACKEND)
// ============================================================
//
// All LNS backends are ACCEL devices on host memory, so the scheduler would
// hand every op they all support to whichever registered first. The selection
// decides which of them answer supports_op at all.
//
// GGML_LNS_BACKEND is a comma-separated list of tokens. A token is either an
// alias from the table below or a device name, both case-insensitive. Unset or
// empty means the default alias. A GPU alias also selects the matching CPU
// backend, so ops the GPU backend declines still run in LNS rather than in
// plain F32 on the CPU backend; the GPU backend wins the ops it does support
// because it registers first. Select a device name alone (e.g.
// GGML_LNS_BACKEND=LNSCUDA16) to turn that fallback off.
//
// Adding a backend means adding its device name here (as an alias target) and
// nothing else in this file.

struct backend_alias {
    const char * token;
    const char * devices; // comma-separated device names
};

inline constexpr backend_alias k_backend_aliases[] = {
    { "16",     "LNS16"           },
    { "32",     "LNS32"           },
    { "cuda",   "LNSCUDA16,LNS16" },
    { "cuda16", "LNSCUDA16,LNS16" },
    { "cuda32", "LNSCUDA32,LNS32" },
};

inline constexpr const char * k_default_backend_selection = "16";

inline bool is_known_device_name(const std::string & name) {
    for (const backend_alias & a : k_backend_aliases) {
        for (const std::string & dev : split_list(a.devices)) {
            if (strcasecmp(dev.c_str(), name.c_str()) == 0) {
                return true;
            }
        }
    }
    return false;
}

// Expands a GGML_LNS_BACKEND value into the selected device names. Unknown
// tokens are reported on stderr and ignored; if nothing valid remains the
// default selection applies, as it always has for unrecognised values.
inline std::vector<std::string> resolve_backend_selection(const char * value) {
    std::vector<std::string> devices;
    const std::string spec = (value != nullptr && value[0] != '\0') ? value : k_default_backend_selection;
    for (const std::string & tok : split_list(spec)) {
        bool matched = false;
        for (const backend_alias & a : k_backend_aliases) {
            if (strcasecmp(a.token, tok.c_str()) == 0) {
                for (const std::string & dev : split_list(a.devices)) {
                    devices.push_back(dev);
                }
                matched = true;
                break;
            }
        }
        if (!matched && is_known_device_name(tok)) {
            devices.push_back(tok);
            matched = true;
        }
        if (!matched) {
            std::fprintf(stderr, "ggml-lns: ignoring unknown GGML_LNS_BACKEND token '%s'\n", tok.c_str());
        }
    }
    if (devices.empty()) {
        return resolve_backend_selection(k_default_backend_selection);
    }
    return devices;
}

// Whether the device with this name may claim ops in this process.
inline bool backend_selected(const char * device_name) {
    static const std::vector<std::string> selected = resolve_backend_selection(std::getenv("GGML_LNS_BACKEND"));
    for (const std::string & dev : selected) {
        if (strcasecmp(dev.c_str(), device_name) == 0) {
            return true;
        }
    }
    return false;
}

// ============================================================
// Op bisection (LNS_DISABLE_OPS)
// ============================================================
//
// LNS_DISABLE_OPS="MUL_MAT,ffn_down" forces matching ops back to the CPU
// backend without a rebuild. A token matches the ggml op name
// (case-insensitive) or, failing that, is a case-sensitive substring of the
// tensor name ("ffn_down", "result_output").

inline bool op_disabled_by_env(const ggml_tensor * op) {
    static const std::vector<std::string> tokens = []() {
        const char * env = std::getenv("LNS_DISABLE_OPS");
        return env ? split_list(env) : std::vector<std::string>();
    }();
    const char * name = ggml_op_name(op->op);
    for (const std::string & tok : tokens) {
        if (strcasecmp(tok.c_str(), name) == 0 ||
            (op->name[0] != '\0' && std::strstr(op->name, tok.c_str()) != nullptr)) {
            return true;
        }
    }
    return false;
}

// ============================================================
// supports_op audit (LNS_OP_AUDIT)
// ============================================================

inline bool audit_enabled() {
    static const bool enabled = env_flag("LNS_OP_AUDIT");
    return enabled;
}

// extra_fields is spliced in after "backend" verbatim and must be either empty
// or a sequence of JSON members each followed by a comma, e.g.
// "\"sum_mode\":\"base\",".
inline void audit_supports_op(const char * backend, const char * extra_fields,
                              const ggml_tensor * op, bool supported, const char * reason) {
    if (!audit_enabled()) {
        return;
    }
    auto type_name = [](const ggml_tensor * t) { return t ? ggml_type_name(t->type) : "null"; };
    std::fprintf(stderr,
        "{\"event\":\"lns_supports_op\","
        "\"backend\":\"%s\","
        "%s"
        "\"op\":\"%s\","
        "\"desc\":\"%s\","
        "\"supported\":%s,"
        "\"dst_type\":\"%s\","
        "\"src_types\":[\"%s\",\"%s\",\"%s\",\"%s\"],"
        "\"shape\":[%lld,%lld,%lld,%lld],"
        "\"reason\":\"%s\"}\n",
        backend,
        extra_fields,
        ggml_op_name(op->op),
        ggml_op_desc(op),
        supported ? "true" : "false",
        ggml_type_name(op->type),
        type_name(op->src[0]), type_name(op->src[1]), type_name(op->src[2]), type_name(op->src[3]),
        (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
        reason);
}

} // namespace lns
