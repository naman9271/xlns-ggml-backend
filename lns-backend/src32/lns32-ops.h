#pragma once

#include "ggml.h"

// Op kernels for the LNS32 backend. All tensors flow as F32 (or a quantized
// weight type with to_float); there is no GGML_TYPE_LNS32 — conversion to and
// from xlns32 happens per-op inside each kernel.

// Matrix multiplication
void lns32_mul_mat(struct ggml_tensor * dst);
void lns32_mul_mat_id(struct ggml_tensor * dst);

// Element-wise
void lns32_add(struct ggml_tensor * dst);
void lns32_mul(struct ggml_tensor * dst);
void lns32_div(struct ggml_tensor * dst);
void lns32_scale(struct ggml_tensor * dst);
void lns32_clamp(struct ggml_tensor * dst);

// Normalization / attention
void lns32_soft_max(struct ggml_tensor * dst);
void lns32_rms_norm(struct ggml_tensor * dst);
void lns32_diag_mask_inf(struct ggml_tensor * dst);

// Activations
void lns32_silu(struct ggml_tensor * dst);
void lns32_swiglu(struct ggml_tensor * dst);
void lns32_gelu(struct ggml_tensor * dst);
void lns32_geglu(struct ggml_tensor * dst);
void lns32_relu(struct ggml_tensor * dst);
void lns32_tanh(struct ggml_tensor * dst);

// Data movement
void lns32_get_rows(struct ggml_tensor * dst);
void lns32_argsort(struct ggml_tensor * dst);

// Positional encoding
void lns32_rope(struct ggml_tensor * dst);
