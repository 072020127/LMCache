// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void makv_dequantize_scatter_out_cuda(
    const at::Tensor& raw16_payload,
    const at::Tensor& int8_payload,
    const at::Tensor& int4_payload,
    const at::Tensor& int2_payload,
    const at::Tensor& int8_scales,
    const at::Tensor& int4_scales,
    const at::Tensor& int2_scales,
    const at::Tensor& pos16,
    const at::Tensor& pos8,
    const at::Tensor& pos4,
    const at::Tensor& pos2,
    at::Tensor& output,
    int64_t importance_layout,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim);

void makv_dequantize_scatter_paged_out_cuda(
    const at::Tensor& raw16, const at::Tensor& int8_payload,
    const at::Tensor& int4_payload, const at::Tensor& int2_payload,
    const at::Tensor& int8_scales, const at::Tensor& int4_scales,
    const at::Tensor& int2_scales, const at::Tensor& pos16,
    const at::Tensor& pos8, const at::Tensor& pos4, const at::Tensor& pos2,
    const at::Tensor& page_ptrs, const at::Tensor& slot_mapping,
    int64_t layout, int64_t chunk_tokens, int64_t num_layers,
    int64_t num_heads, int64_t head_dim, int64_t page_buffer_size,
    int64_t block_size, int64_t format, int64_t output_dtype,
    int64_t skip_prefix);

static void makv_dequantize_scatter_out(
    const at::Tensor& raw16_payload,
    const at::Tensor& int8_payload,
    const at::Tensor& int4_payload,
    const at::Tensor& int2_payload,
    const at::Tensor& int8_scales,
    const at::Tensor& int4_scales,
    const at::Tensor& int2_scales,
    const at::Tensor& pos16,
    const at::Tensor& pos8,
    const at::Tensor& pos4,
    const at::Tensor& pos2,
    at::Tensor& output,
    int64_t importance_layout,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(output.device());
  makv_dequantize_scatter_out_cuda(
      raw16_payload,
      int8_payload,
      int4_payload,
      int2_payload,
      int8_scales,
      int4_scales,
      int2_scales,
      pos16,
      pos8,
      pos4,
      pos2,
      output,
      importance_layout,
      num_layers,
      num_heads,
      head_dim);
}

static void makv_dequantize_scatter_paged_out(
    const at::Tensor& raw16, const at::Tensor& int8_payload,
    const at::Tensor& int4_payload, const at::Tensor& int2_payload,
    const at::Tensor& int8_scales, const at::Tensor& int4_scales,
    const at::Tensor& int2_scales, const at::Tensor& pos16,
    const at::Tensor& pos8, const at::Tensor& pos4, const at::Tensor& pos2,
    const at::Tensor& page_ptrs, const at::Tensor& slot_mapping,
    int64_t layout, int64_t chunk_tokens, int64_t num_layers,
    int64_t num_heads, int64_t head_dim, int64_t page_buffer_size,
    int64_t block_size, int64_t format, int64_t output_dtype,
    int64_t skip_prefix) {
  TORCH_CHECK(page_ptrs.is_cuda(), "page_ptrs must be CUDA");
  c10::cuda::CUDAGuard device_guard(page_ptrs.device());
  makv_dequantize_scatter_paged_out_cuda(
      raw16, int8_payload, int4_payload, int2_payload, int8_scales,
      int4_scales, int2_scales, pos16, pos8, pos4, pos2, page_ptrs,
      slot_mapping, layout, chunk_tokens,
      num_layers, num_heads, head_dim, page_buffer_size, block_size,
      format, output_dtype, skip_prefix);
}

TORCH_LIBRARY(lmcache_makv, m) {
  m.def(
      "dequantize_scatter_paged_out(Tensor raw16, Tensor int8_payload, Tensor int4_payload, Tensor int2_payload, Tensor int8_scales, Tensor int4_scales, Tensor int2_scales, Tensor pos16, Tensor pos8, Tensor pos4, Tensor pos2, Tensor page_ptrs, Tensor slot_mapping, int layout, int chunk_tokens, int num_layers, int num_heads, int head_dim, int page_buffer_size, int block_size, int format, int output_dtype, int skip_prefix) -> ()");
  m.def(
      "dequantize_scatter_out(Tensor raw16_payload, Tensor int8_payload, Tensor int4_payload, Tensor int2_payload, Tensor int8_scales, Tensor int4_scales, Tensor int2_scales, Tensor pos16, Tensor pos8, Tensor pos4, Tensor pos2, Tensor(a!) output, int importance_layout, int num_layers, int num_heads, int head_dim) -> ()");
}

TORCH_LIBRARY_IMPL(lmcache_makv, CUDA, m) {
  m.impl("dequantize_scatter_paged_out",
         &makv_dequantize_scatter_paged_out);
  m.impl("dequantize_scatter_out", &makv_dequantize_scatter_out);
}
