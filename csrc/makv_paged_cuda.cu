// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

enum class BucketKind : int { Raw16 = 0, Int8 = 1, Int4 = 2, Int2 = 3 };

__device__ __forceinline__ int64_t makv_page_offset(
    int format, int kv, int64_t slot, int64_t hidden_offset,
    int64_t hidden_size, int64_t page_buffer_size, int64_t block_size,
    int64_t head_dim) {
  if (format == 0 || format == 1) {
    return kv * page_buffer_size * hidden_size + slot * hidden_size +
           hidden_offset;
  }
  const int64_t block = slot / block_size;
  const int64_t block_offset = slot % block_size;
  if (format == 2) {
    return block * 2 * block_size * hidden_size +
           kv * block_size * hidden_size + block_offset * hidden_size +
           hidden_offset;
  }
  const int64_t head = hidden_offset / head_dim;
  const int64_t dim = hidden_offset % head_dim;
  const int64_t num_heads = hidden_size / head_dim;
  if (format == 6) {
    return kv * page_buffer_size * hidden_size +
           block * num_heads * block_size * head_dim +
           head * block_size * head_dim + block_offset * head_dim + dim;
  }
  if (format == 7) {
    return block * 2 * num_heads * block_size * head_dim +
           kv * num_heads * block_size * head_dim +
           head * block_size * head_dim + block_offset * head_dim + dim;
  }
  if (format == 12) {
    const int64_t content_size = 2 * head_dim;
    return block * num_heads * block_size * content_size +
           head * block_size * content_size +
           block_offset * content_size + kv * head_dim + dim;
  }
  if (format == 13) {
    const int64_t content_size = 2 * head_dim;
    return block * block_size * num_heads * content_size +
           block_offset * num_heads * content_size +
           head * content_size + kv * head_dim + dim;
  }
  return -1;
}

template <typename output_t, typename raw_t, BucketKind kind>
__global__ void makv_paged_kernel(
    const raw_t* __restrict__ raw_payload,
    const int8_t* __restrict__ int8_payload,
    const uint8_t* __restrict__ int4_payload,
    const uint8_t* __restrict__ int2_payload,
    const float* __restrict__ scales,
    const int32_t* __restrict__ positions,
    output_t** __restrict__ page_ptrs,
    const int64_t* __restrict__ slot_mapping,
    int64_t pos_count, int64_t chunk_tokens, int64_t num_layers,
    int64_t num_heads, int64_t head_dim, int64_t page_buffer_size,
    int64_t block_size, int format, int layout, int64_t skip_prefix) {
  const int64_t elements_per_position =
      layout == 0 ? num_layers * 2 * num_heads * head_dim
                  : num_heads * head_dim;
  const int64_t total = pos_count * elements_per_position;
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  tmp /= num_heads;

  int64_t row;
  int64_t layer;
  int64_t kv;
  int64_t token;
  if (layout == 0) {
    const int64_t token_slot = tmp % pos_count;
    tmp /= pos_count;
    kv = tmp % 2;
    layer = tmp / 2;
    token = positions[token_slot];
    row = (((layer * 2 + kv) * pos_count + token_slot) * num_heads) + h;
  } else {
    row = tmp;
    const int64_t flat = positions[row];
    layer = flat / (2 * chunk_tokens);
    const int64_t rem = flat % (2 * chunk_tokens);
    kv = rem / chunk_tokens;
    token = rem % chunk_tokens;
    row = row * num_heads + h;
  }

  if (token < skip_prefix || token < 0 || token >= chunk_tokens ||
      layer < 0 || layer >= num_layers) {
    return;
  }
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  float value;
  if constexpr (kind == BucketKind::Raw16) {
    value = static_cast<float>(raw_payload[idx]);
  } else if constexpr (kind == BucketKind::Int8) {
    value = static_cast<float>(int8_payload[idx]) * scales[row];
  } else if constexpr (kind == BucketKind::Int4) {
    const int64_t packed_dim = (head_dim + 1) / 2;
    const uint8_t packed = int4_payload[row * packed_dim + d / 2];
    const int field = (packed >> ((d & 1) * 4)) & 0x0f;
    const int q = (field ^ 0x8) - 0x8;
    value = static_cast<float>(q) * scales[row];
  } else {
    const int64_t packed_dim = (head_dim + 3) / 4;
    const uint8_t packed = int2_payload[row * packed_dim + d / 4];
    const int field = (packed >> ((d & 3) * 2)) & 0x03;
    const int q = (field ^ 0x2) - 0x2;
    value = static_cast<float>(q) * scales[row];
  }

  const int64_t hidden = h * head_dim + d;
  const int64_t offset =
      makv_page_offset(format, kv, slot, hidden, num_heads * head_dim,
                       page_buffer_size, block_size, head_dim);
  if (offset >= 0) {
    page_ptrs[layer][offset] = static_cast<output_t>(value);
  }
}

template <typename output_t, typename raw_t>
void launch_paged(
    const at::Tensor& raw16, const at::Tensor& int8_payload,
    const at::Tensor& int4_payload, const at::Tensor& int2_payload,
    const at::Tensor& int8_scales, const at::Tensor& int4_scales,
    const at::Tensor& int2_scales, const at::Tensor& pos16,
    const at::Tensor& pos8, const at::Tensor& pos4, const at::Tensor& pos2,
    const at::Tensor& page_ptrs, const at::Tensor& slot_mapping,
    int64_t layout, int64_t chunk_tokens, int64_t num_layers,
    int64_t num_heads, int64_t head_dim, int64_t page_buffer_size,
    int64_t block_size, int64_t format, int64_t skip_prefix) {
  constexpr int threads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_BUCKET(POS, KIND, SCALES)                                    \
  do {                                                                      \
    if ((POS).numel() > 0) {                                                \
      const int64_t per_pos =                                               \
          layout == 0 ? num_layers * 2 * num_heads * head_dim               \
                      : num_heads * head_dim;                                \
      const int64_t total = (POS).numel() * per_pos;                         \
      const int blocks = static_cast<int>((total + threads - 1) / threads);  \
      makv_paged_kernel<output_t, raw_t, KIND>                               \
          <<<blocks, threads, 0, stream>>>(                                  \
              raw16.data_ptr<raw_t>(), int8_payload.data_ptr<int8_t>(),      \
              int4_payload.data_ptr<uint8_t>(), int2_payload.data_ptr<uint8_t>(), \
              (SCALES).data_ptr<float>(),                                    \
              (POS).data_ptr<int32_t>(),                                    \
              reinterpret_cast<output_t**>(page_ptrs.data_ptr<int64_t>()),   \
              slot_mapping.data_ptr<int64_t>(), (POS).numel(), chunk_tokens,\
              num_layers, num_heads, head_dim, page_buffer_size, block_size, \
              static_cast<int>(format), static_cast<int>(layout),            \
              skip_prefix);                                                  \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    }                                                                        \
  } while (0)

  LAUNCH_BUCKET(pos16, BucketKind::Raw16, int8_scales);
  LAUNCH_BUCKET(pos8, BucketKind::Int8, int8_scales);
  LAUNCH_BUCKET(pos4, BucketKind::Int4, int4_scales);
  LAUNCH_BUCKET(pos2, BucketKind::Int2, int2_scales);
#undef LAUNCH_BUCKET
}

}  // namespace

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
    int64_t skip_prefix) {
  TORCH_CHECK(layout == 0 || layout == 1, "invalid MaKV importance layout");
  TORCH_CHECK(output_dtype == 0 || output_dtype == 1,
              "invalid MaKV output dtype");
  TORCH_CHECK(chunk_tokens >= 0 && num_layers >= 0 && skip_prefix >= 0,
              "invalid MaKV paged dimensions");
  TORCH_CHECK(page_buffer_size >= 0, "invalid MaKV page buffer size");
  TORCH_CHECK(format == 0 || format == 1 || format == 2 || format == 6 ||
                  format == 7 || format == 12 || format == 13,
              "MaKV direct paged restore does not support EngineKVFormat ",
              format);
  TORCH_CHECK(raw16.is_cuda() &&
                  (raw16.scalar_type() == at::kHalf ||
                   raw16.scalar_type() == at::kBFloat16),
              "raw16 must be a CUDA FP16 or BF16 tensor");
  TORCH_CHECK(int8_payload.is_cuda() && int8_payload.scalar_type() == at::kChar,
              "int8 payload must be a CUDA int8 tensor");
  TORCH_CHECK(int4_payload.is_cuda() && int4_payload.scalar_type() == at::kByte,
              "int4 payload must be a CUDA uint8 tensor");
  TORCH_CHECK(int2_payload.is_cuda() && int2_payload.scalar_type() == at::kByte,
              "int2 payload must be a CUDA uint8 tensor");
  TORCH_CHECK(int8_scales.is_cuda() && int8_scales.scalar_type() == at::kFloat &&
                  int4_scales.is_cuda() && int4_scales.scalar_type() == at::kFloat &&
                  int2_scales.is_cuda() && int2_scales.scalar_type() == at::kFloat,
              "MaKV paged scales must be CUDA float32 tensors");
  TORCH_CHECK(pos16.is_cuda() && pos16.scalar_type() == at::kInt &&
                  pos8.is_cuda() && pos8.scalar_type() == at::kInt &&
                  pos4.is_cuda() && pos4.scalar_type() == at::kInt &&
                  pos2.is_cuda() && pos2.scalar_type() == at::kInt,
              "MaKV positions must be CUDA int32 tensors");
  TORCH_CHECK(page_ptrs.is_cuda() && page_ptrs.scalar_type() == at::kLong,
              "page_ptrs must be a CUDA int64 tensor");
  TORCH_CHECK(slot_mapping.is_cuda() &&
                  slot_mapping.scalar_type() == at::kLong,
              "slot_mapping must be a CUDA int64 tensor");
  const auto device = page_ptrs.device();
  TORCH_CHECK(raw16.device() == device && int8_payload.device() == device &&
                  int4_payload.device() == device &&
                  int2_payload.device() == device &&
                  int8_scales.device() == device &&
                  int4_scales.device() == device && pos16.device() == device &&
                  int2_scales.device() == device && pos8.device() == device &&
                  pos4.device() == device && pos2.device() == device &&
                  slot_mapping.device() == device,
              "all MaKV paged inputs must be on the same CUDA device");
  TORCH_CHECK(raw16.is_contiguous() && int8_payload.is_contiguous() &&
                  int4_payload.is_contiguous() && int8_scales.is_contiguous() &&
                  int2_payload.is_contiguous() && int4_scales.is_contiguous() &&
                  int2_scales.is_contiguous() && pos16.is_contiguous() &&
                  pos8.is_contiguous() && pos4.is_contiguous() && pos2.is_contiguous() &&
                  page_ptrs.is_contiguous() && slot_mapping.is_contiguous(),
              "all MaKV paged inputs must be contiguous");
  TORCH_CHECK(page_ptrs.numel() == num_layers,
              "page_ptrs length must equal num_layers");
  TORCH_CHECK(slot_mapping.numel() >= chunk_tokens,
              "slot_mapping shorter than MaKV chunk");
  TORCH_CHECK(block_size > 0 && head_dim > 0 && num_heads > 0,
              "invalid MaKV paged geometry");
  c10::cuda::CUDAGuard guard(page_ptrs.device());

  if (output_dtype == 0) {
    if (raw16.scalar_type() == at::kHalf) {
      launch_paged<at::Half, at::Half>(
          raw16, int8_payload, int4_payload, int2_payload, int8_scales,
          int4_scales, int2_scales, pos16, pos8, pos4, pos2, page_ptrs,
          slot_mapping, layout, chunk_tokens,
          num_layers, num_heads, head_dim, page_buffer_size, block_size,
          format, skip_prefix);
    } else {
      launch_paged<at::Half, at::BFloat16>(
          raw16, int8_payload, int4_payload, int2_payload, int8_scales,
          int4_scales, int2_scales, pos16, pos8, pos4, pos2, page_ptrs,
          slot_mapping, layout, chunk_tokens,
          num_layers, num_heads, head_dim, page_buffer_size, block_size,
          format, skip_prefix);
    }
  } else {
    if (raw16.scalar_type() == at::kHalf) {
      launch_paged<at::BFloat16, at::Half>(
          raw16, int8_payload, int4_payload, int2_payload, int8_scales,
          int4_scales, int2_scales, pos16, pos8, pos4, pos2, page_ptrs,
          slot_mapping, layout, chunk_tokens,
          num_layers, num_heads, head_dim, page_buffer_size, block_size,
          format, skip_prefix);
    } else {
      launch_paged<at::BFloat16, at::BFloat16>(
          raw16, int8_payload, int4_payload, int2_payload, int8_scales,
          int4_scales, int2_scales, pos16, pos8, pos4, pos2, page_ptrs,
          slot_mapping, layout, chunk_tokens,
          num_layers, num_heads, head_dim, page_buffer_size, block_size,
          format, skip_prefix);
    }
  }
}
