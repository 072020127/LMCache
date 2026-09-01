// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__device__ inline scalar_t cast_output(float v) {
  return static_cast<scalar_t>(v);
}

template <typename scalar_t, typename raw_t>
__global__ void raw16_token_kernel(
    const raw_t* __restrict__ payload,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_layers * 2 * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  tmp /= num_heads;
  const int64_t token_slot = tmp % pos_count;
  tmp /= pos_count;
  const int64_t kv = tmp % 2;
  const int64_t layer = tmp / 2;
  const int64_t token = positions[token_slot];
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(payload[idx]));
}

template <typename scalar_t>
__global__ void int8_token_kernel(
    const int8_t* __restrict__ payload,
    const float* __restrict__ scales,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_layers * 2 * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  tmp /= num_heads;
  const int64_t token_slot = tmp % pos_count;
  tmp /= pos_count;
  const int64_t kv = tmp % 2;
  const int64_t layer = tmp / 2;
  const int64_t token = positions[token_slot];
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  const int64_t scale_idx = (((layer * 2 + kv) * pos_count + token_slot) * num_heads) + h;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(payload[idx]) * scales[scale_idx]);
}

template <typename scalar_t, int bits>
__global__ void lowbit_token_kernel(
    const uint8_t* __restrict__ payload,
    const float* __restrict__ scales,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_layers * 2 * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  tmp /= num_heads;
  const int64_t token_slot = tmp % pos_count;
  tmp /= pos_count;
  const int64_t kv = tmp % 2;
  const int64_t layer = tmp / 2;
  const int64_t token = positions[token_slot];
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  const int64_t row = (((layer * 2 + kv) * pos_count + token_slot) * num_heads) + h;
  constexpr int values_per_byte = 8 / bits;
  constexpr int mask = (1 << bits) - 1;
  constexpr int sign = 1 << (bits - 1);
  const int64_t packed_dim = (head_dim + values_per_byte - 1) / values_per_byte;
  const int64_t byte_idx = row * packed_dim + (d / values_per_byte);
  const uint8_t packed = payload[byte_idx];
  const int field = (packed >> ((d % values_per_byte) * bits)) & mask;
  const int q = (field ^ sign) - sign;
  const int64_t scale_idx = row;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(q) * scales[scale_idx]);
}

template <typename scalar_t, typename raw_t>
__global__ void raw16_group_kernel(
    const raw_t* __restrict__ payload,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  const int64_t row = tmp / num_heads;
  const int64_t flat = positions[row];
  const int64_t layer = flat / (2 * chunk_tokens);
  const int64_t rem = flat % (2 * chunk_tokens);
  const int64_t kv = rem / chunk_tokens;
  const int64_t token = rem % chunk_tokens;
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(payload[idx]));
}

template <typename scalar_t>
__global__ void int8_group_kernel(
    const int8_t* __restrict__ payload,
    const float* __restrict__ scales,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  const int64_t row = tmp / num_heads;
  const int64_t flat = positions[row];
  const int64_t layer = flat / (2 * chunk_tokens);
  const int64_t rem = flat % (2 * chunk_tokens);
  const int64_t kv = rem / chunk_tokens;
  const int64_t token = rem % chunk_tokens;
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(payload[idx]) * scales[row * num_heads + h]);
}

template <typename scalar_t, int bits>
__global__ void lowbit_group_kernel(
    const uint8_t* __restrict__ payload,
    const float* __restrict__ scales,
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ output,
    int64_t chunk_tokens,
    int64_t pos_count,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  const int64_t total = pos_count * num_heads * head_dim;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  int64_t tmp = idx;
  const int64_t d = tmp % head_dim;
  tmp /= head_dim;
  const int64_t h = tmp % num_heads;
  const int64_t row = tmp / num_heads;
  const int64_t flat = positions[row];
  const int64_t layer = flat / (2 * chunk_tokens);
  const int64_t rem = flat % (2 * chunk_tokens);
  const int64_t kv = rem / chunk_tokens;
  const int64_t token = rem % chunk_tokens;
  const int64_t hidden = h * head_dim + d;
  const int64_t out_idx = (((kv * num_layers + layer) * chunk_tokens + token) * (num_heads * head_dim)) + hidden;
  constexpr int values_per_byte = 8 / bits;
  constexpr int mask = (1 << bits) - 1;
  constexpr int sign = 1 << (bits - 1);
  const int64_t packed_dim = (head_dim + values_per_byte - 1) / values_per_byte;
  const int64_t byte_idx =
      (row * num_heads + h) * packed_dim + (d / values_per_byte);
  const uint8_t packed = payload[byte_idx];
  const int field = (packed >> ((d % values_per_byte) * bits)) & mask;
  const int q = (field ^ sign) - sign;
  output[out_idx] = cast_output<scalar_t>(static_cast<float>(q) * scales[row * num_heads + h]);
}

template <typename scalar_t>
void launch_for_output(
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
  const int64_t chunk_tokens = output.size(2);
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (importance_layout == 0) {
    if (pos16.numel() > 0) {
      const int64_t total = pos16.numel() * num_layers * 2 * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      if (raw16_payload.scalar_type() == at::kHalf) {
        raw16_token_kernel<scalar_t, at::Half><<<blocks, threads, 0, stream>>>(
            raw16_payload.data_ptr<at::Half>(), pos16.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), chunk_tokens, pos16.numel(),
            num_layers, num_heads, head_dim);
      } else {
        raw16_token_kernel<scalar_t, at::BFloat16><<<blocks, threads, 0, stream>>>(
            raw16_payload.data_ptr<at::BFloat16>(), pos16.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), chunk_tokens, pos16.numel(),
            num_layers, num_heads, head_dim);
      }
    }
    if (pos8.numel() > 0) {
      const int64_t total = pos8.numel() * num_layers * 2 * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      int8_token_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          int8_payload.data_ptr<int8_t>(), int8_scales.data_ptr<float>(),
          pos8.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos8.numel(), num_layers, num_heads, head_dim);
    }
    if (pos4.numel() > 0) {
      const int64_t total = pos4.numel() * num_layers * 2 * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      lowbit_token_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
          int4_payload.data_ptr<uint8_t>(), int4_scales.data_ptr<float>(),
          pos4.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos4.numel(), num_layers, num_heads, head_dim);
    }
    if (pos2.numel() > 0) {
      const int64_t total = pos2.numel() * num_layers * 2 * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      lowbit_token_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
          int2_payload.data_ptr<uint8_t>(), int2_scales.data_ptr<float>(),
          pos2.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos2.numel(), num_layers, num_heads, head_dim);
    }
  } else {
    if (pos16.numel() > 0) {
      const int64_t total = pos16.numel() * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      if (raw16_payload.scalar_type() == at::kHalf) {
        raw16_group_kernel<scalar_t, at::Half><<<blocks, threads, 0, stream>>>(
            raw16_payload.data_ptr<at::Half>(), pos16.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), chunk_tokens, pos16.numel(),
            num_layers, num_heads, head_dim);
      } else {
        raw16_group_kernel<scalar_t, at::BFloat16><<<blocks, threads, 0, stream>>>(
            raw16_payload.data_ptr<at::BFloat16>(), pos16.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), chunk_tokens, pos16.numel(),
            num_layers, num_heads, head_dim);
      }
    }
    if (pos8.numel() > 0) {
      const int64_t total = pos8.numel() * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      int8_group_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          int8_payload.data_ptr<int8_t>(), int8_scales.data_ptr<float>(),
          pos8.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos8.numel(), num_layers, num_heads, head_dim);
    }
    if (pos4.numel() > 0) {
      const int64_t total = pos4.numel() * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      lowbit_group_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
          int4_payload.data_ptr<uint8_t>(), int4_scales.data_ptr<float>(),
          pos4.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos4.numel(), num_layers, num_heads, head_dim);
    }
    if (pos2.numel() > 0) {
      const int64_t total = pos2.numel() * num_heads * head_dim;
      const int blocks = (total + threads - 1) / threads;
      lowbit_group_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
          int2_payload.data_ptr<uint8_t>(), int2_scales.data_ptr<float>(),
          pos2.data_ptr<int32_t>(), output.data_ptr<scalar_t>(), chunk_tokens,
          pos2.numel(), num_layers, num_heads, head_dim);
    }
  }
}

}  // namespace

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
    int64_t head_dim) {
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(output.scalar_type() == at::kHalf || output.scalar_type() == at::kBFloat16,
              "output must be float16 or bfloat16");
  output.zero_();
  auto pos16_i32 = pos16.numel() == 0 ? pos16.to(output.device(), at::kInt) : pos16.to(output.device(), at::kInt, true, true);
  auto pos8_i32 = pos8.numel() == 0 ? pos8.to(output.device(), at::kInt) : pos8.to(output.device(), at::kInt, true, true);
  auto pos4_i32 = pos4.numel() == 0 ? pos4.to(output.device(), at::kInt) : pos4.to(output.device(), at::kInt, true, true);
  auto pos2_i32 = pos2.numel() == 0 ? pos2.to(output.device(), at::kInt) : pos2.to(output.device(), at::kInt, true, true);
  auto raw16_dev = raw16_payload.numel() == 0 ? raw16_payload.to(output.device()) : raw16_payload.to(output.device(), raw16_payload.scalar_type(), true, true);
  auto int8_dev = int8_payload.numel() == 0 ? int8_payload.to(output.device(), at::kChar) : int8_payload.to(output.device(), at::kChar, true, true);
  auto int4_dev = int4_payload.numel() == 0 ? int4_payload.to(output.device(), at::kByte) : int4_payload.to(output.device(), at::kByte, true, true);
  auto int2_dev = int2_payload.numel() == 0 ? int2_payload.to(output.device(), at::kByte) : int2_payload.to(output.device(), at::kByte, true, true);
  auto int8_scales_f = int8_scales.numel() == 0 ? int8_scales.to(output.device(), at::kFloat) : int8_scales.to(output.device(), at::kFloat, true, true);
  auto int4_scales_f = int4_scales.numel() == 0 ? int4_scales.to(output.device(), at::kFloat) : int4_scales.to(output.device(), at::kFloat, true, true);
  auto int2_scales_f = int2_scales.numel() == 0 ? int2_scales.to(output.device(), at::kFloat) : int2_scales.to(output.device(), at::kFloat, true, true);

  if (output.scalar_type() == at::kHalf) {
    launch_for_output<at::Half>(
        raw16_dev,
        int8_dev,
        int4_dev,
        int2_dev,
        int8_scales_f,
        int4_scales_f,
        int2_scales_f,
        pos16_i32,
        pos8_i32,
        pos4_i32,
        pos2_i32,
        output,
        importance_layout,
        num_layers,
        num_heads,
        head_dim);
  } else {
    launch_for_output<at::BFloat16>(
        raw16_dev,
        int8_dev,
        int4_dev,
        int2_dev,
        int8_scales_f,
        int4_scales_f,
        int2_scales_f,
        pos16_i32,
        pos8_i32,
        pos4_i32,
        pos2_i32,
        output,
        importance_layout,
        num_layers,
        num_heads,
        head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
