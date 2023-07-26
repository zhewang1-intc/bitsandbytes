#include <torch/script.h>

#include "pyt_all_reduce_kernel.hh"
#define BLOCKX_DIM 256

template <typename scalar_t>
void cpu_all_reduce(float* sum, scalar_t* data, int n) {
  scalar_t temp_sum = 0;
  for (int i = 0; i < n; ++i) {
    temp_sum += data[i];
  }
  *sum = temp_sum;
}

template <typename scalar_t>
__global__ void gpu_all_reduce(float* sum, scalar_t* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  scalar_t temp = 0;
  for (int i = idx; i < n; i += stride) {
    temp += data[i];
  }

  atomicAdd(sum, temp);
}

torch::Tensor all_reduce_launcher(torch::Tensor input) {
  torch::Device device(torch::kCUDA, 0);
  torch::Tensor output = torch::zeros(1, torch::kFloat);
  if (input.device() == device) {
    output = output.to(device);
    dim3 blockSize(BLOCKX_DIM);
    dim3 gridSize((input.size(0) + BLOCKX_DIM - 1) / BLOCKX_DIM);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "gpu_all_reduce", ([&] {
          gpu_all_reduce<scalar_t><<<gridSize, blockSize, 0, stream>>>(
              output.data_ptr<float>(), input.data_ptr<scalar_t>(),
              input.size(0));
        }));
  } else {
    cpu_all_reduce<int>(output.data_ptr<float>(), input.data_ptr<int>(),
                        input.size(0));
  }
  return output;
}

float dDequantizeFP4Tree(unsigned char val, float absmax) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4)                   // 0
    if ((val & 0b0010) == 2)                 // 01
      if ((val & 0b0001) == 1)               // 111
        return 0.25000000f * absmax * sign;  // 1111
      else
        return 0.16666667f * absmax * sign;  // 1110
    else if ((val & 0b0001) == 1)            // 110
      return 0.50000000f * absmax * sign;    // 1101
    else
      return 0.33333333f * absmax * sign;  // 1100
  else if ((val & 0b0010) == 2)            // 10
    if ((val & 0b0001) == 1)               // 101
      return 1.00000000f * absmax * sign;  // 1011
    else
      return 0.66666667f * absmax * sign;     // 1010
  else if ((val & 0b0001) == 1)               // 100
    return 5.208333333e-03f * absmax * sign;  // 1001
  else
    return 0.00000000f * absmax * sign;  // 1000
}

unsigned char dQuantizeFP4(float x) {
  // FP4 with bias of 3
  // first bit is a sign
  // subnormals
  // 0b000 = 0
  // 0b001 = 0.0625
  // 0b110 = 2
  // 0b111 = 3
  // 0b100 = 4
  // 0b101 = 6
  // 0b010 = 8
  // 0b011 = 12

  // we do a binary search
  // the pivots are divided by 12 (the FP4 absmax)
  // since we assum input data is in [-1.0, 1.0]

  // !be careful here, its easy to make a mistake
  // that is difficult to noice if you add an extra
  // zero somewhere!

  int sign = x < 0 ? 0b1000 : 0b0000;
  x = fabsf(x);
  if (x > 0.29166667f)
    if (x > 0.583333f)
      if (x > 0.8333333f)
        return 0b0011 + sign;
      else
        return 0b0010 + sign;
    else if (x > 0.4166667f)
      return 0b101 + sign;
    else
      return 0b100 + sign;
  else if (x > 0.0859375f)
    if (x > 0.20833333f)
      return 0b0111 + sign;
    else
      return 0b0110 + sign;
  else if (x > 0.00260417f)
    return 0b0001 + sign;
  else
    return 0b0000 + sign;
}

void fp4_quantize_launcher(const torch::Tensor& A, torch::Tensor& absmax,
                           torch::Tensor& out, int64_t blocksize, int64_t n) {
  auto blocks = absmax.sizes()[0];
  auto src = A.data_ptr<float>();
  auto absmax_ptr = absmax.data_ptr<float>();
  auto out_ptr = out.data_ptr<unsigned char>();
  for (int b = 0; b < blocks; b++) {
    float max = -99999999999999.f;
    size_t offset = b * blocksize;
    for (int i = 0; i < blocksize; i++) {
      if (offset + i >= n) break;
      max = std::abs(src[offset + i]) > max ? std::abs(src[offset + i]) : max;
    }
    absmax_ptr[b] = max;
    for (int i = 0; i < blocksize / 2; i++) {
      unsigned char packed_4bit = 0;
      if (offset + i * 2 >= n) break;
      packed_4bit |= dQuantizeFP4(src[offset + 2 * i] * (1.f / max)) << 4;
      packed_4bit |= dQuantizeFP4(src[offset + 2 * i + 1] * (1.f / max));
      out_ptr[offset / 2 + i] = packed_4bit;
    }
  }
}

void fp4_dequantize_launcher(const torch::Tensor& A, torch::Tensor& absmax,
                             torch::Tensor& out, int64_t blocksize, int64_t n) {
  auto blocks = absmax.sizes()[0];
  auto src = A.data_ptr<unsigned char>();
  auto absmax_ptr = absmax.data_ptr<float>();
  auto out_ptr = out.data_ptr<float>();
  for (int b = 0; b < blocks; b++) {
    size_t offset = b * blocksize;
    auto max = absmax_ptr[b];
    for (int i = 0; i < blocksize / 2; i++) {
      unsigned char packed_4bit = 0;
      if (offset + i * 2 >= n) break;
      out_ptr[offset + 2 * i] =
          dDequantizeFP4Tree(src[offset / 2 + i] >> 4, max);
      out_ptr[offset + 2 * i + 1] =
          dDequantizeFP4Tree(src[offset / 2 + i] & 0x0f, max);
    }
  }
}