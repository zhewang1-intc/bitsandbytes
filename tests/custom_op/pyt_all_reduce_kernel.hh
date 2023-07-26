#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>

torch::Tensor all_reduce_launcher(torch::Tensor input);

void fp4_quantize_launcher(const torch::Tensor& A, torch::Tensor& absmax,
                           torch::Tensor& out, int64_t blocksize, int64_t n);

void fp4_dequantize_launcher(const torch::Tensor& A, torch::Tensor& absmax,
                             torch::Tensor& out, int64_t blocksize, int64_t n);