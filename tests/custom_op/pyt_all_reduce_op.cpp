#include <torch/script.h>
#include <torch/torch.h>

#include "pyt_all_reduce_kernel.hh"

static torch::Tensor custom_allreduce(torch::Tensor input) {
  return all_reduce_launcher(input);
}

static void ref_fp4_quantize(const torch::Tensor& A,
                             const torch::Tensor& absmax, torch::Tensor& out,
                             int64_t blocksize, int64_t n) {}

// static auto registry = torch::RegisterOperators("myop::skbmm", &skbmm);
TORCH_LIBRARY(my_ops, m) {
  m.def("custom_allreduce", &custom_allreduce);
  m.def("ref_fp4_quantize", &ref_fp4_quantize);
}
