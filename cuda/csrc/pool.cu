#include <Python.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cmath>
#include <tuple>

namespace {

void check_metadata(const at::Tensor& values, const at::Tensor& logits,
                    const at::Tensor& offsets) {
  for (const auto* tensor : {&values, &logits, &offsets}) {
    TORCH_CHECK(tensor->is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(tensor->layout() == at::kStrided, "inputs must have strided layout");
    TORCH_CHECK(tensor->is_contiguous(), "forward inputs must be contiguous");
    TORCH_CHECK(!tensor->is_neg() && !tensor->is_conj(),
                "logical negative/conjugate primal views are not supported");
    TORCH_CHECK(tensor->device() == values.device(), "inputs must share one CUDA device");
  }
  TORCH_CHECK(values.dim() == 2, "values must have shape [N,D]");
  TORCH_CHECK(logits.dim() == 1, "logits must have shape [N]");
  TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [B+1]");
  TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kDouble,
              "values must be float32 or float64");
  TORCH_CHECK(logits.scalar_type() == values.scalar_type(), "values/logits dtypes must match");
  TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
  TORCH_CHECK(values.size(0) == logits.size(0), "values/logits N must match");
  TORCH_CHECK(values.size(0) <= 4096, "N must be <=4096");
  TORCH_CHECK(values.size(1) >= 1 && values.size(1) <= 128, "D must be in [1,128]");
  TORCH_CHECK(offsets.numel() >= 1 && offsets.numel() <= 129, "B must be in [0,128]");
}

void check_grad_metadata(const at::Tensor& grad, const at::Tensor& values,
                         const at::Tensor& offsets) {
  TORCH_CHECK(grad.device() == values.device() && grad.layout() == at::kStrided,
              "grad_output must be strided on the input CUDA device");
  TORCH_CHECK(grad.scalar_type() == values.scalar_type(), "grad_output dtype must match values");
  TORCH_CHECK(grad.dim() == 2 && grad.size(0) == offsets.numel() - 1 &&
              grad.size(1) == values.size(1), "grad_output must have shape [B,D]");
}

template <typename scalar_t>
__global__ void validate_contents(const scalar_t* values, const scalar_t* logits,
                                  const int64_t* offsets, const scalar_t* grad,
                                  int64_t n, int64_t width, int64_t groups,
                                  int* status) {
  // One writer, linear bounded reads only. No untrusted offset is an address.
  int invalid = 0;
  for (int64_t b = 0; b <= groups; ++b) {
    const int64_t current = offsets[b];
    if (current < 0 || current > n || (b == 0 && current != 0) ||
        (b == groups && current != n) || (b > 0 && offsets[b - 1] > current)) {
      invalid |= 1;
    }
  }
  for (int64_t i = 0; i < n * width; ++i) {
    const double value = static_cast<double>(values[i]);
    if (!isfinite(value) || fabs(value) > 1000.0) invalid |= 2;
  }
  for (int64_t i = 0; i < n; ++i) {
    const double value = static_cast<double>(logits[i]);
    if (!isfinite(value) || fabs(value) > 1000.0) invalid |= 4;
  }
  if (grad != nullptr) {
    for (int64_t i = 0; i < groups * width; ++i) {
      const double value = static_cast<double>(grad[i]);
      if (!isfinite(value) || fabs(value) > 1000.0) invalid |= 8;
    }
  }
  *status = invalid;
}

void admit_contents(const at::Tensor& values, const at::Tensor& logits,
                    const at::Tensor& offsets, const at::Tensor* grad,
                    cudaStream_t stream) {
  auto status = at::empty({1}, values.options().dtype(at::kInt));
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_admission", [&] {
    validate_contents<scalar_t><<<1, 1, 0, stream>>>(
        values.const_data_ptr<scalar_t>(), logits.const_data_ptr<scalar_t>(),
        offsets.const_data_ptr<int64_t>(), grad ? grad->const_data_ptr<scalar_t>() : nullptr,
        values.size(0), values.size(1), offsets.numel() - 1, status.mutable_data_ptr<int>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int invalid = 0;
  // An explicit four-byte device-to-host transfer synchronizes this stream.
  // Keep status alive until the copy completes. Indexing is admitted only after.
  c10::cuda::memcpy_and_sync(&invalid, status.const_data_ptr<int>(), sizeof(invalid),
                            cudaMemcpyDeviceToHost, stream);
  TORCH_CHECK(!(invalid & 1), "offsets must start at zero, end at N, stay in [0,N] and be nondecreasing");
  TORCH_CHECK(!(invalid & 6), "values/logits must be finite and within [-1000,1000]");
  TORCH_CHECK(!(invalid & 8), "grad_output must be finite and within [-1000,1000]");
}

template <typename scalar_t>
__device__ double group_maximum(const scalar_t* logits, int64_t start, int64_t end) {
  double maximum = static_cast<double>(logits[start]);
  for (int64_t i = start + 1; i < end; ++i) {
    const double value = static_cast<double>(logits[i]);
    maximum = maximum > value ? maximum : value;
  }
  return maximum;
}

template <typename scalar_t>
__device__ double group_denominator(const scalar_t* logits, int64_t start,
                                    int64_t end, double maximum) {
  double denominator = 0.0;
  for (int64_t i = start; i < end; ++i) {
    denominator += exp(static_cast<double>(logits[i]) - maximum);
  }
  return denominator;
}

template <typename scalar_t>
__global__ void pool_forward(const scalar_t* values, const scalar_t* logits,
                             const int64_t* offsets, int64_t width,
                             scalar_t* output) {
  const int64_t group = blockIdx.x;
  const int64_t start = offsets[group], end = offsets[group + 1];
  if (start == end) return;  // The host allocated a zero-filled output.
  const double maximum = group_maximum(logits, start, end);
  const double denominator = group_denominator(logits, start, end, maximum);
  double pooled[128] = {};
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    for (int64_t d = 0; d < width; ++d) {
      pooled[d] += weight * static_cast<double>(values[i * width + d]);
    }
  }
  for (int64_t d = 0; d < width; ++d) {
    output[group * width + d] = static_cast<scalar_t>(pooled[d]);
  }
}

template <typename scalar_t>
__global__ void pool_backward(const scalar_t* grad, const scalar_t* values,
                              const scalar_t* logits, const int64_t* offsets,
                              int64_t width, scalar_t* grad_values,
                              scalar_t* grad_logits) {
  const int64_t group = blockIdx.x;
  const int64_t start = offsets[group], end = offsets[group + 1];
  if (start == end) return;
  const double maximum = group_maximum(logits, start, end);
  const double denominator = group_denominator(logits, start, end, maximum);
  double pooled[128] = {};
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    for (int64_t d = 0; d < width; ++d) {
      pooled[d] += weight * static_cast<double>(values[i * width + d]);
    }
  }
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    double dot = 0.0;
    for (int64_t d = 0; d < width; ++d) {
      const double upstream = static_cast<double>(grad[group * width + d]);
      grad_values[i * width + d] = static_cast<scalar_t>(weight * upstream);
      dot += upstream * (static_cast<double>(values[i * width + d]) - pooled[d]);
    }
    grad_logits[i] = static_cast<scalar_t>(weight * dot);
  }
}

at::Tensor forward_cuda(const at::Tensor& values, const at::Tensor& logits,
                        const at::Tensor& offsets) {
  check_metadata(values, logits, offsets);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  admit_contents(values, logits, offsets, nullptr, stream);
  const int64_t groups = offsets.numel() - 1, width = values.size(1);
  auto output = at::zeros({groups, width}, values.options());
  if (groups > 0) {
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_forward", [&] {
      pool_forward<scalar_t><<<groups, 1, 0, stream>>>(
          values.const_data_ptr<scalar_t>(), logits.const_data_ptr<scalar_t>(),
          offsets.const_data_ptr<int64_t>(), width, output.mutable_data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> backward_cuda(
    const at::Tensor& grad_output, const at::Tensor& values,
    const at::Tensor& logits, const at::Tensor& offsets) {
  check_metadata(values, logits, offsets);
  check_grad_metadata(grad_output, values, offsets);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  const auto grad = grad_output.resolve_neg().contiguous();
  admit_contents(values, logits, offsets, &grad, stream);
  auto grad_values = at::zeros_like(values);
  auto grad_logits = at::zeros_like(logits);
  const int64_t groups = offsets.numel() - 1;
  if (groups > 0) {
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_backward", [&] {
      pool_backward<scalar_t><<<groups, 1, 0, stream>>>(
          grad.const_data_ptr<scalar_t>(), values.const_data_ptr<scalar_t>(),
          logits.const_data_ptr<scalar_t>(), offsets.const_data_ptr<int64_t>(),
          values.size(1), grad_values.mutable_data_ptr<scalar_t>(),
          grad_logits.mutable_data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_values, grad_logits};
}

}  // namespace

TORCH_LIBRARY(ragged_pool_cuda, library) {
  library.def("pool(Tensor values, Tensor logits, Tensor offsets) -> Tensor");
  library.def("_backward(Tensor grad_output, Tensor values, Tensor logits, Tensor offsets) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(ragged_pool_cuda, CUDA, library) {
  library.impl("pool", forward_cuda);
  library.impl("_backward", backward_cuda);
}
// Inspect original primals before a generic Negative fallback could copy them.
// Backward intentionally resolves only its admitted logical upstream gradient.
TORCH_LIBRARY_IMPL(ragged_pool_cuda, Negative, library) {
  library.impl("pool", forward_cuda);
  library.impl("_backward", backward_cuda);
}

// Setuptools exports this CPython symbol on Windows. Tensor calls use the
// dispatcher registrations above, without a separate pybind11 interface.
PyMODINIT_FUNC PyInit__C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
