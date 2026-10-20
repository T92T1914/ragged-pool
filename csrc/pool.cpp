#include <Python.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace {

void check_inputs(const at::Tensor& values, const at::Tensor& logits,
                  const at::Tensor& offsets) {
  for (const auto* t : {&values, &logits, &offsets}) {
    TORCH_CHECK(t->device().is_cpu(), "inputs must be on CPU");
    TORCH_CHECK(t->layout() == at::kStrided, "inputs must have strided layout");
    TORCH_CHECK(t->is_contiguous(), "forward inputs must be contiguous");
  }
  TORCH_CHECK(values.dim() == 2, "values must have shape [N,D]");
  TORCH_CHECK(logits.dim() == 1, "logits must have shape [N]");
  TORCH_CHECK(offsets.dim() == 1, "offsets must have shape [B+1]");
  TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kDouble,
              "values must be float32 or float64");
  TORCH_CHECK(logits.scalar_type() == values.scalar_type(), "logits and values must have the same dtype");
  TORCH_CHECK(offsets.scalar_type() == at::kLong, "offsets must be int64");
  TORCH_CHECK(values.size(0) == logits.size(0), "values and logits N must match");
  TORCH_CHECK(values.size(0) <= 4096, "N must be <=4096");
  TORCH_CHECK(values.size(1) >= 1 && values.size(1) <= 128, "D must be in [1,128]");
  TORCH_CHECK(offsets.numel() >= 1 && offsets.numel() <= 129, "B must be in [0,128]");
  const auto* o = offsets.const_data_ptr<int64_t>();
  TORCH_CHECK(o[0] == 0 && o[offsets.numel() - 1] == values.size(0),
              "offsets must start at zero and end at N");
  for (int64_t b = 1; b < offsets.numel(); ++b) {
    TORCH_CHECK(o[b - 1] <= o[b], "offsets must be nondecreasing");
  }
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "check_pool_range", [&] {
    for (const auto* t : {&values, &logits}) {
      const auto* p = t->const_data_ptr<scalar_t>();
      for (int64_t i = 0; i < t->numel(); ++i) {
        TORCH_CHECK(std::isfinite(static_cast<double>(p[i])) && std::abs(static_cast<double>(p[i])) <= 1000.0,
                    "values and logits must be finite and within [-1000,1000]");
      }
    }
  });
}

template <typename scalar_t>
std::vector<double> weights_for(const scalar_t* logits, int64_t start, int64_t end) {
  const double maximum = static_cast<double>(*std::max_element(logits + start, logits + end));
  std::vector<double> weights(static_cast<size_t>(end - start));
  double denominator = 0.0;
  for (int64_t i = start; i < end; ++i) {
    const double w = std::exp(static_cast<double>(logits[i]) - maximum);
    weights[static_cast<size_t>(i - start)] = w;
    denominator += w;
  }
  for (auto& w : weights) w /= denominator;
  return weights;
}

at::Tensor pool_cpu(const at::Tensor& values, const at::Tensor& logits,
                    const at::Tensor& offsets) {
  check_inputs(values, logits, offsets);
  const int64_t batches = offsets.numel() - 1, width = values.size(1);
  auto result = at::zeros({batches, width}, values.options());
  const auto* o = offsets.const_data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "pool_cpu", [&] {
    const auto* v = values.const_data_ptr<scalar_t>();
    const auto* l = logits.const_data_ptr<scalar_t>();
    auto* y = result.mutable_data_ptr<scalar_t>();
    for (int64_t b = 0; b < batches; ++b) {
      const int64_t start = o[b], end = o[b + 1];
      if (start == end) continue;
      const auto weights = weights_for(l, start, end);
      for (int64_t d = 0; d < width; ++d) {
        double sum = 0.0;
        for (int64_t i = start; i < end; ++i) sum += weights[i - start] * static_cast<double>(v[i * width + d]);
        y[b * width + d] = static_cast<scalar_t>(sum);
      }
    }
  });
  return result;
}

std::tuple<at::Tensor, at::Tensor> backward_cpu(
    const at::Tensor& grad_output, const at::Tensor& values,
    const at::Tensor& logits, const at::Tensor& offsets) {
  check_inputs(values, logits, offsets);
  const int64_t batches = offsets.numel() - 1, width = values.size(1);
  TORCH_CHECK(grad_output.device().is_cpu() && grad_output.layout() == at::kStrided,
              "grad_output must be strided CPU");
  TORCH_CHECK(grad_output.scalar_type() == values.scalar_type(), "grad_output dtype must match values");
  TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(0) == batches && grad_output.size(1) == width,
              "grad_output must have shape [B,D]");
  // Autograd commonly supplies expanded zero strides for sum() and views for VJPs.
  const auto grad = grad_output.resolve_neg().contiguous();
  auto grad_values = at::zeros_like(values);
  auto grad_logits = at::zeros_like(logits);
  const auto* o = offsets.const_data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "pool_backward_cpu", [&] {
    const auto* v = values.const_data_ptr<scalar_t>();
    const auto* l = logits.const_data_ptr<scalar_t>();
    const auto* g = grad.const_data_ptr<scalar_t>();
    auto* dv = grad_values.mutable_data_ptr<scalar_t>();
    auto* dl = grad_logits.mutable_data_ptr<scalar_t>();
    for (int64_t b = 0; b < batches; ++b) {
      const int64_t start = o[b], end = o[b + 1];
      if (start == end) continue;
      const auto weights = weights_for(l, start, end);
      std::vector<double> pooled(static_cast<size_t>(width), 0.0);
      for (int64_t i = start; i < end; ++i) {
        for (int64_t d = 0; d < width; ++d) pooled[d] += weights[i - start] * static_cast<double>(v[i * width + d]);
      }
      for (int64_t i = start; i < end; ++i) {
        double dot = 0.0;
        for (int64_t d = 0; d < width; ++d) {
          const double upstream = static_cast<double>(g[b * width + d]);
          dv[i * width + d] = static_cast<scalar_t>(weights[i - start] * upstream);
          dot += upstream * (static_cast<double>(v[i * width + d]) - pooled[d]);
        }
        dl[i] = static_cast<scalar_t>(weights[i - start] * dot);
      }
    }
  });
  return {grad_values, grad_logits};
}

}  // namespace

TORCH_LIBRARY(ragged_pool, m) {
  m.def("pool(Tensor values, Tensor logits, Tensor offsets) -> Tensor");
  m.def("_backward(Tensor grad_output, Tensor values, Tensor logits, Tensor offsets) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(ragged_pool, CPU, m) {
  m.impl("pool", &pool_cpu);
  m.impl("_backward", &backward_cpu);
}

// setuptools links a CPython extension entrypoint on Windows. Operator calls use
// the dispatcher after load_library, rather than a second pybind11 API.
PyMODINIT_FUNC PyInit__C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
