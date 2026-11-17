#include <Python.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

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

void check_tangent_metadata(const at::Tensor& tangent, const at::Tensor& primal) {
  TORCH_CHECK(tangent.device() == primal.device() && tangent.layout() == at::kStrided,
              "tangents must be strided on the primal CUDA device");
  TORCH_CHECK(tangent.scalar_type() == primal.scalar_type(),
              "tangent dtype must match its primal");
  TORCH_CHECK(tangent.sizes() == primal.sizes(), "tangent shape must match its primal");
  TORCH_CHECK(!tangent.is_conj(), "conjugate tangent views are not supported");
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
__global__ void validate_tangents(const scalar_t* dvalues, const scalar_t* dlogits,
                                  int64_t value_count, int64_t logit_count,
                                  int* status) {
  // Tensor lengths come from admitted metadata. Offsets are never addresses here.
  int invalid = 0;
  for (int64_t i = 0; i < value_count; ++i) {
    const double value = static_cast<double>(dvalues[i]);
    if (!isfinite(value) || fabs(value) > 1000.0) invalid = 1;
  }
  for (int64_t i = 0; i < logit_count; ++i) {
    const double value = static_cast<double>(dlogits[i]);
    if (!isfinite(value) || fabs(value) > 1000.0) invalid = 1;
  }
  *status = invalid;
}

void admit_tangents(const at::Tensor& dvalues, const at::Tensor& dlogits,
                    cudaStream_t stream) {
  auto status = at::empty({1}, dvalues.options().dtype(at::kInt));
  AT_DISPATCH_FLOATING_TYPES(dvalues.scalar_type(), "ragged_cuda_tangent_admission", [&] {
    validate_tangents<scalar_t><<<1, 1, 0, stream>>>(
        dvalues.const_data_ptr<scalar_t>(), dlogits.const_data_ptr<scalar_t>(),
        dvalues.numel(), dlogits.numel(), status.mutable_data_ptr<int>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int invalid = 0;
  c10::cuda::memcpy_and_sync(&invalid, status.const_data_ptr<int>(), sizeof(invalid),
                            cudaMemcpyDeviceToHost, stream);
  TORCH_CHECK(!invalid, "tangents must be finite and within [-1000,1000]");
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

// The adaptive forward kernel. FORWARD_KERNEL.md gives the design and the
// study. pool_forward above stays selectable, and every derivative kernel
// below keeps one thread per group.
constexpr int kAdaptiveThreads = 256;
constexpr int kAdaptiveWarps = kAdaptiveThreads / 32;
constexpr int64_t kDefaultSwitchLength = 32;
constexpr int64_t kMaximumItems = 4096;
constexpr unsigned kFullWarp = 0xffffffffu;

// Butterfly reductions. Every lane ends with the same bits, because a + b
// and b + a round identically.
__device__ __forceinline__ double warp_maximum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const double other = __shfl_xor_sync(kFullWarp, value, offset);
    value = value > other ? value : other;
  }
  return value;
}

__device__ __forceinline__ double warp_total(double value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullWarp, value, offset);
  }
  return value;
}

// Lanes that share one row: the smallest power of two covering min(D, 32).
__device__ __forceinline__ int row_lanes(int64_t width) {
  int lanes = 1;
  while (lanes < 32 && lanes < width) lanes <<= 1;
  return lanes;
}

// Each lane keeps up to four column sums over the items of its slot, then the
// slots of this warp are combined. Every lane takes part in every shuffle.
template <typename scalar_t>
__device__ void accumulate_slots(const scalar_t* values, const double* weights,
                                 int64_t start, int64_t length, int64_t width,
                                 int lanes, int64_t first_slot, int64_t slot_stride,
                                 double partial[4]) {
  const int column = (threadIdx.x & 31) % lanes;
  for (int64_t i = first_slot; i < length; i += slot_stride) {
    const double weight = weights[i];
    const scalar_t* row = values + (start + i) * width;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int64_t d = column + static_cast<int64_t>(c) * lanes;
      if (d < width) partial[c] += weight * static_cast<double>(row[d]);
    }
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    for (int offset = lanes; offset < 32; offset <<= 1) {
      partial[c] += __shfl_xor_sync(kFullWarp, partial[c], offset);
    }
  }
}

template <typename scalar_t>
__device__ void pool_group_warp(const scalar_t* values, const scalar_t* logits,
                                int64_t start, int64_t length, int64_t width,
                                double* weights, scalar_t* output_row) {
  const int lane = threadIdx.x & 31;
  double maximum = -CUDART_INF;
  for (int64_t i = lane; i < length; i += 32) {
    const double value = static_cast<double>(logits[start + i]);
    maximum = maximum > value ? maximum : value;
  }
  maximum = warp_maximum(maximum);
  double denominator = 0.0;
  for (int64_t i = lane; i < length; i += 32) {
    const double unnormalized = exp(static_cast<double>(logits[start + i]) - maximum);
    weights[i] = unnormalized;
    denominator += unnormalized;
  }
  denominator = warp_total(denominator);
  // Each lane rescales only the entries it wrote, then the warp shares them.
  for (int64_t i = lane; i < length; i += 32) weights[i] /= denominator;
  __syncwarp();
  const int lanes = row_lanes(width);
  double partial[4] = {0.0, 0.0, 0.0, 0.0};
  accumulate_slots(values, weights, start, length, width, lanes, lane / lanes, 32 / lanes, partial);
  if (lane / lanes == 0) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int64_t d = lane + static_cast<int64_t>(c) * lanes;
      if (d < width) output_row[d] = static_cast<scalar_t>(partial[c]);
    }
  }
}

template <typename scalar_t>
__device__ void pool_group_block(const scalar_t* values, const scalar_t* logits,
                                 int64_t start, int64_t length, int64_t width,
                                 double* weights, double* partials, scalar_t* output_row) {
  __shared__ double warp_values[kAdaptiveWarps];
  const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
  double maximum = -CUDART_INF;
  for (int64_t i = thread; i < length; i += kAdaptiveThreads) {
    const double value = static_cast<double>(logits[start + i]);
    maximum = maximum > value ? maximum : value;
  }
  maximum = warp_maximum(maximum);
  if (lane == 0) warp_values[warp] = maximum;
  __syncthreads();
  maximum = warp_values[0];
  for (int w = 1; w < kAdaptiveWarps; ++w) {
    maximum = maximum > warp_values[w] ? maximum : warp_values[w];
  }
  __syncthreads();  // Every thread has read the maxima before the slots are reused.
  double denominator = 0.0;
  for (int64_t i = thread; i < length; i += kAdaptiveThreads) {
    const double unnormalized = exp(static_cast<double>(logits[start + i]) - maximum);
    weights[i] = unnormalized;
    denominator += unnormalized;
  }
  denominator = warp_total(denominator);
  if (lane == 0) warp_values[warp] = denominator;
  __syncthreads();
  denominator = warp_values[0];
  for (int w = 1; w < kAdaptiveWarps; ++w) denominator += warp_values[w];
  for (int64_t i = thread; i < length; i += kAdaptiveThreads) weights[i] /= denominator;
  __syncthreads();
  const int lanes = row_lanes(width);
  const int slots_per_warp = 32 / lanes;
  double partial[4] = {0.0, 0.0, 0.0, 0.0};
  accumulate_slots(values, weights, start, length, width, lanes,
                   static_cast<int64_t>(warp) * slots_per_warp + lane / lanes,
                   static_cast<int64_t>(kAdaptiveWarps) * slots_per_warp, partial);
  if (lane / lanes == 0) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int64_t d = lane + static_cast<int64_t>(c) * lanes;
      if (d < width) partials[warp * width + d] = partial[c];
    }
  }
  __syncthreads();
  for (int64_t d = thread; d < width; d += kAdaptiveThreads) {
    double total = partials[d];
    for (int w = 1; w < kAdaptiveWarps; ++w) total += partials[w * width + d];
    output_row[d] = static_cast<scalar_t>(total);
  }
}

// One block of 256 threads per group. The group's own length picks the path,
// so the branch is uniform across the block and the warp path never reaches a
// block barrier. Dynamic shared memory holds `capacity` weights, then 8 rows
// of `width` partial sums.
template <typename scalar_t>
__global__ void __launch_bounds__(kAdaptiveThreads) pool_forward_adaptive(
    const scalar_t* values, const scalar_t* logits, const int64_t* offsets,
    int64_t width, int64_t capacity, int64_t switch_length, scalar_t* output) {
  extern __shared__ double shared[];
  const int64_t group = blockIdx.x;
  const int64_t start = offsets[group], length = offsets[group + 1] - start;
  if (length == 0) return;  // The host already zeroed this row.
  scalar_t* row = output + group * width;
  if (length <= switch_length) {
    if (threadIdx.x >= 32) return;
    pool_group_warp(values, logits, start, length, width, shared, row);
  } else {
    pool_group_block(values, logits, start, length, width, shared, shared + capacity, row);
  }
}

// Keeps the stream busy while the host enqueues a block of timed launches,
// so no event pair contains time the GPU spent waiting for the host.
__global__ void hold_stream(unsigned long long nanoseconds) {
  unsigned long long begin = 0, now = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - begin < nanoseconds);
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

template <typename scalar_t>
__global__ void pool_jvp(const scalar_t* values, const scalar_t* logits,
                         const int64_t* offsets, const scalar_t* dvalues,
                         const scalar_t* dlogits, int64_t width,
                         scalar_t* tangent_output) {
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
  double tangent[128] = {};
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    const double dlogit = static_cast<double>(dlogits[i]);
    for (int64_t d = 0; d < width; ++d) {
      tangent[d] += weight * (static_cast<double>(dvalues[i * width + d]) +
          (static_cast<double>(values[i * width + d]) - pooled[d]) * dlogit);
    }
  }
  for (int64_t d = 0; d < width; ++d) {
    tangent_output[group * width + d] = static_cast<scalar_t>(tangent[d]);
  }
}

template <typename scalar_t>
__global__ void pool_double_backward(
    const scalar_t* value_upstream, const scalar_t* logit_upstream,
    const scalar_t* grad, const scalar_t* values, const scalar_t* logits,
    const int64_t* offsets, int64_t width, scalar_t* grad_grad_output,
    scalar_t* grad_values, scalar_t* grad_logits) {
  const int64_t group = blockIdx.x;
  const int64_t start = offsets[group], end = offsets[group + 1];
  if (start == end) return;
  if (end - start == 1) {
    // The singleton backward is dv=g, dl=0. Its only derivative is dg=p.
    for (int64_t d = 0; d < width; ++d) {
      grad_grad_output[group * width + d] = value_upstream[start * width + d];
    }
    return;
  }
  const double maximum = group_maximum(logits, start, end);
  const double denominator = group_denominator(logits, start, end, maximum);
  double pooled[128] = {};
  double combined[128] = {};
  double mean_logit_upstream = 0.0;
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    mean_logit_upstream += weight * static_cast<double>(logit_upstream[i]);
    for (int64_t d = 0; d < width; ++d) {
      pooled[d] += weight * static_cast<double>(values[i * width + d]);
      combined[d] += weight * static_cast<double>(value_upstream[i * width + d]);
    }
  }
  // For S=<p,dv>+<q,dl>, H=E[p]+E[(q-E[q])*(v-E[v])] is dS/dg.
  // Center both covariance factors before accumulating in float64.
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    const double centered_q = static_cast<double>(logit_upstream[i]) - mean_logit_upstream;
    for (int64_t d = 0; d < width; ++d) {
      combined[d] += weight * centered_q *
                     (static_cast<double>(values[i * width + d]) - pooled[d]);
    }
  }
  for (int64_t d = 0; d < width; ++d) {
    grad_grad_output[group * width + d] = static_cast<scalar_t>(combined[d]);
  }
  for (int64_t i = start; i < end; ++i) {
    const double weight = exp(static_cast<double>(logits[i]) - maximum) / denominator;
    const double centered_q = static_cast<double>(logit_upstream[i]) - mean_logit_upstream;
    double logit_dot = 0.0;
    for (int64_t d = 0; d < width; ++d) {
      const double upstream = static_cast<double>(grad[group * width + d]);
      const double centered_value = static_cast<double>(values[i * width + d]) - pooled[d];
      grad_values[i * width + d] = static_cast<scalar_t>(weight * centered_q * upstream);
      logit_dot += upstream * (static_cast<double>(value_upstream[i * width + d]) -
                              combined[d] + centered_q * centered_value);
    }
    grad_logits[i] = static_cast<scalar_t>(weight * logit_dot);
  }
}

constexpr int64_t kExistingForward = 0;
constexpr int64_t kAdaptiveForward = 1;

// Launch one forward kernel on admitted inputs into an output of zeros.
void launch_forward(const at::Tensor& values, const at::Tensor& logits,
                    const at::Tensor& offsets, at::Tensor& output, int64_t kernel,
                    int64_t switch_length, cudaStream_t stream) {
  const int64_t groups = offsets.numel() - 1, width = values.size(1);
  if (groups == 0) return;
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_forward", [&] {
    if (kernel == kExistingForward) {
      pool_forward<scalar_t><<<groups, 1, 0, stream>>>(
          values.const_data_ptr<scalar_t>(), logits.const_data_ptr<scalar_t>(),
          offsets.const_data_ptr<int64_t>(), width, output.mutable_data_ptr<scalar_t>());
    } else {
      // No group is longer than N, and admission keeps N <= 4096 and D <= 128,
      // so this stays under the 48 KiB a launch may request without opting in.
      const int64_t capacity = values.size(0);
      const size_t bytes = static_cast<size_t>(capacity + kAdaptiveWarps * width) * sizeof(double);
      pool_forward_adaptive<scalar_t><<<groups, kAdaptiveThreads, bytes, stream>>>(
          values.const_data_ptr<scalar_t>(), logits.const_data_ptr<scalar_t>(),
          offsets.const_data_ptr<int64_t>(), width, capacity, switch_length,
          output.mutable_data_ptr<scalar_t>());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// RAGGED_POOL_CUDA_FORWARD is read once, when the library loads.
const std::string kForwardChoice = [] {
  const char* choice = std::getenv("RAGGED_POOL_CUDA_FORWARD");
  return std::string(choice == nullptr ? "" : choice);
}();

int64_t public_forward_kernel() {
  if (kForwardChoice.empty() || kForwardChoice == "existing") return kExistingForward;
  TORCH_CHECK(kForwardChoice == "adaptive",
              "RAGGED_POOL_CUDA_FORWARD must be unset, existing or adaptive");
  return kAdaptiveForward;
}

at::Tensor forward_cuda(const at::Tensor& values, const at::Tensor& logits,
                        const at::Tensor& offsets) {
  const int64_t kernel = public_forward_kernel();
  check_metadata(values, logits, offsets);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  admit_contents(values, logits, offsets, nullptr, stream);
  auto output = at::zeros({offsets.numel() - 1, values.size(1)}, values.options());
  launch_forward(values, logits, offsets, output, kernel, kDefaultSwitchLength, stream);
  return output;
}

// Destroys the events it created, including after a failed check.
struct StudyEvents {
  std::vector<cudaEvent_t> events;
  StudyEvents() = default;
  StudyEvents(const StudyEvents&) = delete;
  StudyEvents& operator=(const StudyEvents&) = delete;
  ~StudyEvents() {
    for (cudaEvent_t event : events) cudaEventDestroy(event);
  }
  cudaEvent_t add() {
    cudaEvent_t event = nullptr;
    C10_CUDA_CHECK(cudaEventCreate(&event));
    events.push_back(event);
    return event;
  }
};

// The private study operation from FORWARD_KERNEL.md. With repeats > 0 the
// timing tensor holds the milliseconds of each launch from its own event pair,
// then one more entry for the span from the first start to the last stop.
std::tuple<at::Tensor, at::Tensor> forward_study_cuda(
    const at::Tensor& values, const at::Tensor& logits, const at::Tensor& offsets,
    int64_t kernel, int64_t switch_length, int64_t repeats) {
  TORCH_CHECK(kernel == kExistingForward || kernel == kAdaptiveForward,
              "kernel must be 0 for the existing kernel or 1 for the adaptive kernel");
  TORCH_CHECK(switch_length >= 0 && switch_length <= kMaximumItems,
              "switch_length must be in [0,4096]");
  // Each repeat queues three commands. Staying well under the driver's queue
  // keeps the host from blocking while the hold spins.
  TORCH_CHECK(repeats >= 0 && repeats <= 200, "repeats must be in [0,200]");
  check_metadata(values, logits, offsets);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  admit_contents(values, logits, offsets, nullptr, stream);
  auto output = at::zeros({offsets.numel() - 1, values.size(1)}, values.options());
  auto timing = at::empty({repeats == 0 ? 0 : repeats + 1}, at::TensorOptions().dtype(at::kDouble));
  if (repeats == 0) {
    launch_forward(values, logits, offsets, output, kernel, switch_length, stream);
    return {output, timing};
  }
  StudyEvents owned;
  const cudaEvent_t held = owned.add();
  std::vector<cudaEvent_t> starts, stops;
  for (int64_t r = 0; r < repeats; ++r) {
    starts.push_back(owned.add());
    stops.push_back(owned.add());
  }
  // Hold the stream until the host has enqueued every timed launch. If the
  // hold ended first, the GPU may have waited inside a pair, so hold longer
  // and run the block again. The output is the same each time.
  unsigned long long hold = 5000000ull;
  for (int attempt = 0;; ++attempt) {
    hold_stream<<<1, 1, 0, stream>>>(hold);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(held, stream));
    for (int64_t r = 0; r < repeats; ++r) {
      C10_CUDA_CHECK(cudaEventRecord(starts[r], stream));
      launch_forward(values, logits, offsets, output, kernel, switch_length, stream);
      C10_CUDA_CHECK(cudaEventRecord(stops[r], stream));
    }
    const cudaError_t state = cudaEventQuery(held);
    if (state == cudaErrorNotReady) {
      (void)cudaGetLastError();  // Not ready is a state, not a failure.
      break;
    }
    C10_CUDA_CHECK(state);
    TORCH_CHECK(attempt < 4, "the stream hold ended before the host enqueued the timed launches");
    hold *= 4;
  }
  C10_CUDA_CHECK(cudaEventSynchronize(stops.back()));
  auto* times = timing.mutable_data_ptr<double>();
  for (int64_t r = 0; r < repeats; ++r) {
    float milliseconds = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, starts[r], stops[r]));
    times[r] = milliseconds;
  }
  float span = 0.0f;
  C10_CUDA_CHECK(cudaEventElapsedTime(&span, starts.front(), stops.back()));
  times[repeats] = span;
  return {output, timing};
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

at::Tensor jvp_cuda(const at::Tensor& values, const at::Tensor& logits,
                    const at::Tensor& offsets, const at::Tensor& dvalues,
                    const at::Tensor& dlogits) {
  check_metadata(values, logits, offsets);
  check_tangent_metadata(dvalues, values);
  check_tangent_metadata(dlogits, logits);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  const auto contiguous_dvalues = dvalues.resolve_neg().contiguous();
  const auto contiguous_dlogits = dlogits.resolve_neg().contiguous();
  admit_contents(values, logits, offsets, nullptr, stream);
  admit_tangents(contiguous_dvalues, contiguous_dlogits, stream);
  const int64_t groups = offsets.numel() - 1;
  auto output = at::zeros({groups, values.size(1)}, values.options());
  if (groups > 0) {
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_jvp", [&] {
      pool_jvp<scalar_t><<<groups, 1, 0, stream>>>(
          values.const_data_ptr<scalar_t>(), logits.const_data_ptr<scalar_t>(),
          offsets.const_data_ptr<int64_t>(), contiguous_dvalues.const_data_ptr<scalar_t>(),
          contiguous_dlogits.const_data_ptr<scalar_t>(), values.size(1),
          output.mutable_data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> double_backward_cuda(
    const at::Tensor& grad_grad_values, const at::Tensor& grad_grad_logits,
    const at::Tensor& grad_output, const at::Tensor& values,
    const at::Tensor& logits, const at::Tensor& offsets) {
  check_metadata(values, logits, offsets);
  check_grad_metadata(grad_output, values, offsets);
  check_tangent_metadata(grad_grad_values, values);
  check_tangent_metadata(grad_grad_logits, logits);
  const c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  // Only derivative inputs are normalized. Admission sees the original primals.
  const auto value_upstream = grad_grad_values.resolve_neg().contiguous();
  const auto logit_upstream = grad_grad_logits.resolve_neg().contiguous();
  const auto grad = grad_output.resolve_neg().contiguous();
  admit_contents(values, logits, offsets, &grad, stream);
  admit_tangents(value_upstream, logit_upstream, stream);
  const int64_t groups = offsets.numel() - 1;
  auto grad_grad_output = at::zeros({groups, values.size(1)}, values.options());
  auto grad_values = at::zeros_like(values);
  auto grad_logits = at::zeros_like(logits);
  if (groups > 0) {
    AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "ragged_cuda_double_backward", [&] {
      pool_double_backward<scalar_t><<<groups, 1, 0, stream>>>(
          value_upstream.const_data_ptr<scalar_t>(), logit_upstream.const_data_ptr<scalar_t>(),
          grad.const_data_ptr<scalar_t>(), values.const_data_ptr<scalar_t>(),
          logits.const_data_ptr<scalar_t>(), offsets.const_data_ptr<int64_t>(), values.size(1),
          grad_grad_output.mutable_data_ptr<scalar_t>(), grad_values.mutable_data_ptr<scalar_t>(),
          grad_logits.mutable_data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_grad_output, grad_values, grad_logits};
}

}  // namespace

TORCH_LIBRARY(ragged_pool_cuda, library) {
  library.def("pool(Tensor values, Tensor logits, Tensor offsets) -> Tensor");
  library.def("_backward(Tensor grad_output, Tensor values, Tensor logits, Tensor offsets) -> (Tensor, Tensor)");
  library.def("_jvp(Tensor values, Tensor logits, Tensor offsets, Tensor dvalues, Tensor dlogits) -> Tensor");
  library.def("_double_backward(Tensor grad_grad_values, Tensor grad_grad_logits, Tensor grad_output, Tensor values, Tensor logits, Tensor offsets) -> (Tensor, Tensor, Tensor)");
  library.def("_forward_study(Tensor values, Tensor logits, Tensor offsets, int kernel, int switch_length, int repeats) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(ragged_pool_cuda, CUDA, library) {
  library.impl("pool", forward_cuda);
  library.impl("_backward", backward_cuda);
  library.impl("_jvp", jvp_cuda);
  library.impl("_double_backward", double_backward_cuda);
  library.impl("_forward_study", forward_study_cuda);
}
// Inspect original primals before a generic Negative fallback could copy them.
// Backward intentionally resolves only its admitted logical upstream gradient.
TORCH_LIBRARY_IMPL(ragged_pool_cuda, Negative, library) {
  library.impl("pool", forward_cuda);
  library.impl("_backward", backward_cuda);
  library.impl("_jvp", jvp_cuda);
  library.impl("_double_backward", double_backward_cuda);
  library.impl("_forward_study", forward_study_cuda);
}

// Setuptools exports this CPython symbol on Windows. Tensor calls use the
// dispatcher registrations above, without a separate pybind11 interface.
PyMODINIT_FUNC PyInit__C() {
  static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module);
}
