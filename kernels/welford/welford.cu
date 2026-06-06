#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

struct WelfordWarp {
    int n;
    float mean;
    float M2;
};

__device__ __forceinline__ WelfordWarp merge_warp(WelfordWarp a, WelfordWarp b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordWarp out;
    out.n = a.n + b.n;
    float delta = b.mean - a.mean;
    out.mean = a.mean + delta * ((float)b.n / out.n);
    out.M2   = a.M2 + b.M2 + delta * delta * ((float)a.n * b.n / out.n);
    return out;
}

__global__ void WELFORD_ATTN_KERNEL(
    const half* __restrict__ A,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int total_rows,
    int D
) {
    int warp_in_block = threadIdx.x >> 5; // threadIdx.x / 32
    int lane = threadIdx.x & 31;          // threadIdx.x % 32
    
    int row = (blockIdx.x * (blockDim.x >> 5)) + warp_in_block;

    if (row >= total_rows) return;

    const half* row_ptr = A + row * D;
    WelfordWarp local = {0, 0.f, 0.f};

    #pragma unroll
    for (int i = lane; i < D; i += 32) {
        float x = __half2float(row_ptr[i]);
        local.n++;
        float delta = x - local.mean;
        local.mean += delta / local.n;
        local.M2 += delta * (x - local.mean);
    }

    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordWarp other;
        other.n    = __shfl_down_sync(FULL_MASK, local.n, offset);
        other.mean = __shfl_down_sync(FULL_MASK, local.mean, offset);
        other.M2   = __shfl_down_sync(FULL_MASK, local.M2, offset);
        local = merge_warp(local, other);
    }

    if (lane == 0) {
        mean_out[row] = local.mean;
        var_out[row]  = (local.n > 0) ? (local.M2 / local.n) : 0.f;
    }
}

void launch_welford_kernel(const torch::Tensor& A, torch::Tensor& mean_out, torch::Tensor& var_out) {
    int total_rows = A.size(0);
    int D = A.size(1);

    int threads_per_block = 256; 
    int warps_per_block = threads_per_block / 32; // 8 rows per block
    
    // Round up grid sizing
    int grid_size = (total_rows + warps_per_block - 1) / warps_per_block;

    WELFORD_ATTN_KERNEL<<<grid_size, threads_per_block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        mean_out.data_ptr<float>(),
        var_out.data_ptr<float>(),
        total_rows,
        D
    );
}