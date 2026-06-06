#ifndef WELFORD_H_
#define WELFORD_H_

#include <cuda.h>


struct Welford {
    int n;       // Count of accumulated elements
    float mean;  // Running mean
    float M2;    // Running sum of squares of differences from the mean
};

__device__ __forceinline__
void update_welford(Welford &local, float x) {
    local.n++;
    float delta = x - local.mean;
    local.mean += delta / local.n;
    float delta2 = x - local.mean;
    local.M2 += delta * delta2;
}

__device__ __forceinline__
Welford merge(const Welford &a, const Welford &b)
{
    if (a.n == 0) return b;
    if (b.n == 0) return a;

    Welford out;
    out.n = a.n + b.n;

    float delta = b.mean - a.mean;
    out.mean = a.mean + delta * ((float)b.n / out.n);
    out.M2   = a.M2 + b.M2 + delta * delta * ((float)a.n * b.n / out.n);

    return out;
}

__device__ __forceinline__
Welford warp_reduce(Welford val)
{
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        Welford other;
        other.n    = __shfl_down_sync(FULL_MASK, val.n, offset);
        other.mean = __shfl_down_sync(FULL_MASK, val.mean, offset);
        other.M2   = __shfl_down_sync(FULL_MASK, val.M2, offset);
        
        val = merge(val, other);
    }
    return val;
}

#endif // WELFORD_H_