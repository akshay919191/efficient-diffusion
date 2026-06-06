# Custom Flash Attention — CUDA Kernel (Forward + Backward)

Hand-written Flash Attention forward and backward pass in CUDA using raw
`mma.sync.aligned.m16n8k16` PTX instructions. No Triton, no cuBLAS, no cuDNN.

---

## What's implemented

- **Forward pass** — tiled softmax attention with online logsumexp, writes output and LSE statistics
- **Backward pass** — full dQ, dK, dV gradients via two kernels:
  - `backwardFlashAttention_DQ` — computes dQ and saves softmax probabilities + dL/dS as intermediates
  - `backwardFlashAttention_DK_DV` — computes dK and dV from saved intermediates

---

## Validation results

Tested against PyTorch reference (`F.softmax` + autograd) across multiple configurations.
All gradients pass `allclose(atol=1e-2)`. Observed errors are fp16 rounding noise only (~1e-4 to 2e-3).

| batch | heads | seq_len | headdim | dQ | dK | dV |
|-------|-------|---------|---------|-----|-----|-----|
| 1     | 1     | 64      | 64      | ✓  | ✓  | ✓  |
| 1     | 1     | 128     | 64      | ✓  | ✓  | ✓  |
| 1     | 8     | 64      | 64      | ✓  | ✓  | ✓  |
| 1     | 8     | 128     | 64      | ✓  | ✓  | ✓  |
| 2     | 8     | 64      | 64      | ✓  | ✓  | ✓  |
| 2     | 8     | 128     | 64      | ✓  | ✓  | ✓  |
| 4     | 8     | 256     | 64      | ✓  | ✓  | ✓  |

Errors are uniform across all rows and well within fp16 precision limits.
No NaNs, no out-of-bounds writes, no race conditions.

---

## Disclaimer / Known Limitations

### NUM_HEADS is hardcoded at launch time

`NUM_HEADS=8` is a compile-time template parameter used for computing batch/head
memory strides. The kernel will produce **wrong results silently** if launched
with a different number of heads.

```cpp
// current — hardcoded
backwardFlashAttention_DQ<32, 64, 8><<<...>>>(...)
//                                 ^ NUM_HEADS=8 baked in
```

**To support arbitrary head counts**, replace all `NUM_HEADS` stride calculations
with a runtime `int num_heads` argument:

```cpp
// change kernel signature
__global__ void backwardFlashAttention_DQ(
    ...,
    int seq_len, int headdim, int num_heads   // add num_heads here
)

// replace all NUM_HEADS usages
const long long base = (long long)batchid * num_heads * seq_len * headdim +
                        (long long)headid  * seq_len * headdim;

// pass from wrapper
backwardFlashAttention_DQ<<<...>>>(..., static_cast<int>(h));
```

Same change applies to `backwardFlashAttention_DK_DV`.
This is a one-time refactor with no performance cost.

### Other fixed constraints

- `headdim` must be 64 (separate kernel instantiation exists for 32)
- `Br=64` for DK/DV kernel, `Br=32` for DQ kernel — tile sizes are compile-time
- `seq_len` must be divisible by the tile size (64 for DK/DV, 32 for DQ)
- Requires sm_80+ for `mma.m16n8k16` and `atomicAdd` on `__half`
- Tested on sm_86 (RTX 30 series) with CUDA 11.8

### No causal masking

The current implementation is non-causal (full attention). Causal masking would
require masking the score tile before softmax in the forward pass and zeroing the
corresponding dL/dS entries in the backward pass.

---

## Hardware

- GPU: NVIDIA sm_86 (tested on RTX 30 series)
- CUDA: 11.8
- dtype: fp16 throughout (accumulation in fp32 via mma)
