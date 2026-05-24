DIFFUSION FROM SCRATCH – FLICKR8K + CUSTOM CUDA KERNELS

A small, from‑scratch text‑conditioned diffusion model built to test and
benchmark two custom CUDA kernels: a fused FlashAttention and an online
Welford algorithm for normalisation layers. The code is purposely minimal
and uses the Flickr8k dataset (~8k images) so that training cycles are fast
and kernel behaviour can be probed in a realistic setting.

WHAT’S INSIDE
  - A classic U‑Net with cross‑attention (text conditioning).
  - A simple VAE for latent‑space work (optional).
  - DDPM / DDIM sampler.
  - Two hand‑written CUDA kernels loaded via torch.utils.cpp_extension:
      * flashattn – fused multi‑head attention (forward + backward)
      * welford  – numerically stable mean/variance for LayerNorm/GroupNorm
  - A benchmark harness that profiles latency, GPU memory, and arithmetic
    intensity for each kernel and for the full training/inference pipeline.

MOTIVATION
  This is not a model for pretty pictures. It’s a playground where I can
  compare kernel implementations against cuDNN and Torch’s own compiled
  backend. I needed a real neural net that was small enough to iterate
  quickly but complex enough to exercise attention and normalisation ops
  end‑to‑end. Flickr8k is perfect: plenty of diverse captions, tiny
  footprint, and it trains in couple of days on a single GPU.

## Training Journey
- VAE: trained from scratch, ~31 hours on {RTX 3050}(37M parameters)
- UNET: trained from scratch, ~ 16- hours on {RTX 3050}(10.7M parameter)(sufficient to test kernels)

| Component     | Parameters | Trained |
|--------------|-----------|---------|
| VAE          | 37.41M    | From scratch |
| UNET         | 10.33M    | From scratch |
| CLIP         | ~63M      | Frozen |
| **Trainable**| **47.74M**| |
| **Total**    | **~110M** | |

## Generated Samples
Trained on Flickr8k (8k images). Results show correct scene understanding 
and text alignment despite limited training data and compute.

FOLDER STRUCTURE

  diffusion-kernel-bench/
  |
  |-- configs/
  |   `-- base.yaml                # all knobs (model size, kernel flags, etc.)
  |
  |-- data/
  |   `-- flickr8k/                # downloaded & pre‑processed images + captions
  |
  |-- kernels/
  |   |-- flashattn/
  |   |   |-- flashattn.cu         # forward / backward fused kernel
  |   |   `-- flashattn.h
  |   |-- welford/
  |   |   |-- welford.cu           # online mean/var + affine
  |   |   `-- welford.h
  |   |-- setup.py                # build script (python setup.py build_ext --inplace)
  |   `-- __init__.py             # Python wrapper that loads the compiled modules
  |
  |-- models/
  |   |-- unet.py                 # U‑Net definition (ResBlocks + Attention)
  |   |-- vae.py                  # encoder / decoder
  |   |-- attention.py            # pluggable attention: calls flashattn or pytorch
  |   |-- norms.py                # LayerNorm/GroupNorm using welford when requested
  |   |-- text_encoder.py         # a tiny CLIP‑like transformer for captions
  |   `-- diffusion.py            # noise schedule, forward diffusion, losses
  |
  |-- benchmarks/
  |   |-- run_kernels.py          # micro‑benchmark individual kernels
  |   `-- profiler_utils.py       # helper to capture memory & timing traces
  |
  |-- scripts/
  |   `-- dataaligning.py     # download & tokenise dataset
  |
  |-- train.py                    # main training loop with profiling hooks
  |-- inference.py                # generate images; compare kernel/nonkernel modes
  |-- requirements.txt
  `-- README.md                   # you are reading it


USAGE

  Training with kernel profiling

      $ python train.py \
          --use_flashattn \
          --use_welford \
          --profile_memory \
          --log_interval 100

      This trains the U‑Net on latent representations (default is 128x128
      latent space from a tiny VAE). Every 100 steps it dumps a breakdown of
      time and memory per layer into logs/timings.json. The same data is
      also printed to stdout.

  Inference with side‑by‑side kernel comparison

      $ python inference.py \
          --prompt "a dog running on a beach" \
          --samples 4 \
          --compare_kernels

      This generates four images using the baseline PyTorch ops and then
      again using the custom kernels, printing the total forward time, peak
      memory, and saving the images into outputs/. A small text file
      outputs/comparison.txt records the numbers.

  Micro‑benchmark a single kernel

      $ python benchmarks/run_kernels.py \
          --kernel flashattn \
          --batch 16 \
          --seq_len 256 \
          --head_dim 64 \
          --iters 1000

      Runs a parameter sweep over different shapes if you omit the shape
      arguments. Results are printed as a table with FLOPS and bandwidth
      utilisation.


BENCHMARK EXAMPLES(coming soon)
  (averaged over 1000 iterations, RTX 3050 6gb, CUDA 12.1, torch 2.2.0)

  Operation                       PyTorch/cuDNN    Custom Kernel   Speedup
  --------------------------------------------------------------------------
  Self‑Attention (16 × 64 × 1024)       ms           ms        ×
  Cross‑Attention (txt_len=77)          ms           ms        ×
  GroupNorm (32 channels, 128×128)      ms           ms        ×
  Full UNet fwd (single noise step)     ms            ms        ×

  Peak GPU memory during training (batch=8, 32 *32 latents):
      Standard implementation:   GB
      Custom kernels enabled:   GB   (% reduction)

  The Welford kernel avoids saving large intermediates for mean/variance
  computation and the FlashAttention kernel never materialises the full
  S×S attention matrix, which together give the memory saving.


NOTES ON CODE QUALITY
  The code is intentionally kept simple and a bit repetitive – readability
  beats cleverness here. I want to see exactly what PyTorch ops are being
  replaced, so you’ll find explicit branches in the model forward methods
  that select between native and custom paths. Production code would use a
  cleaner dispatch, but that would obscure the kernel’s insertion points.

  The VAE is extremely small (only ~1.2M params) because it only serves to
  provide a latent space; don’t expect amazing reconstructions.

LICENSE
  MIT – use it however you like. If you find bugs in the kernels, open an
  issue. I’m happy to accept improvements.

CONTACT
  Just an anonymous kernel tinkerer – use the GitHub issue tracker.