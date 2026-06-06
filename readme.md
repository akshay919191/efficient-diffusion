# Diffusion From Scratch – Flickr8k + Custom CUDA Kernels

A small, from-scratch text-conditioned diffusion model built to test and
benchmark two custom CUDA kernels: a fused FlashAttention and an online
Welford algorithm for normalisation layers. The code is purposely minimal
and uses the Flickr8k dataset (~8k images) so that training cycles are fast
and kernel behaviour can be probed in a realistic setting.

## What's Inside
- A U-Net with cross-attention (text conditioning) — trained from scratch
- A VAE for latent-space encoding — trained from scratch
- DDPM / DDIM sampler
- Two hand-written CUDA kernels loaded via `torch.utils.cpp_extension`:
  - `flashattn` – fused multi-head attention (forward + backward)
  - `welford` – numerically stable online mean/variance for LayerNorm/GroupNorm
- A benchmark harness that profiles latency and GPU memory per kernel

## disclaimer - my kernels are hardcoded for some specific dims acc to my needs you can adjust

## Motivation
This is not a model for pretty pictures. It's a playground to compare
custom kernel implementations against PyTorch's compiled backend. Flickr8k
is perfect: diverse captions, tiny footprint, trains in a couple of days
on a single consumer GPU.

## Model Parameters(reason for this much big models for small dataset is to check speed nothing else)

| Component      | Parameters | Status          |
|---------------|-----------|-----------------|
| VAE            | 37.41M    | Trained from scratch |
| UNET           | 50.79M    | Trained from scratch |
| CLIP           | ~63M      | Frozen          |
| **Trainable**  | **88.20M**|                 |
| **Total**      | **~151M** |                 |

## Training Journey
- **VAE**: ~31 hours on RTX 3050 6GB (37.41M parameters)
- **UNET**: ~6.5 day(still going) on RTX 3050 6GB (50.79M parameters)

## Generated Samples
Trained on Flickr8k (8k images). Results show correct scene understanding
and text alignment despite limited training data and compute.


## Folder Structure

```
diffusion-kernel-bench/
├── configs/
│   └── base.yaml
├── kernels/
│   ├── flashattn/
│   │   ├── flashattn.cu
│   │   └── flashattn.h
│   ├── welford/
│   │   ├── welford.cu
│   │   └── welford.h
│   ├── setup.py
│   └── __init__.py
├── models/
│   ├── vae_unet.py
│   ├── attn_mech.py
│   └── clip_text.py
├── scripts/
│   ├── data_alinging.py
│   ├── ddpm_ddim.py
│   └── pipeline.py
├── benchmarks/
├── inferencePIPELINE.py
├── training.py
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n diffkern python=3.10 -y
conda activate diffkern

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

cd kernels && python setup.py build_ext --inplace && cd ..
```

## Usage

**Training:**
```bash
python training.py
```

**Inference:**
```bash
python inferencePIPELINE.py --prompt "a dog running on a beach"
```

**Benchmark a kernel:**
```bash
python benchmarks/run_kernels.py --kernel welford
python benchmarks/run_kernels.py --kernel flashattn
```

## Benchmarks
Coming soon — benchmarked on RTX 3050 6GB, CUDA 12.1, PyTorch 2.2.0.

| Operation | PyTorch/cuDNN | Custom Kernel | Speedup |
|-----------|--------------|---------------|---------|
| Welford mean/var | — | — | — |
| Self-Attention | — | — | — |
| Cross-Attention | — | — | — |
| Full UNet forward | — | — | — |

## Notes on Code Quality
The code is intentionally kept simple and readable. Explicit branches in
the model forward methods show exactly which PyTorch ops are being replaced
by custom kernels. Production code would use cleaner dispatch, but that
would obscure the kernel insertion points.

## License
MIT — use it however you like. Open an issue if you find bugs in the kernels.

## Contact
GitHub: [akshay919191](https://github.com/akshay919191)