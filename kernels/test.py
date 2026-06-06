import sys
sys.path.insert(0, "/home/toxiccccterabaap/Downloads/EFFICIENT DIFFUSION/kernels")

import torch
import torch.nn.functional as F
import custom_flash_attn

def check_all_gradients(batch=1, heads=1, seq_len=64, headdim=64, seed=42):
    torch.manual_seed(seed)
    device = 'cuda'
    dtype  = torch.float16

    Q = torch.randn(batch, heads, seq_len, headdim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, headdim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, headdim, device=device, dtype=dtype)
    dL_dout = torch.randn(batch, heads, seq_len, headdim, device=device, dtype=dtype)

    Qr = Q.clone().requires_grad_(True)
    Kr = K.clone().requires_grad_(True)
    Vr = V.clone().requires_grad_(True)

    scale     = headdim ** -0.5
    score_raw = Qr @ Kr.transpose(-2, -1)
    P         = F.softmax(score_raw * scale, dim=-1)
    out_ref   = P @ Vr
    out_ref.backward(dL_dout)

    dQ_ref = Qr.grad.clone().detach()
    dK_ref = Kr.grad.clone().detach()
    dV_ref = Vr.grad.clone().detach()

    cuda_out, cuda_L = custom_flash_attn.forward(Q, K, V)
    cuda_dQ, cuda_dK, cuda_dV , _ , _ = custom_flash_attn.backward(Q, K, V, cuda_out, cuda_L, dL_dout)

    def report(name, cuda_val, ref_val):
        err = (cuda_val.float() - ref_val.float()).abs()
        ok  = torch.allclose(cuda_val.float(), ref_val.float(), atol=1e-2)
        print(f"  {name:<10} | max {err.max():.6f} | mean {err.mean():.6f} | {'✓' if ok else '✗'}")

    print(f"{'='*55}")
    print(f"seq_len={seq_len}  headdim={headdim}  batch={batch}  heads={heads}")
    print(f"{'='*55}")
    report("dQ", cuda_dQ, dQ_ref)
    report("dK", cuda_dK, dK_ref)
    report("dV", cuda_dV, dV_ref)
    print(f"{'='*55}")

    # after getting cuda_dK, check which rows are wrong
    err = (cuda_dK.float() - dK_ref.float()).abs()
    print("dK error per row:", err[0,0].mean(dim=-1))  # mean error per seq position
    print("dK error first half rows:", err[0,0,:64].mean())
    print("dK error second half rows:", err[0,0,64:].mean())


check_all_gradients(batch=2, heads=8, seq_len=64,  headdim=64)
check_all_gradients(batch=2, heads=8, seq_len=128,  headdim=64)
check_all_gradients(batch=1, heads=8, seq_len=64,  headdim=64)
check_all_gradients(batch=1, heads=8, seq_len=128, headdim=64)
check_all_gradients(batch=2, heads=8, seq_len=64,  headdim=64)
check_all_gradients(batch=2, heads=8, seq_len=128, headdim=64)
check_all_gradients(batch=4, heads=8, seq_len=256, headdim=64)