import torch
import torch.nn as nn

def ddpm_scheduler(T=1000, device="cpu"):
    betas = torch.linspace(1e-4, 0.02, T, device=device)

    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    return {
        "beta": betas,
        "alpha": alphas,
        "alpha_hat": alpha_hat
    }

scheduler = ddpm_scheduler(Time = 1000 , device = None)

def noiseADD(x0 , noise , t):
    device = x0.device

    s1 = torch.sqrt(scheduler["alpha_hat"].to(device)[t]).view(-1 , 1 , 1 , 1)
    s2 = scheduler["rooted_one_minus"].to(device)[t].view(-1 , 1 , 1 , 1)

    return s1 * x0 + s2 * noise , noise

def ddpm_step(x_t, pred_noise, t, scheduler):
    device = x_t.device

    betas = scheduler["beta"].to(device)
    alphas = scheduler["alpha"].to(device)
    alpha_hat = scheduler["alpha_hat"].to(device)

    beta_t = betas[t].view(-1,1,1,1)
    alpha_t = alphas[t].view(-1,1,1,1)
    alpha_hat_t = alpha_hat[t].view(-1,1,1,1)

    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1 - alpha_hat_t)) * pred_noise
    )

    noise = torch.randn_like(x_t)

    mask = (t != 0).float().view(-1,1,1,1)

    x_prev = mean + mask * torch.sqrt(beta_t) * noise

    return x_prev