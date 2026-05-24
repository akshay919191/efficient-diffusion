import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from models.attn_mech import AttnWrapper , CrossAttentionBlock

def get_groups(channels):
    if channels % 8 == 0:
        return 8
    elif channels % 4 == 0:
        return 4
    else:
        return 1

class RESIDUAL(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel


        self.grp1 = nn.GroupNorm(
        num_groups=get_groups(in_channel),
        num_channels=in_channel
        )
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        # Time projection mapping time_dim to out_channel
        self.timeproj = nn.Linear(time_dim, out_channel)

        self.grp2 = nn.GroupNorm(
            num_groups=get_groups(out_channel),
            num_channels=out_channel
        )
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.grp1(x)))
        
        time_emb = self.timeproj(self.act1(t_emb))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(self.act2(self.grp2(h)))
        return h + self.shortcut(x)


class UNET(nn.Module):
    def __init__(self, input_dim, timedim , textDIM = 512):
        super().__init__()
        self.time_dim = timedim

        self.time_mlp = nn.Sequential(
            nn.Linear(timedim, timedim),
            nn.SiLU(),
            nn.Linear(timedim, timedim)
        )

        self.enc1 = RESIDUAL(input_dim, 128, timedim)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = RESIDUAL(128, 256, timedim)
        self.down2 = nn.MaxPool2d(2)

        self.bottleneck_res = RESIDUAL(256, 512, timedim)
        self.attn = AttnWrapper(512, 8)
        self.cross_attn = CrossAttentionBlock(channels=512, text_dim=textDIM, num_heads=8)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = RESIDUAL(512, 256, timedim) 

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = RESIDUAL(256, 64, timedim)

        self.final = nn.Conv2d(64, input_dim, kernel_size=1)

    def forward(self, x, t, text_embeddings):
        t_emb = get_time_embedding(t, self.time_dim, device=t.device)
        t_emb = self.time_mlp(t_emb)

        s1 = self.enc1(x, t_emb)
        x = self.down1(s1)

        s2 = self.enc2(x, t_emb)
        x = self.down2(s2)

        x = self.bottleneck_res(x, t_emb)
        x = self.attn(x)
        x = self.cross_attn(x, text_embeddings)

        x = self.up1(x)
        x = torch.cat([x, s2], dim=1) # Concatenate skip connection
        x = self.dec1(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, s1], dim=1) # Concatenate skip connection
        x = self.dec2(x, t_emb)

        return self.final(x)


def get_time_embedding(timesteps, embedding_dim, device=None):
    if device is None:
        device = timesteps.device
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return emb


class RES(nn.Module):
    """Standard Residual Block for VAE (no time component)."""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.grp1  = nn.GroupNorm(num_groups=8, num_channels=in_channel)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)

        self.grp2  = nn.GroupNorm(num_groups=8, num_channels=out_channel)
        self.act2  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(self.act1(self.grp1(x)))
        h = self.conv2(self.act2(self.grp2(h)))
        return h + self.shortcut(x)


import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, inchannel, latentdim):
        super().__init__()
        self.latentdim = latentdim

        self.encodeblock1 = nn.Sequential(
            nn.Conv2d(inchannel, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            RES(128, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # Downsample 1
        )

        self.encodeblock2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            RES(256, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # Downsample 2
        )

        self.encodeblock3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.SiLU(),
            RES(512, 512),
            nn.SiLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1) # Downsample 3 -> 1024 channels
        )

        self.to_latent = nn.Conv2d(1024, latentdim, kernel_size=1)

        self.mu = nn.Conv2d(latentdim, latentdim, kernel_size=1)
        self.logvar = nn.Conv2d(latentdim, latentdim, kernel_size=1)

        self.from_latent = nn.Conv2d(latentdim, 1024, kernel_size=1)
        
        # 2. Upsample stages
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.res1 = RES(512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.res2 = RES(256, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.res3 = RES(128, 128)

        self.final = nn.Conv2d(128, inchannel, kernel_size=3, padding=1)

    def encode(self, x):
        features = self.encodeblock3(
            self.encodeblock2(
                self.encodeblock1(x)
            )
        )
        latent_features = self.to_latent(features)
        mu = self.mu(latent_features)
        logvar = self.logvar(latent_features)
        logvar = torch.clamp(logvar, -10, 4)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, x):
        x = self.from_latent(x)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        return self.final(x)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        features = self.encodeblock3(
            self.encodeblock2(
                self.encodeblock1(x)
            )
        )
        latent_features = self.to_latent(features)

        mu = self.mu(latent_features)
        logvar = self.logvar(latent_features)
        logvar = torch.clamp(logvar, -10, 4)

        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar
    
def vae_loss(x, recon_logits, mu, logvar, kl_weight=1e-4):

    x = (x + 1.0) * 0.5  # [-1,1] → [0,1]

    recon = F.mse_loss(torch.sigmoid(recon_logits), x, reduction="mean")

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    

    return recon + kl_weight * kl, recon, kl