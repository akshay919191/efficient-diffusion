import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from attn_mech import AttnWrapper


class UNET(nn.Module):
    def __init__(self, input_dim, timedim):
        super().__init__()

        self.time_dim = timedim
        self.label_emb = nn.Embedding(11, timedim)


        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )

        self.down1 = nn.MaxPool2d(2)

        self.timemlp1 = nn.Sequential(
            nn.Linear(timedim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
        )

        self.down2 = nn.MaxPool2d(2)

        self.timemlp2 = nn.Sequential(
            nn.Linear(timedim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )


        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.SiLU(),
            AttnWrapper(512, 8)
        )

        self.timemlp3 = nn.Sequential(
            nn.Linear(timedim, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )


        self.up1 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=2,
            stride=2
        )

        self.up1_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU()
        )

        self.timemlp4 = nn.Sequential(
            nn.Linear(timedim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.up2 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )

        self.up2_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU()
        )

        self.timemlp5 = nn.Sequential(
            nn.Linear(timedim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.final = nn.Conv2d(
            64,
            input_dim,
            kernel_size=1
        )

    def forward(self, x, t, y):

        t_emb = get_time_embedding(
            t,
            self.time_dim,
            device=t.device
        )

        label_emb = self.label_emb(y)
        time_embed = t_emb + label_emb

        # time embeddings
        time1 = self.timemlp1(time_embed)[:, :, None, None]
        time2 = self.timemlp2(time_embed)[:, :, None, None]
        time3 = self.timemlp3(time_embed)[:, :, None, None]
        time4 = self.timemlp4(time_embed)[:, :, None, None]
        time5 = self.timemlp5(time_embed)[:, :, None, None]


        enc1 = self.enc_conv1(x) + time1
        x = self.down1(enc1)

        enc2 = self.enc_conv2(x) + time2
        x = self.down2(enc2)


        bottle = self.bottleneck(x) + time3


        dec1 = self.up1(bottle) + time4
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.up1_conv(dec1)

        dec2 = self.up2(dec1) + time5
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.up2_conv(dec2)

        out = self.final(dec2)

        return out


def get_time_embedding(timesteps, embedding_dim, device=None):

    if device is None:
        device = timesteps.device

    half_dim = embedding_dim // 2

    emb_scale = math.log(10000) / (half_dim - 1)

    emb = torch.exp(
        torch.arange(half_dim, device=device) * -emb_scale
    )

    emb = timesteps[:, None].float() * emb[None, :]

    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

    return emb