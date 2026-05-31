import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.vae_unet import UNET , VAE 
from scripts.data_alinging import trainloader
from models.clip_text import text_encoder , tokenizer


LATENTDIM = 4
INCHANNEL = 3
TIMEDIM   = 256
DEVICE = 'cuda'
custom_scaling_factor = 1.0 # placeholder , when training use 1 / mean of std dev as a scaling factor 

vae  = VAE(INCHANNEL  , LATENTDIM)
unet = UNET(LATENTDIM , TIMEDIM).to(DEVICE)
vae = vae.to(DEVICE)

text_encoder.eval()
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

vae.to(DEVICE)
text_encoder.to(DEVICE)
unet.to(DEVICE)

optimizer1 = torch.optim.AdamW(unet.parameters() , lr = 1e-4 , weight_decay = 1e-2)
optimizer2 = torch.optim.AdamW(vae.parameters() , lr = 1e-4)

def get_custom_vae_latents(vae_model, clean_images):

    features = vae_model.encodeblock3(
        vae_model.encodeblock2(
            vae_model.encodeblock1(clean_images)
        )
    )
    latent_features = vae_model.to_latent(features)
    
    mu = vae_model.mu(latent_features)
    logvar = vae_model.logvar(latent_features)
    logvar = torch.clamp(logvar, -4, 2)
    
    return mu

empty_tokens = tokenizer(
    "",
    padding="max_length",
    truncation=True,
    max_length=77,
    return_tensors="pt"
)

empty_tokens = tokenizer(
    "",
    padding="max_length",
    truncation=True,
    max_length=77,
    return_tensors="pt"
)

null_input_ids = empty_tokens.input_ids.squeeze(0).to(DEVICE)
null_attn_mask = empty_tokens.attention_mask.squeeze(0).to(DEVICE)

