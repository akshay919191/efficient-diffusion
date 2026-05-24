import torch
from training import vae , unet
from models.clip_text import tokenizer , text_encoder
from scripts.ddpm_ddim import ddpm_scheduler , ddpm_step
from scripts.pipeline import custom_scaling_factor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DEVICE = next(vae.parameters()).device
custom_scaling_factor = 1.0 

vae.eval()
unet.eval()

@torch.no_grad()
def inference(
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt,
    device,
    steps=1000,
    guidance_scale=7.5,
    latent_shape=(1, 16, 32, 32),
):
    def encode(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = text_encoder(**tokens)
        return out.last_hidden_state

    cond_emb = encode(prompt)
    uncond_emb = encode("")

    x = torch.randn(latent_shape, device=device)

    T = steps
    timesteps = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for t in timesteps:

        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)

        eps_uncond = unet(x, t_batch, uncond_emb)
        eps_cond = unet(x, t_batch, cond_emb)

        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        x = ddpm_step(x, eps, t_batch, scheduler)


    x = x / custom_scaling_factor  

    image = torch.sigmoid(vae.decode(x))
    return image

scheduler = ddpm_scheduler(T=1000, device=DEVICE)


generated_tensor = inference(
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    prompt="a futuristic city at sunset",
    device=DEVICE,
    steps=1000,          
    guidance_scale=7.5
)

img_array = generated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

img_uint8 = (img_array * 255).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(img_uint8)
plt.axis("off")
plt.title("Generated Image: a futuristic city at sunset")
plt.show()
pil_img = Image.fromarray(img_uint8)