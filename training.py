import torch , os
import torch.nn.functional as F
from scripts.pipeline import (vae , unet , custom_scaling_factor ,
                    optimizer1 , optimizer2 , text_encoder , tokenizer ,
                    get_custom_vae_latents , null_attn_mask , null_input_ids , inp)
from models.vae_unet import vae_loss
from scripts.ddpm_ddim import ddpm_scheduler , ddpm_step , noiseADD
from scripts.data_alinging import trainloader , valloader , Dataset , DataLoader


# vae training LOOP
"""
optimizer2 for vae , it gives (B , 16 , 32 , 32)(encoded) latents

flow:

    image -> vae encode and decode -> save mu and logvar ->  calculate loss on decoded image and the real -> return kl , recon and active loss 

"""

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

STARTEPOCH = 0
ENDEPOCH   = 25 # depends on u

# u can add cosine scheduler(i am not doing it , if u want there are tutorials if u use it will give a impact)


for epoch in range(STARTEPOCH , ENDEPOCH):
    # now we will iterate over data

    trainloss = 0 # this gives loss for each epoch
    for i , batch in enumerate(trainloader):   # you can use tqdm here it will be better 

        vae.train() # we are training now

        optimizer2.zero_grad() # zero all the grads so each image can have impact 

        image = batch['image'].to(DEVICE)

        # now we will predict using VAE , our vae returns recon(means the image) , mu , logvar
        recon , mu , logvar = vae(image)

        # here u can use KL annealing 
        kl = 1e-3 * min(1.0 , epoch / 10)
        kl = max(kl , 1e-4)

        #now kl divergence loss from paper ddpm
        loss , reconloss , klLOSS = vae_loss(image , recon , mu , logvar , kl)

        if torch.isnan(loss).any():
            print("NaN in loss")

        if torch.isnan(recon).any():
            print("NaN in recon")

        if torch.isnan(mu).any():
            print("NaN in mu")

        if torch.isnan(logvar).any():
            print("NaN in logvar")
        
        loss.backward() # backprop

        torch.nn.utils.clip_grad_norm_(vae.parameters() , 1.0) # clipping so we can be safe from a spike which can happen from a single image affecting the whole
        
        optimizer2.step() # now after backprop we have all grads saved , nwo changing it accordingly 
        trainloss += loss.item()

        """
        if (i % 20 == 0): 
            print(f"epoch {epoch} : step {i} : reconloss {reconloss} : klloss {klLOSS}")

        i am not using this , but u have to , so you can take a look on a training , typically 0 - 4 is a healthy recon loss and kl is 2 - 8 (its based on RTX 3050)
        """

    # now we will validate it (its your choice its slows down training , it won't affect much)
    vae.eval()
    valloss = 0.0

    with torch.no_grad(): # stopping the gradient saving cuz we are just testing
        for batch in valloader:
            image = batch['image'].to(DEVICE)
            recon , mu , logvar = vae(image)

            loss , _ , _  = vae_loss(image , recon , mu , logvar) # no kl here we will go with default
            valloss += loss.item()

    # now we will gave our VAE after each epoch , and will print avg loss over training and validation
    torch.save(vae.state_dict() , "VAE.pth") 
    print(f" epoch {epoch} : trainAVG {trainloss / len(trainloader):.4f} : valAVG {valloss / len(valloader):.4f}") # i used .4f you can use 2 or 5 depends on u



"""

recommendations:

    will recommend you to use loops separatly in a ipynb file instead of py 
    will recommend you to save latent first and then train unet (will be much faster , but will be good)(i will use too , will be little lenthy)

"""

# unet training loop
"""
optimizer1 for unet , it gives (B , 16 , 32 , 32)

flow:

    random numbers(noise)(latents like) -> predict -> calculate loss to vae latents

"""

def GETLATENTS():
    vae.eval()
    text_encoder.eval()

    with torch.no_grad(): # only calculating latents 
        for i , batch in enumerate(trainloader):
            image    = batch['image'].to(DEVICE)
            inputids = batch['input_ids'].to(DEVICE)
            attnmask = batch["attention_mask"].to(DEVICE)

            latents = get_custom_vae_latents(vae , image)
            textout = text_encoder(inputids , attnmask)
            textemb = textout.last_hidden_state()

            #move to cpu before saving
            latents = latents.to('cpu')
            textemb = textemb.to('cpu')

            torch.save({
                "latents" : latents,
                "textemb" : textemb,
                "input_ids" : inputids,
                "attnmask" : attnmask
            } , f"latent_cache/batch_{i}.pt")

GETLATENTS() # use this 


# you can use this functions for calling and easily unpack indices using DATASET
class Cachedlatents(Dataset):
    def __init__(self , cacheDIR , numbatch):
        self.paths = [
            f"{cacheDIR}/batch_idx{i}.pt"
            for i in range(numbatch)
        ]

        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self , idx):
            data = torch.load(self.paths[idx])
            return data
        
cached = Cachedlatents("latent_cache" , len(trainloader))
cacheLOADER = DataLoader(cached , batch_size = 1 , shuffle = True)

# now we have latents now unet training loop

# first we will load the VAE hence i don't need it
# vae.load_state_dict(torch.load("unet.pth")) or if you did a checkpoint you can go that way

scheduler = ddpm_scheduler(T = 1000 , device = DEVICE)
START = 0
END   = 30

if os.path.exists("unet_checkpoint.pt"):
    ckpt = torch.load("unet_checkpoint.pt")
    unet.load_state_dict(ckpt["unet"])
    optimizer1.load_state_dict(ckpt["optimizer"])
    

# as we are training unet , we need empty tokens too 
for epoch in range(START , END):
    totalloss = 0.0
    unet.train()

    for i , batch in cacheLOADER:🔥 
        optimizer1.zero_grad()

        latents = batch["latents"].to(DEVICE)  # as our latent shape is (1 , 4 , 32 , 32)
        input_ids = batch['input_ids'].to(DEVICE)
        attnmask  = batch["attnmask"].to(DEVICE)
        
        # now integrate CFG for text conditioning
        dropmask = (torch.rand(1 , device = DEVICE) < 0.10) # random True or False
        input_ids[dropmask] = null_input_ids # randomly choose if to nullify or not
        attnmask[dropmask]  = null_attn_mask

        with torch.no_grad(): 
            latents *= custom_scaling_factor
            textemb = text_encoder(input_ids , attnmask).last_hidden_state

        noise = torch.randn_like(latents)
        timesteps = torch.randnint(0 , 1000 , (1,) , device = DEVICE) # this way it learn to remove randomness at any level
        noisy , target = noiseADD(latents , noise , timesteps) # we are training to pred noise so we need prev noise too (target == noise)


        pred = unet(noisy , timesteps , textemb)
        loss = F.mse_loss(pred , target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters() , 1.0)

        if torch.isnan(loss):
            print("NAN detected")

        optimizer1.step()

        totalloss += loss.item()
        
        
    avg = totalloss / len(trainloader)
    print(f"Epoch {epoch} avg loss: {avg:.4f}")
    torch.save({
        'epoch': epoch,
        'unet': unet.state_dict(),  
        'optimizer': optimizer1.state_dict(),
    }, 'unet_checkpoint.pt')