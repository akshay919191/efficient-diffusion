import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T
from pathlib import Path
import matplotlib.pyplot as plt

from models.clip_text import tokenizer 


PATH = Path("/home/toxiccccterabaap/Downloads/stuff/archive")
caption = PATH / "captions.txt"

transformer = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
])



class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, caption_file, tokenizer, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.tokenizer = tokenizer

        self.data = []
        with open(caption_file, "r") as f:
            next(f)
            for line in f:
                img, cap = line.strip().split(",", 1)
                self.data.append((img, cap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]

        image = Image.open(self.root / img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0)
        }
    

data = ImageCaptionDataset(
    root_dir=PATH / "Images",
    caption_file=caption,
    tokenizer=tokenizer,
    transform=transformer
)

train_size = int(0.8 * len(data))
val_size = len(data) - train_size

train_dataset, val_dataset = random_split(data, [train_size, val_size])

trainloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)