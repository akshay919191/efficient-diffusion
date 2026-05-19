from transformers import CLIPTextModel , CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-base-patch32"
)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

for param in text_encoder.parameters():
    param.requires_grad = False
