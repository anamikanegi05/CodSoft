import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Loading model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

imgPath = "ImageCaptioning/flower.jpg"

try:
    img = Image.open(imgPath).convert("RGB")
except:
    print("Error: Image not found!")
    exit()

inputs = processor(img, return_tensors="pt")
inputs = inputs.to(device)

result = model.generate(
    **inputs,
    max_length=40,
    num_beams=3
)

caption = processor.decode(result[0], skip_special_tokens=True)

print("\nCaption Generated:")
print(caption)
