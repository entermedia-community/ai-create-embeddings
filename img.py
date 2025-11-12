from transformers import AutoImageProcessor
from torchvision.io import read_image

model_name = "Qwen/Qwen3-VL-8B-Instruct"


processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

images = read_image("./fordcasepage3.png")
images_processed = processor(images, return_tensors="pt", device="cuda")

print(images_processed.keys())