import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen3-VL-8B-Instruct"
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Image
    ## Local file path
    [{"role": "user", "content": [{"type": "image", "image": "file:///workspace/ai-create-embeddings/fordcasepage3.png"}, {"type": "text", "text": "Describe this image."}]}],
]

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto")

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, _ = process_vision_info(messages, image_patch_size=16)

print(images)
print(type(images))

inputs = processor(text=text, images=images, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs)
print(generated_ids)


def extract_visual_features(image):
    image = image.convert("RGB")
    
    # Process image (no text prompt needed)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.float16)
    
    # Forward through vision encoder only
    with torch.no_grad():
        visual_features = model.visual(pixel_values)
        torch.save(visual_features.cpu(), "visual_features.pt")


extract_visual_features(images[0])