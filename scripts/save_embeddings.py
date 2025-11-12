#!/usr/bin/env python3
"""Save image embeddings for later inference.

This script processes an image through the Qwen3V model and saves its embeddings
along with the original text prompt. The embeddings can be loaded by run_inference.py
for faster repeated inference without needing to reprocess the image.

Usage:
  python scripts/save_embeddings.py --image image.jpg --text "Description" --output embeddings.pt
"""
import argparse
import logging
import os
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--text', required=True, help='Text prompt/description')
    parser.add_argument('--output', required=True, help='Path to save embeddings .pt file')
    parser.add_argument('--model', required=False, default=None, help='Local model path')
    parser.add_argument('--device', required=False, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    log_level = logging.DEBUG # if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger('save_embeddings')

    if not os.path.exists(args.image):
        logger.error('Image file not found: %s', args.image)
        raise SystemExit(1)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Running on device: %s', device)

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:
        logger.exception('Failed to import Qwen3V classes from transformers')
        raise

    model_path = args.model or 'Qwen/Qwen3-VL-2B-Instruct'
    # if not os.path.exists(model_path):
    #     logger.error('Model path does not exist: %s', model_path)
    #     raise SystemExit(1)
    print("Model:", model_path)
    
    logger.info('Loading processor and model from: %s', model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        dtype="auto",
        device_map="auto",
    )

    # Load and process image
    logger.info('Processing image: %s', args.image)
    image = Image.open(args.image) #.convert('RGB')
    
    # Get image inputs (pixel_values, etc.)
    image_inputs = processor(images=[args.image], text=[args.text], return_tensors='pt')

    # Save the processed inputs (pixel_values, etc.) instead of embeddings
    # This allows us to use them directly in generate()
    logger.info('Saving processed image inputs to: %s', args.output)
    
    # Extract only the image-related keys (pixel_values, image_grid_thw, etc.)
    image_data = {k: v for k, v in image_inputs.items() if k not in ['input_ids', 'attention_mask']}
    
    torch.save({
        'text': args.text,
        'image_inputs': image_data  # Changed from image_embeds to image_inputs
    }, args.output)
    logger.info('Done! Saved keys: %s', list(image_data.keys()))
    logger.info('Use run_smart_prompt.py with this file to generate outputs')

if __name__ == "__main__":
    main()
