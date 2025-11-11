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

    model_path = args.model or 'Qwen/Qwen3-VL-8B-Instruct'
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
    image = Image.open(args.image).convert('RGB')
    
    # Process inputs for the model
    logger.info('Processing inputs with processor')
    inputs = processor(images=image, text=args.text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    logger.debug('Input keys: %s', list(inputs.keys()))
    
    # Extract embeddings using the model's encoder
    visual_emb = None
    text_emb = None
    
    with torch.no_grad():
        # Try to use get_image_features and get_text_features if available
        if hasattr(model, 'get_image_features') and hasattr(model, 'get_text_features'):
            try:
                visual_emb = model.get_image_features(pixel_values=inputs.get('pixel_values'))
                logger.info('Extracted visual embeddings via get_image_features: shape=%s', visual_emb.shape)
            except Exception as e:
                logger.warning('get_image_features failed: %s', e)
                visual_emb = None
            
            try:
                text_emb = model.get_text_features(input_ids=inputs.get('input_ids'))
                logger.info('Extracted text embeddings via get_text_features: shape=%s', text_emb.shape)
            except Exception as e:
                logger.warning('get_text_features failed: %s', e)
                text_emb = None
        
        # Fallback: use encoder directly
        if visual_emb is None or text_emb is None:
            logger.info('Using encoder fallback method')
            try:
                encoder = model.get_encoder()
                encoder_output = encoder(**inputs)
                
                # Try to get image_embeds and text_embeds from encoder output
                if visual_emb is None:
                    visual_emb = getattr(encoder_output, 'image_embeds', None)
                if text_emb is None:
                    text_emb = getattr(encoder_output, 'text_embeds', None)
                
                # Last resort: use last_hidden_state with mean pooling
                if visual_emb is None and hasattr(encoder_output, 'last_hidden_state'):
                    visual_emb = encoder_output.last_hidden_state.mean(dim=1)
                    logger.info('Using mean-pooled last_hidden_state for visual: shape=%s', visual_emb.shape)
                if text_emb is None and hasattr(encoder_output, 'last_hidden_state'):
                    text_emb = encoder_output.last_hidden_state.mean(dim=1)
                    logger.info('Using mean-pooled last_hidden_state for text: shape=%s', text_emb.shape)
                    
            except Exception as e:
                logger.exception('Encoder fallback failed')
                raise RuntimeError('Unable to extract embeddings from model') from e
        
        if visual_emb is None or text_emb is None:
            raise RuntimeError(f'Failed to extract embeddings: visual={visual_emb is not None}, text={text_emb is not None}')
        
        # Create multimodal embedding by concatenating visual and text
        multimodal_emb = torch.cat([visual_emb, text_emb], dim=-1).squeeze()
        logger.info('Created multimodal embedding: shape=%s', multimodal_emb.shape)

    # Save embeddings and text
    logger.info('Saving embeddings to: %s', args.output)
    torch.save({
        'visual_embedding': visual_emb.cpu().squeeze(),
        'text_embedding': text_emb.cpu().squeeze(),
        'multimodal_embedding': multimodal_emb.cpu(),
        'input_ids': inputs.get('input_ids', torch.tensor([])).cpu(),
        'text': args.text,
        'image_path': args.image,
    }, args.output)
    
    logger.info('Embedding shapes: visual=%s, text=%s, multimodal=%s',
                visual_emb.shape, text_emb.shape, multimodal_emb.shape)
    logger.info('Done! Use run_inference.py with this file to generate outputs')


if __name__ == "__main__":
    main()
