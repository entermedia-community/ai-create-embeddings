#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from src.embedder.multimodal import save_visual_embeddings


def main():
    p = ArgumentParser()
    p.add_argument("text", help="Text prompt for the image")
    p.add_argument("image_path", help="Path to image file /home/shanti/Downloads/ford4.png")
    p.add_argument("output_path", help="Path to save embeddings (.pt)")
    p.add_argument("--model", dest="model_name", default=None, help="Optional model name (HF repo id) to use for extraction")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = p.parse_args()
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

    save_visual_embeddings(args.text, args.image_path, args.output_path, model_name=args.model_name)


if __name__ == "__main__":
    main()
