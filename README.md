# ai-createembeddings

This repository contains a Python library for producing multimodal embeddings using PyTorch and local Qwen3V models. It supports joint text-image embeddings using locally stored GGUF or other model formats.

## Quick Start

1. Create a virtual environment and install requirements:

```bash
python -m venv .venv;source .venv/bin/activate;pip install -r requirements.txt
```

2. Ensure you have a local Qwen3V model file. The default path is:
```
/models/unsloth_Qwen3-VL-8B-Instruct-GGUF_Qwen3-VL-8B-Instruct-Q4_K_M.gguf
```

3. Run tests with your local model:
```bash
pytest -q
```

## Multimodal Embeddings

The library creates joint text-image embeddings using a local Qwen3V model.

### Basic Usage

```bash
# Generate embeddings with default model path
python scripts/save_embeddings.py "Describe this landscape" path/to/image.jpg embeddings.pt

# Use verbose mode for detailed logging
python scripts/save_embeddings.py "Describe this image" path/to/image.jpg out.pt --verbose

# Specify CUDA device if available
python scripts/save_embeddings.py "A description" image.jpg out.pt --device cuda:0

# Specify a different local model path
python scripts/save_embeddings.py "Text prompt" image.jpg out.pt --model "/path/to/your/local/qwen3v-model.gguf"
```

### Output Formats

The script saves embeddings in two formats:

1. PyTorch format (`.pt`):
   - Contains full context including input text and image path
   - Includes text, visual, and multimodal embeddings
   - Preserves input IDs for token analysis
   
2. NumPy format (`.npz`):
   - Lightweight format for just the embeddings
   - Easy to load in other frameworks
   - Contains `visual`, `text`, and `multimodal` arrays

### Model Support

The library works with local Qwen3V models:
- Supports GGUF format (recommended for efficiency)
- Compatible with .bin model files
- Can load from model directories
- Default path: `/models/unsloth_Qwen3-VL-8B-Instruct-GGUF_Qwen3-VL-8B-Instruct-Q4_K_M.gguf`

### Programming Interface

```python
from src.embedder.multimodal import extract_multimodal_embeddings

# Get embeddings dictionary using default model path
embeddings = extract_multimodal_embeddings(
    text="A beautiful mountain landscape",
    image_path="landscape.jpg",
    device="cuda"  # or "cpu"
)

# Or specify a custom model path
embeddings = extract_multimodal_embeddings(
    text="A beautiful mountain landscape",
    image_path="landscape.jpg",
    device="cuda",
    model_name="/path/to/your/model.gguf"
)

# Access individual embeddings
visual_emb = embeddings["visual"]       # Image features
text_emb = embeddings["text"]          # Text features
multimodal = embeddings["multimodal"]   # Combined features
```

## Notes

- Requires a local Qwen3V model file (GGUF format recommended)

modelscope download --model ggml-org/Qwen3-VL-2B-Instruct-GGUF README.md --local_dir /models

- Environment variable `TRANSFORMERS_CACHE` can be set to control model loading cache location
- Model files are loaded in local-only mode with trusted code execution
- Use the `--verbose` flag for detailed logging of model loading, processing steps, and embedding shapes
- All embeddings are normalized and can be used directly for similarity computations
- For better performance, consider using quantized models (Q4_K_M or similar)
