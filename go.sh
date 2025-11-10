export PYTHONPATH="$(pwd)"

python scripts/save_embeddings.py "Create a 3-8 word semantic summary of each paragraph" /workspace/ai-create-embeddings/fordcasepage3.png out.pt --model "/workspace/models/unsloth_Qwen3-VL-8B-I
nstruct-GGUF_Qwen3-VL-8B-Instruct-Q4_K_M.gguf" --verbose
