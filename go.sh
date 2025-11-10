export PYTHONPATH="$(pwd)"

python scripts/save_embeddings.py "Create a 3-8 word semantic summary of each paragraph" /workspace/ai-create-embeddings/fordcasepage3.png out.pt --model "Qwen/Qwen3-VL-8B-Instruct" --verbose
