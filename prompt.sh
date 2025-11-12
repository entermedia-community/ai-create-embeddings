export PYTHONPATH="$(pwd)"

python scripts/run_prompt.py \
  --prompt "Hello" \
  --output out.pt \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --verbose
