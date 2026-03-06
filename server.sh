export CUDA_VISIBLE_DEVICES=1
uvicorn main:app --port 4600 --host 0.0.0.0 > /dev/null 2>&1 &