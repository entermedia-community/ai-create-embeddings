#!/bin/bash
lsof -ti :4600 | xargs -r kill -9

export CUDA_VISIBLE_DEVICES=1
uvicorn main:app \
	--host 0.0.0.0 \
	--port 4600 \
	--timeout-keep-alive "${UVICORN_TIMEOUT_KEEP_ALIVE:-120}" \
	--workers 1 > /dev/null 2>&1 &