sudo docker run --name documentembedding \
	-v "/models/hf:/models/hf" \
  -e HF_HOME=/models/hf \
	documentembedding