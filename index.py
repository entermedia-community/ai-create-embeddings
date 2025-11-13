from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
  documents,
  embed_model=Settings.embed_model,
)
index.storage_context.persist("storage")