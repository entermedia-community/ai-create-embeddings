import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display

# dimensions of text-ada-embedding-002
d = 768
faiss_index = faiss.IndexFlatL2(d)


documents = SimpleDirectoryReader("./data").load_data()

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
  documents, storage_context=storage_context
)

index.storage_context.persist()
