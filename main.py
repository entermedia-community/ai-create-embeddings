from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import asyncio
import os
import faiss

faiss_index = faiss.IndexFlatL2(768)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

Settings.llm = OpenAILike(
    api_key="fake-key",
    api_base="http://142.113.71.170:36238/v1",
    is_chat_model=True,
    is_function_calling_model=True
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

faiss_storage_dir = "faiss_storage"

def get_index():
    """Get or create the FAISS index."""
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    if os.path.exists(faiss_storage_dir):
        print("Loading existing FAISS index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=faiss_storage_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("Creating new FAISS index...")
        documents = SimpleDirectoryReader("data").load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist(persist_dir=faiss_storage_dir)
    return index

index = get_index()

query_engine = index.as_query_engine()

index.storage_context.persist()

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an enhanced workflow with both tools
agent = FunctionAgent(
    tools=[multiply, search_documents],
    system_prompt="""You are a helpful assistant that can perform calculations
    and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
    response = await agent.run(
        "What did the author do in college? Also, what's 7 * 8?"
    )
    print(response)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())