from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# os.environ["OLLAMA_API_URL"] = "http://localhost:11434"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Settings control global defaults
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
	model="qwen3:4b",
	request_timeout=360.0,
	context_window=10000,
)

data_path = "./data"
storage_path = "./storage"

def get_index():
	if os.path.exists(storage_path) and os.listdir(storage_path):
		print("Loading existing index from storage...")
		storage_context = StorageContext.from_defaults(persist_dir=storage_path)
		index = load_index_from_storage(storage_context)
		print("Loaded existing index from storage.")
	else:
		print("Creating new index from documents...")
		documents = SimpleDirectoryReader(data_path).load_data()
		index = VectorStoreIndex.from_documents(documents)
		index.storage_context.persist(storage_path)
		print("Created new index and persisted to storage.")
	
	return index

index = get_index()
query_engine = index.as_query_engine()

def multiply(a: float, b: float) -> float:
	"""Useful for multiplying two numbers."""
	return a * b

async def search_documents(query: str) -> str:
	"""Useful for answering natural language questions about an personal essay written by Paul Graham."""
	print(f"Searching documents with query: {query}")
	response = await query_engine.aquery(query)
	print(f"Document search response: {response}")
	return str(response)


# Create an enhanced workflow with both tools
agent = AgentWorkflow.from_tools_or_functions(
	[multiply, search_documents],
	llm=Settings.llm,
	system_prompt="""You are a helpful assistant that can perform calculations
	and search through documents to answer questions.""",
)


# Now we can ask questions about the documents or do calculations
async def main():
	print("Running agent...")
	response = await agent.run(
		"What did the author do in college? Also, what's 7 * 8?"
	)
	print(str(response))


# Run the agent
if __name__ == "__main__":
	asyncio.run(main())