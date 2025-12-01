from typing import Optional
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List
from fastapi import FastAPI, status, Header
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from llama_index.vector_stores.qdrant import QdrantVectorStore

from utils.document_maker import DocumentMaker

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.llm = OpenAILike(
    api_base="https://llama.thoughtframe.ai/v1",
    is_chat_model=True,
    is_function_calling_model=True
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

client = {}
vector_store = {}
index = {}

COLLECTION_NAME = "documents"

def get_index(customerkey: str):
    if customerkey not in client:
        print("Creating Qdrant client for customerkey:", customerkey)
        client[customerkey] = QdrantClient(path="./db/"+customerkey)
 

    if not client[customerkey].collection_exists(collection_name=COLLECTION_NAME):
        client[customerkey].create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            )
        )

    if customerkey not in vector_store:
        vector_store[customerkey] = QdrantVectorStore(client=client[customerkey], collection_name=COLLECTION_NAME)

    if customerkey not in index:
        try:
            index[customerkey] = VectorStoreIndex.from_vector_store(vector_store[customerkey])
            print("Loaded existing index")
        except:
            print("Creating new index with storage_context")
            storage_context = StorageContext.from_defaults(vector_store=vector_store[customerkey])
            index[customerkey] = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context 
            )
    
    return index[customerkey]

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}

class CreateEmbeddingData(BaseModel):
    page_id: str
    text: str
    page_label: str | None = None

class CreateEmbeddingRequest(BaseModel):
    doc_id: str
    file_name: str
    file_type: str | None = None
    creation_date: str | None = None
    pages: List[CreateEmbeddingData]

@app.post("/save")
async def embed_document(
    all_data: CreateEmbeddingRequest,
    x_customerkey: Optional[str] = Header(None) #Send a x-customerkey header with the request
):
    if not x_customerkey:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid request."}
        )
    
    index = get_index(x_customerkey)

    doc_id = all_data.doc_id
    file_name = all_data.file_name
    file_type = all_data.file_type
    creation_date = all_data.creation_date

    if not doc_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Document ID is required."}
        )
    
    processed = set()
    failed = set()
    skipped = set()
    print("Adding pages for document ID:", doc_id)
    for data in all_data.pages:
        page_id = data.page_id
        if not page_id:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Document ID is required."}
            )
        
        page_label = data.page_label

        text = data.text

        if not text or text.strip() == "":
            skipped.add(page_id)
            continue

        doc_maker = DocumentMaker(
            id=page_id,
            parent_id=doc_id,
            page_label=page_label,
            file_name=file_name,
            file_type=file_type,
            creation_date=creation_date,
        )
        try:
            document = doc_maker.create_document(text)
            index.insert(document)
            processed.add(page_id)
            print("Added page ID:", page_id)
        except Exception as e:
            failed.add(page_id)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Document with ID {doc_id} embedded successfully.",
            "processed": list(processed),
            "skipped": list(skipped),
            "failed": list(failed),
        }
    )
    
class QueryDocsRequest(BaseModel):
    query: str
    doc_ids: list[str]

@app.post("/query")
async def query_docs(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Header(None) #Send a x-customerkey header with the request
):
    if not x_customerkey:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid request."}
        )
    
    index = get_index(x_customerkey)

    try:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="parent_id", operator=FilterOperator.IN, value=data.doc_ids)
            ]
        )

        query_engine = index.as_query_engine(filters=filters)
        
        response = query_engine.query(data.query)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
            "query": data.query,
            "answer": str(response),
            "sources": [
                {
                    **node.node.metadata,
                    "score": node.score,
                }
                for node in response.source_nodes
            ]
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
