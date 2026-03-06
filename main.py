from typing import Optional, List
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Optional
from fastapi import FastAPI, status, Header, Depends
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from llama_index.core import Settings, PromptTemplate, get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client.http.models import Filter, FieldCondition, MatchAny


from utils.document_maker import DocumentMaker

from db_manager import IndexRegistry

app = FastAPI()

logger = logging.getLogger(__name__)

registry = IndexRegistry(dim=1024)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

llm = OpenAILike(
    api_base="http://0.0.0.0:7600/", # Server uses local LLM
    # api_base="https://llamat.emediaworkspace.com/", # Use this for testing locally with the remote LLM
    is_chat_model=True,
    is_function_calling_model=True
)

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
  model_name="BAAI/bge-m3"
)

outline_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, create a list of sections header following the instructions in query. Provide only the list no additional text.\n"
    "Query: {query_str}\n"
    "Answer: "
)

class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: OpenAILike
    qa_prompt: PromptTemplate

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)


def get_collection_name(x_customerkey: Optional[str] = Header(None)):
    if not x_customerkey.isalnum():
        raise ValueError("Invalid customer key.")
    return f'client_{x_customerkey}_embeddings'

@app.get("/")
async def root():
    return {"message": "Running AI Embedding Service"}

class CreateEmbeddingData(BaseModel):
    page_id: str = Field(..., min_length=1, description="The page ID.")
    text: str = Field(..., min_length=1, description="The text content of the page.")
    page_label: str | None = Field(None, description="The label of the page.")

class CreateEmbeddingRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, description="The document ID.")
    file_name: str | None = Field(None, description="The file name.")
    file_type: str | None = Field(None, description="The file type.")
    creation_date: str | None = Field(None, description="The creation date.")
    pages: List[CreateEmbeddingData] = Field(..., min_length=1, description="List of pages to embed.")

@app.post("/save")
async def embed_document(
    all_data: CreateEmbeddingRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    
    index = registry.get(collection_name=x_customerkey)

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

        index.delete(doc_id=page_id, delete_from_docstore=True)
        
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
            logger.error(f"Error embedding page {page_id} of document {doc_id}: {str(e)}")
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"Document {doc_id} embedded successfully.",
            "processed": list(processed),
            "skipped": list(skipped),
            "failed": list(failed),
        }
    )
    
class QueryDocsRequest(BaseModel):
    query: str = Field(..., min_length=5, description="The query string.")
    parent_ids: List[str] = Field(..., min_length=1, description="List of parent document IDs to filter by.")

@app.post("/query")
async def query_docs(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    
    index = registry.get(collection_name=x_customerkey)

    try:        
        filters = Filter(
            must=[
                FieldCondition(
                    key="parent_id",
                    match=MatchAny(
                        any=data.parent_ids
                    )
                )
            ]
        )

        query_engine = index.as_query_engine(vector_store_kwargs={"qdrant_filters": filters})
        
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
        logger.error("Error during query_docs: " + str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


@app.post("/create_outline")
async def create_outline(
    data: QueryDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    index = registry.get(collection_name=x_customerkey)

    try:        
        filters = Filter(
            must=[
                FieldCondition(
                    key="parent_id",
                    match=MatchAny(
                        any=data.parent_ids
                    )
                )
            ]
        )

        retriever = index.as_retriever(vector_store_kwargs={"qdrant_filters": filters})

        query_engine = RAGStringQueryEngine.from_args(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(),
            llm=llm,
            qa_prompt=outline_prompt,
        )

        # query_engine = index.as_query_engine(vector_store_kwargs={"qdrant_filters": filters})

        response = query_engine.query(data.query)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"outline": str(response) }
        )

    except Exception as e:
        logger.error("Error during create_outline: " + str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
    
class DeleteDocsRequest(BaseModel):
    node_ids: List[str] = Field(..., description="List of node IDs to delete.")
    
@app.post("/delete_document")
async def delete_document(
    data: DeleteDocsRequest,
    x_customerkey: Optional[str] = Depends(get_collection_name)
):
    index = registry.get(collection_name=x_customerkey)

    for node_id in data.node_ids:
        index.delete(doc_id=node_id, delete_from_docstore=True)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Nodes deleted successfully."}
    )