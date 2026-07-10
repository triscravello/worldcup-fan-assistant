import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from src.config import CHROMA_PERSIST_DIR
from src.rag_service import query_knowledge_base
from src.ingest import ingest_documents

# Configure structured logging for production observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="World Cup 2026 Fan Assistant",
    description="RAG-powered API for 2026 FIFA World Cup information",
    version="1.0.0",
)

# Enable cross-origin requests so frontends can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response schemas with Pydantic
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    processing_time_ms: float
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    documents: int


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int

@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = chroma_client.get_or_create_collection(name="worldcup2026")
        doc_count = collection.count()
        return HealthResponse(status="healthy", documents=doc_count)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    logger.info(json.dumps({"event": "query", "question": request.question}))
    try:
        result = query_knowledge_base(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint():
    logger.info(json.dumps({"event": "ingest_start"}))
    try:
        count = ingest_documents()
        logger.info(json.dumps({"event": "ingest_complete", "chunks": count}))
        return IngestResponse(status="complete", chunks_ingested=count)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))