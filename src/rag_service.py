import json
import time
import chromadb
import boto3
from src.config import (
    AWS_REGION,
    EMBED_MODEL_ID,
    LLM_MODEL_ID,
    CHROMA_PERSIST_DIR,
    TOP_K,
)


# Initialize the Bedrock client for API calls
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


# Connect to the persistent ChromaDB collection
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return chroma_client.get_or_create_collection(
        name="worldcup2026",
        metadata={"hnsw:space": "cosine"},
    )


# Generate a vector embedding for a given text string
def generate_embedding(client, text: str) -> list[float]:
    response = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def query_knowledge_base(question: str) -> dict:
    start_time = time.time()
    client = get_bedrock_client()
    collection = get_collection()

    # Embed the user's question into a vector
    query_embedding = generate_embedding(client, question)

    # Search ChromaDB for the top-5 most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    context_chunks = results["documents"][0] if results["documents"] else []
    sources = results["metadatas"][0] if results["metadatas"] else []

    # Join retrieved chunks into a single context block
    context = "\n\n---\n\n".join(context_chunks)

    # Build the prompt instructing Nova Lite to use only the provided context
    prompt_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"You are a helpful 2026 FIFA World Cup assistant. "
                            f"Answer the following question using ONLY the provided context. "
                            f"If the context does not contain enough information, say so. "
                            f"Cite which source documents you used.\n\n"
                            f"Context:\n{context}\n\n"
                            f"Question: {question}"
                        )
                    }
                ],
            }
        ],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.3, "topP": 0.9},
    }

    # Invoke Nova Lite to generate the answer
    response = client.invoke_model(
        modelId=LLM_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(prompt_body),
    )

    response_body = json.loads(response["body"].read())
    answer = response_body["output"]["message"]["content"][0]["text"]

    # Calculate total processing time in milliseconds
    processing_time = (time.time() - start_time) * 1000

    return {
        "answer": answer,
        "sources": [s["source"] for s in sources],
        "processing_time_ms": round(processing_time, 2),
        "chunks_retrieved": len(context_chunks),
    }

