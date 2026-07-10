import os
import json
import boto3
import chromadb
from src.config import (
    AWS_REGION,
    EMBED_MODEL_ID,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def load_documents(data_dir: str = "data") -> list[dict]:
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            documents.append({"content": content, "source": filename})
    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks



def generate_embedding(client, text: str) -> list[float]:
    response = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]



def ingest_documents(data_dir: str = "data") -> int:
    client = get_bedrock_client()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    collection = chroma_client.get_or_create_collection(
        name="worldcup2026",
        metadata={"hnsw:space": "cosine"},
    )

    documents = load_documents(data_dir)
    doc_id = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            embedding = generate_embedding(client, chunk)
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["source"]}],
            )
            doc_id += 1

    return doc_id


if __name__ == "__main__":
    count = ingest_documents()
    print(f"Ingested {count} chunks into ChromaDB")