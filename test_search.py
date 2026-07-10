import chromadb
import json
import boto3
from src.config import CHROMA_PERSIST_DIR, EMBED_MODEL_ID, AWS_REGION

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name="worldcup2026")

print(f"Total chunks in collection: {collection.count()}")

response = client.invoke_model(
    modelId=EMBED_MODEL_ID,
    contentType="application/json",
    accept="application/json",
    body=json.dumps({"inputText": "When is the World Cup final?"}),
)

result = json.loads(response["body"].read())
query_embedding = result["embedding"]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

top_document = results["documents"][0][0]
top_metadata = results["metadatas"][0][0]

print("\nQuestion: When is the World Cup final?")
print(f"Source: {top_metadata['source']}")
print(f"Retrieved context:\n{top_document}")

assert "July 19, 2026" in top_document
print("\n✅ Retrieval test passed: the context contains July 19, 2026.")

print("\nTop results:\n")

for i, (doc, meta, dist) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )
):
    print(f"Result {i+1}")
    print(f"Similarity: {1 - dist:.4f}")
    print("Source:", meta["source"])
    print(doc[:150])
    print()