# World Cup 2026 Fan Assistant

RAG-powered REST API for 2026 FIFA World Cup information, built with FastAPI, Amazon Bedrock, and ChromaDB.

## Architecture

```
User Question -> FastAPI -> Titan Embeddings V2 (embed) -> ChromaDB (search) -> Nova Lite (generate) -> Response
```

### Components
- **FastAPI** - REST API framework with automatic OpenAPI documentation
- **Amazon Bedrock** - Managed AI (Titan Embeddings V2 + Nova Lite)
- **ChromaDB** - Local vector database for semantic search
- **Docker** - Containerized deployment with multi-stage builds
- **GitHub Actions** - CI/CD pipeline (lint, test, build)

## Setup Instructions

1. Clone the repository and create a virtual environment:
```bash
git clone https://github.com/<your-username>/worldcup-fan-assistant.git
cd worldcup-fan-assistant
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials with Bedrock access in us-east-1.

4. Ingest the knowledge base:
```bash
python -m src.ingest
```

5. Run the API:
```bash
uvicorn src.main:app --reload
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock calls |
| `AWS_ACCESS_KEY_ID` | - | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key |
| `AWS_SESSION_TOKEN` | - | Optional session token |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

## Example API Calls

### Health Check
```bash
curl http://localhost:8000/health
```

### Query the Knowledge Base
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is the World Cup final?"}'
```

### Trigger Re-ingestion
```bash
curl -X POST http://localhost:8000/ingest
```

## Cost Breakdown

| Service | Usage | Estimated Cost |
|---------|-------|----------------|
| Bedrock Titan Embeddings V2 | ~50 embeddings during ingest + queries | < $0.01 |
| Bedrock Nova Lite | ~20 query responses during development | < $0.50 |
| S3 | Not used (local ChromaDB) | $0.00 |
| **Total** | | **< $1.00** |