import logging
import json
from datetime import date
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from src.config import CHROMA_PERSIST_DIR
from src.rag_service import query_knowledge_base, get_bedrock_client
from src.ingest import ingest_documents

MATCHES_FILE = Path("data/matches.json")


def load_matches() -> list[dict]:
    if not MATCHES_FILE.exists():
        logger.warning(
            json.dumps(
                {
                    "event": "matches_file_missing",
                    "path": str(MATCHES_FILE),
                }
            )
        )
        return []

    with MATCHES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)

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


class MatchBriefing(BaseModel):
    teams: str
    venue: str
    kickoff_time: str
    narrative: str


class MatchDayResponse(BaseModel):
    date: str
    matches: list[MatchBriefing]

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


@app.get("/matches/today", response_model=MatchDayResponse)
def matches_today(target_date: date | None = None):
    check_date = (target_date or date.today()).isoformat()

    logger.info(
        json.dumps(
            {
                "event": "match_day_query",
                "date": check_date,
            }
        )
    )

    try:
        scheduled_matches = [
            match
            for match in load_matches()
            if match.get("date") == check_date
        ]

        if not scheduled_matches:
            return MatchDayResponse(
                date=check_date,
                matches=[],
            )

        client = get_bedrock_client()
        briefings: list[MatchBriefing] = []

        for match in scheduled_matches:
            teams = f"{match['home_team']} vs {match['away_team']}"

            prompt_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": (
                                    "Write an exciting two-sentence fan preview "
                                    "for this 2026 FIFA World Cup match. "
                                    "Do not change or invent the teams, venue, "
                                    "kickoff time, or date.\n\n"
                                    f"Date: {check_date}\n"
                                    f"Teams: {teams}\n"
                                    f"Venue: {match['venue']}\n"
                                    f"Kickoff time: {match['kickoff_time']}"
                                )
                            }
                        ],
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 250,
                    "temperature": 0.7,
                    "topP": 0.9,
                },
            }

            response = client.invoke_model(
                modelId="amazon.nova-lite-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps(prompt_body),
            )

            response_body = json.loads(response["body"].read())
            narrative = response_body["output"]["message"]["content"][0]["text"]

            briefings.append(
                MatchBriefing(
                    teams=teams,
                    venue=match["venue"],
                    kickoff_time=match["kickoff_time"],
                    narrative=narrative.strip(),
                )
            )

        logger.info(
            json.dumps(
                {
                    "event": "match_day_complete",
                    "date": check_date,
                    "matches": len(briefings),
                }
            )
        )

        return MatchDayResponse(
            date=check_date,
            matches=briefings,
        )

    except json.JSONDecodeError as exc:
        logger.exception(
            json.dumps(
                {
                    "event": "match_data_invalid",
                    "error": str(exc),
                }
            )
        )
        raise HTTPException(
            status_code=500,
            detail="Match data is not valid JSON.",
        ) from exc

    except Exception as exc:
        logger.exception(
            json.dumps(
                {
                    "event": "match_day_failed",
                    "date": check_date,
                    "error": str(exc),
                }
            )
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to generate match-day briefings.",
        ) from exc