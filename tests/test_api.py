import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "documents" in data


@patch("src.rag_service.get_bedrock_client")
@patch("src.rag_service.get_collection")
def test_query_endpoint(mock_collection, mock_bedrock):
    mock_col = MagicMock()
    mock_col.query.return_value = {
        "documents": [
            ["The final is at MetLife Stadium on July 19, 2026."]
        ],
        "metadatas": [[{"source": "schedule.md"}]],
    }
    mock_collection.return_value = mock_col

    mock_client = MagicMock()

    mock_embed_response = {
        "body": MagicMock(
            read=lambda: json.dumps(
                {"embedding": [0.1] * 1024}
            ).encode("utf-8")
        )
    }

    mock_llm_response = {
        "body": MagicMock(
            read=lambda: json.dumps(
                {
                    "output": {
                        "message": {
                            "content": [
                                {
                                    "text": (
                                        "The World Cup final is at "
                                        "MetLife Stadium."
                                    )
                                }
                            ]
                        }
                    }
                }
            ).encode("utf-8")
        )
    }

    mock_client.invoke_model.side_effect = [
        mock_embed_response,
        mock_llm_response,
    ]
    mock_bedrock.return_value = mock_client

    response = client.post(
        "/query",
        json={"question": "Where is the final?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "MetLife Stadium" in data["answer"]
    assert "schedule.md" in data["sources"]

    assert mock_client.invoke_model.call_count == 2
    mock_col.query.assert_called_once()