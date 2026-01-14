import pytest


@pytest.fixture
def client(monkeypatch):
    # Import here to avoid running the Slack handler
    from main import flask_app

    def fake_retrieve(query, mode="internal", top_k=3, min_relevance=0.60):
        return {
            "contexts": ["stub context"],
            "citations": [{"source": "test_source", "filename": "file.txt"}],
        }

    def fake_generate_chat_response(query, contexts):
        return f"answer for {query}"

    monkeypatch.setattr("main.retrieve", fake_retrieve)
    monkeypatch.setattr("main.generate_chat_response", fake_generate_chat_response)

    with flask_app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_chat_returns_response_and_citations(client):
    res = client.post("/chat", json={"message": "hello"})
    data = res.get_json()
    assert res.status_code == 200
    assert "response" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)


def test_qa_returns_answer_and_citations(client):
    res = client.post("/qa", json={"query": "hello", "mode": "internal"})
    data = res.get_json()
    assert res.status_code == 200
    assert "answer" in data
    assert "citations" in data


def test_api_retrieve_get(client):
    res = client.get("/api/retrieve", query_string={"query": "hello"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["query"] == "hello"
    assert "answer" in data
    assert "citations" in data
