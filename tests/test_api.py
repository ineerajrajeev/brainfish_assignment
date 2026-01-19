import pytest
import json


@pytest.fixture
def client(monkeypatch):
    # Import here to avoid running the Slack handler
    from main import flask_app

    def fake_retrieve(query, mode="internal", top_k=3, min_relevance=0.45):
        return {
            "contexts": ["stub context"],
            "citations": [{"source": "test_source", "filename": "file.txt"}],
            "documents": [{"text": "stub context", "metadata": {"source": "test_source"}}],
            "unique_sources": ["test_source"]
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
    assert "documents" in data
    assert isinstance(data["citations"], list)
    assert isinstance(data["documents"], list)


def test_qa_returns_answer_and_citations(client):
    res = client.post("/qa", json={"query": "hello", "mode": "internal"})
    data = res.get_json()
    assert res.status_code == 200
    assert "answer" in data
    assert "citations" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_api_retrieve_get(client):
    res = client.get("/api/retrieve", query_string={"query": "hello"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["query"] == "hello"
    assert "answer" in data
    assert "citations" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)
def test_api_returns_valid_json(client):
    """Test that all API endpoints return valid, well-formed JSON data."""
    endpoints = [
        {
            "method": "GET",
            "path": "/health",
            "expected_fields": ["status"],
            "expected_types": {"status": str}
        },
        {
            "method": "POST",
            "path": "/chat",
            "data": {"message": "test query"},
            "expected_fields": ["response", "citations", "documents"],
            "expected_types": {
                "response": str,
                "citations": list,
                "documents": list
            }
        },
        {
            "method": "POST",
            "path": "/qa",
            "data": {"query": "test query", "mode": "internal"},
            "expected_fields": ["answer", "citations", "documents"],
            "expected_types": {
                "answer": str,
                "citations": list,
                "documents": list
            }
        },
        {
            "method": "GET",
            "path": "/api/retrieve",
            "query_string": {"query": "test query"},
            "expected_fields": ["answer", "citations", "documents", "query", "mode", "num_sources"],
            "expected_types": {
                "answer": str,
                "citations": list,
                "documents": list,
                "query": str,
                "mode": str,
                "num_sources": int
            }
        },
        {
            "method": "POST",
            "path": "/api/retrieve",
            "data": {"query": "test query", "mode": "customer", "top_k": 5},
            "expected_fields": ["answer", "citations", "documents", "query", "mode", "num_sources"],
            "expected_types": {
                "answer": str,
                "citations": list,
                "documents": list,
                "query": str,
                "mode": str,
                "num_sources": int
            }
        }
    ]
    
    for endpoint in endpoints:
        method = endpoint["method"]
        path = endpoint["path"]
        
        if method == "GET":
            query_string = endpoint.get("query_string", {})
            res = client.get(path, query_string=query_string)
        else:
            data = endpoint.get("data", {})
            res = client.post(path, json=data)
        
        assert res.status_code in [200, 400, 500], f"{method} {path} returned unexpected status {res.status_code}"
        assert res.content_type == "application/json", f"{method} {path} did not return JSON content type"
        
        try:
            json_data = res.get_json()
            assert json_data is not None, f"{method} {path} returned null JSON"
        except Exception as e:
            pytest.fail(f"{method} {path} returned invalid JSON: {e}")
        
        try:
            json_str = json.dumps(json_data)
            parsed_back = json.loads(json_str)
            assert parsed_back == json_data, f"{method} {path} JSON is not perfectly serializable"
        except (TypeError, ValueError) as e:
            pytest.fail(f"{method} {path} JSON contains non-serializable data: {e}")
        
        if res.status_code == 200:
            expected_fields = endpoint.get("expected_fields", [])
            expected_types = endpoint.get("expected_types", {})
            
            for field in expected_fields:
                assert field in json_data, f"{method} {path} missing required field: {field}"
            
            for field, expected_type in expected_types.items():
                if field in json_data:
                    actual_value = json_data[field]
                    assert isinstance(actual_value, expected_type), \
                        f"{method} {path} field '{field}' has wrong type. Expected {expected_type.__name__}, got {type(actual_value).__name__}"
            
            if "citations" in json_data:
                assert isinstance(json_data["citations"], list), \
                    f"{method} {path} citations must be a list"
                for citation in json_data["citations"]:
                    assert isinstance(citation, dict), \
                        f"{method} {path} each citation must be a dictionary"
            
            if "documents" in json_data:
                assert isinstance(json_data["documents"], list), \
                    f"{method} {path} documents must be a list"
                for doc in json_data["documents"]:
                    assert isinstance(doc, dict), \
                        f"{method} {path} each document must be a dictionary"
                    assert "text" in doc or "metadata" in doc, \
                        f"{method} {path} each document must have 'text' or 'metadata' field"
        
        elif res.status_code in [400, 500]:
            assert "error" in json_data, f"{method} {path} error response missing 'error' field"
            assert isinstance(json_data["error"], str), \
                f"{method} {path} error field must be a string"
