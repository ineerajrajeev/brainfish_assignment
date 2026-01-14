# Brainfish Knowledge Assistant

A RAG (Retrieval-Augmented Generation) system that ingests Slack conversations and documents, stores them as embeddings in MongoDB, and provides Q&A with traceable citations and customer-safe mode.

![Gemma MLX Model Ingestion Flow](docs/Gemma MLX Ingestion Flow-2026-01-14-142636.png)

## Features

| Feature | Status |
|---------|--------|
| Slack connector (channels, threads, replies) | Yes |
| Document ingestion (PDF, DOCX, TXT, MD) | Yes |
| Thread-aware chunking | Yes |
| LLM-based worthiness classification | Yes |
| Source-of-truth detection | Yes |
| Embeddings + MongoDB vector storage | Yes |
| REST API for Q&A | Yes |
| Web UI with markdown rendering | Yes |
| Internal vs Customer citation modes | Yes |
| Customer-safe citation policy | Yes |
| Message edit/delete sync | Yes |
| Deduplication (in-memory + DB) | Yes |
| Docker support | Yes |
| Unit tests | Yes |

## Quick Start

### Local Development

1. Create `.env`:
```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=...
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=ai_assistant_db
# Model config (choose one):
MLX_MODEL_PATH=mlx_model  # local quantized model
# HF_MODEL_ID=google/gemma-2b  # or download from HF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run:
```bash
python main.py
```
This starts both the Slack bot and Flask server on port 8000.

### Docker

```bash
docker compose up --build
```

Services:
- `app`: Python application (Slack bot + Flask API on `:8000`)
- `mongo`: MongoDB 7 with persistent volume

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with internal context |
| `/qa` | POST | Q&A with mode selection |
| `/api/retrieve` | GET/POST | Full retrieval API |

### `/chat` Example
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I configure the API?"}'
```

Response:
```json
{
  "response": "Based on the documentation...",
  "citations": [
    {"source": "docs", "filename": "api-guide.pdf", "public": true}
  ]
}
```

### `/qa` Example (Customer Mode)
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the pricing tiers?", "mode": "customer"}'
```

Response with customer-safe citations only.

## Slack Commands

| Command | Description |
|---------|-------------|
| `@aiOS:PUSH <text>` | Force-ingest message to knowledge DB (bypasses worthiness check) |
| `@aiOS:ASK <question>` | Answer in-thread using all sources (internal mode) |

## Definitions

### Source of Truth
Messages or documents marked `source_of_truth=True`:
- Content from `#final_changes` channel (confirmed decisions)
- Content from `#docs` channel (official documentation)
- Messages using `@aiOS:PUSH` command

These represent **final decisions, confirmed fixes, specifications, or approved documentation**.

### Knowledge Worth Indexing
Content classified as valuable by the LLM classifier:
- **DOCUMENT**: Technical facts, specs, fixes, how-to guides
- **BUG**: Bug reports with reproduction steps or error details
- **IDEA**: Feature requests or product suggestions
- **FEEDBACK**: Constructive user feedback

Content classified as **NOISE** is discarded:
- Casual chatter, greetings, acknowledgments
- Scheduling messages
- Vague or incomplete thoughts

### Contradiction & Update Handling
- Messages with the same `ts` (timestamp) overwrite prior entries
- Message edits trigger re-embedding of the content
- Message deletions remove corresponding entries from the database
- Thread updates re-process the entire thread context

## Customer-Safe Citation Policy

| Mode | Citable Sources | Behavior |
|------|-----------------|----------|
| `internal` | All sources | Cites any document |
| `customer` | `public=True` or source in `{docs, tickets}` | Filters internal discussions |

If no public sources meet the relevance threshold (30%), the system returns:
```json
{"answer": "No relevant documents found.", "citations": []}
```

## Testing

```bash
pytest
```

### Test Coverage
- **API tests**: Validates REST endpoint response formats
- **Connectivity tests**: MongoDB and Slack connection checks (skip if env vars missing)
- **LLM tests**: Gemma generation smoke test (enable with `RUN_LLM_TESTS=1`)

Environment variables for tests:
```bash
MONGO_URI=mongodb://localhost:27017/  # Required for DB tests
SLACK_BOT_TOKEN=xoxb-...              # Required for Slack tests
RUN_LLM_TESTS=1                       # Enable LLM tests
```

## Project Structure

```
brainfish/
- main.py              # Entry point (Slack + Flask)
- config.py            # Environment config
- bot.py               # Slack Bolt app setup
- handlers.py          # Slack event handlers
- ai_engine.py         # Embeddings + LLM generation
- retrieval.py         # Vector search + citation policy
- database.py          # MongoDB connection
- chunking_engine.py   # Thread-aware chunking
- templates/
  - index.html       # Web UI
- tests/
  - test_api.py
  - test_connectivity.py
  - test_llm.py
- requirements.txt
- Dockerfile
- docker-compose.yml
```

## Tradeoffs & Limitations

| Tradeoff | Current Approach | Alternative |
|----------|------------------|-------------|
| Vector DB | MongoDB (in-process cosine sim) | Pinecone/pgvector for scale |
| LLM | Gemma via MLX (macOS only) | OpenAI/Anthropic API for cross-platform |
| Classifier | LLM-based (slow but accurate) | Rules-based (faster, less accurate) |
| Deduplication | In-memory + DB check | Redis for distributed setups |

## Next Steps

1. **Hybrid retrieval**: Combine vector search with BM25 keyword matching
2. **Reranking**: Add a cross-encoder reranker for better relevance
3. **Incremental sync**: Periodic Slack history backfill
4. **Evaluation harness**: Automated retrieval quality metrics
5. **Policy tagging workflow**: Admin UI for marking sources as public/internal
