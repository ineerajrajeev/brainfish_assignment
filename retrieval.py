from typing import List, Dict, Tuple
import re
from ai_engine import get_embedding, rerank_documents
from database import get_all_knowledge_docs
from config import logger
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

PUBLIC_SOURCES = {"docs", "tickets", "public_ticket"}
STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "and", "but", "if", "or", "because",
    "until", "while", "about", "against", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "tell", "me", "about"
}
BM25_CACHE = {"key": None, "bm25": None, "tokenized": []}


def is_public(metadata: Dict) -> bool:
    """Determine if a document can be cited in customer mode."""
    if not metadata:
        return False
    if metadata.get("public", False):
        return True
    source = metadata.get("source")
    return source in PUBLIC_SOURCES


def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text)
    return [w for w in words if w.lower() not in STOP_WORDS and len(w) > 1]


def tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def compute_keyword_score(query: str, doc_text: str) -> float:
    query_keywords = extract_keywords(query.lower())
    if not query_keywords:
        return 0.0
    
    doc_text_lower = doc_text.lower()
    
    matches = 0
    for kw in query_keywords:
        if kw in doc_text_lower:
            matches += 1
            if re.search(rf'\b{re.escape(kw)}\b', doc_text_lower):
                matches += 0.5
    
    return min(matches / len(query_keywords), 1.0)


def get_bm25(docs: List[Dict]):
    if BM25Okapi is None:
        return None
    if not docs:
        return None
    key = (len(docs), (docs[0].get("text", "")[:50] if docs else ""))
    if BM25_CACHE["key"] == key and BM25_CACHE["bm25"] is not None:
        return BM25_CACHE["bm25"]
    tokenized = [tokenize(d.get("text", "")) for d in docs]
    bm25 = BM25Okapi(tokenized)
    BM25_CACHE["key"] = key
    BM25_CACHE["bm25"] = bm25
    BM25_CACHE["tokenized"] = tokenized
    return bm25


def hybrid_score_documents(
    query: str,
    query_embedding,
    docs: List[Dict],
    top_k: int = 15,
    min_relevance: float = 0.45
) -> List[Tuple[float, Dict]]:
    if not docs:
        return []
    from sentence_transformers import util
    filtered_docs = [d for d in docs if d.get("vector")]
    if not filtered_docs:
        return []
    vectors = [d["vector"] for d in filtered_docs]
    vector_scores = util.cos_sim(query_embedding, vectors)[0]
    keyword_scores = [compute_keyword_score(query, d.get("text", "")) for d in filtered_docs]
    bm25 = get_bm25(filtered_docs)
    bm25_scores = []
    if bm25 is not None:
        raw_scores = bm25.get_scores(tokenize(query))
        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score > 0:
            bm25_scores = [s / max_score for s in raw_scores]
        else:
            bm25_scores = [0.0 for _ in raw_scores]
    else:
        bm25_scores = [0.0 for _ in filtered_docs]
    combined_scores = []
    for i, doc in enumerate(filtered_docs):
        vec_score = vector_scores[i].item()
        kw_score = keyword_scores[i]
        bm_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
        if kw_score > 0.5:
            combined = (vec_score * 0.35) + (kw_score * 0.45) + (bm_score * 0.20)
        elif kw_score > 0:
            combined = (vec_score * 0.50) + (kw_score * 0.30) + (bm_score * 0.20)
        else:
            combined = (vec_score * 0.70) + (bm_score * 0.30)
        combined_scores.append((combined, doc, vec_score, kw_score))
    combined_scores.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"[Hybrid] Top 5 combined scores:")
    for i, (combined, doc, vec, kw) in enumerate(combined_scores[:5]):
        text_preview = doc.get("text", "")[:40]
        logger.info(f"   {i+1}. combined={combined:.3f} (vec={vec:.3f}, kw={kw:.3f}) | '{text_preview}...'")
    filtered = [(score, doc) for score, doc, _, _ in combined_scores if score >= min_relevance]
    return filtered[:top_k]


def retrieve(query: str, mode: str = "internal", top_k: int = 3, min_relevance: float = 0.45) -> Dict:
    logger.info(f"[Retrieval] Query: '{query[:50]}...' | Mode: {mode} | top_k: {top_k}")

    docs = get_all_knowledge_docs()
    logger.info(f"[Retrieval] Total documents in DB: {len(docs)}")

    if not docs:
        logger.warning("[Retrieval] No documents in knowledge base")
        return {"answer": "No knowledge available", "citations": [], "contexts": []}

    query_vec = get_embedding(query)
    initial_k = top_k * 5
    scored_docs = hybrid_score_documents(
        query,
        query_vec,
        docs,
        top_k=initial_k,
        min_relevance=min_relevance
    )
    
    logger.info(
        f"[Retrieval] Found {len(scored_docs)} candidates from hybrid search (threshold={min_relevance})"
    )

    if scored_docs:
        candidate_docs = [doc for _, doc in scored_docs]
        final_docs_list = rerank_documents(query, candidate_docs, top_k=top_k)
        final_docs = [(0.99, d) for d in final_docs_list]
        
        logger.info(f"[Retrieval] Re-ranked to top {len(final_docs)}")
    else:
        final_docs = []

    for i, (_, doc) in enumerate(final_docs[:3]):
        text_preview = doc.get("text", "")[:50]
        source = doc.get("metadata", {}).get("source", "unknown")
        logger.info(f"   Final {i+1}: source={source} | text='{text_preview}...'")

    if mode == "customer":
        safe_docs = [(score, doc) for score, doc in final_docs if is_public(doc.get("metadata", {}))]
        logger.info(f"[Retrieval] Customer mode: {len(safe_docs)} public docs after filtering")
        
        if not safe_docs:
            return {
                "answer": "Insufficient public sources available.",
                "contexts": [],
                "citations": []
            }
        
        final_docs = safe_docs

    else:
        if not final_docs:
            logger.info("[Retrieval] No documents available above threshold")
            return {"answer": "No relevant documents found.", "contexts": [], "citations": []}

    contexts = [doc["text"] for _, doc in final_docs]
    citations = [doc.get("metadata", {}) for _, doc in final_docs]
    documents = [{"text": doc["text"], "metadata": doc.get("metadata", {})} for _, doc in final_docs]
    unique_sources = list({meta.get("source", "unknown") for meta in citations})

    logger.info(f"[Retrieval] Returning {len(contexts)} context(s) from sources: {unique_sources}")

    return {
        "contexts": contexts,
        "citations": citations,
        "documents": documents,
        "unique_sources": unique_sources
    }
