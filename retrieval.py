from typing import List, Dict, Tuple
from ai_engine import get_embedding
from database import get_all_knowledge_docs
from config import logger

PUBLIC_SOURCES = {"docs", "tickets", "public_ticket"}


def is_public(metadata: Dict) -> bool:
    """Determine if a document can be cited in customer mode."""
    if not metadata:
        return False
    if metadata.get("public", False):
        return True
    source = metadata.get("source")
    return source in PUBLIC_SOURCES


def score_documents(query_embedding, docs: List[Dict], top_k: int = 5, min_relevance: float = 0.25) -> Tuple[List[Tuple[float, Dict]], List[float]]:
    """
    Returns a list of (score, doc) sorted by score desc, filtered by min_relevance,
    plus the raw score list for average calculations.
    """
    if not docs:
        return [], []
    from sentence_transformers import util
    vectors = [d.get("vector", []) for d in docs if d.get("vector")]
    filtered_docs = [d for d in docs if d.get("vector")]
    if not vectors:
        return [], []
    scores = util.cos_sim(query_embedding, vectors)[0]
    
    # Create all pairs with scores
    all_pairs: List[Tuple[float, Dict]] = []
    for i, score in enumerate(scores):
        val = score.item()
        all_pairs.append((val, filtered_docs[i]))
    
    # Sort by score descending
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Log top 5 scores for debugging
    if all_pairs:
        logger.info(f"[Scoring] Top 5 scores: {[f'{p[0]:.3f}' for p in all_pairs[:5]]}")
    
    # Filter by min_relevance
    filtered_pairs = [(s, d) for s, d in all_pairs if s >= min_relevance]
    
    # If nothing meets threshold, return top-k anyway with warning
    if not filtered_pairs and all_pairs:
        logger.warning(f"[Scoring] No docs above {min_relevance}, returning top {top_k} anyway")
        return all_pairs[:top_k], [s.item() for s in scores]
    
    return filtered_pairs[:top_k], [s.item() for s in scores]


def retrieve(query: str, mode: str = "internal", top_k: int = 5, min_relevance: float = 0.25) -> Dict:
    """
    Retrieve top documents with strict mode-based filtering and citation policies.
    
    Args:
        query (str): The search string.
        mode (str): 'internal' (full access) or 'customer' (public safe only).
        top_k (int): Number of documents to retrieve.
        min_relevance (float): Baseline score for individual document inclusion.

    Returns:
        Dict: Contains 'contexts' (list of text), 'citations' (list of metadata),
              and 'unique_sources' (set of source names for internal use).
    """
    logger.info(f"[Retrieval] Query: '{query[:50]}...' | Mode: {mode} | top_k: {top_k}")

    docs = get_all_knowledge_docs()
    logger.info(f"[Retrieval] Total documents in DB: {len(docs)}")

    if not docs:
        logger.warning("[Retrieval] No documents in knowledge base")
        return {"answer": "No knowledge available", "citations": [], "contexts": []}

    query_vec = get_embedding(query)
    scored_docs, all_scores = score_documents(query_vec, docs, top_k=top_k, min_relevance=min_relevance)
    
    logger.info(f"[Retrieval] Found {len(scored_docs)} docs for context")

    # Log top scored documents
    for i, (score, doc) in enumerate(scored_docs[:3]):
        text_preview = doc.get("text", "")[:50]
        source = doc.get("metadata", {}).get("source", "unknown")
        logger.info(f"   Match {i+1}: score={score:.3f} | source={source} | text='{text_preview}...'")

    if mode == "customer":
        safe_docs = [(score, doc) for score, doc in scored_docs if is_public(doc.get("metadata", {}))]
        logger.info(f"[Retrieval] Customer mode: {len(safe_docs)} public docs after filtering")

        if not safe_docs:
            return {
                "answer": "Insufficient public sources available.",
                "contexts": [],
                "citations": []
            }
        avg_relevance = sum(score for score, _ in safe_docs) / len(safe_docs)
        logger.info(f"[Retrieval] Average relevance of public docs: {avg_relevance:.3f}")
        
        # Only reject if average is very low
        if avg_relevance < 0.15:
            return {
                "answer": "No relevant documents found.",
                "contexts": [],
                "citations": []
            }
        final_docs = safe_docs

    else:
        final_docs = scored_docs
        
        if not final_docs:
            logger.info("[Retrieval] No documents available")
            return {"answer": "No relevant documents found.", "contexts": [], "citations": []}

    contexts = [doc["text"] for _, doc in final_docs]
    citations = [doc.get("metadata", {}) for _, doc in final_docs]
    unique_sources = list({meta.get("source", "unknown") for meta in citations})

    logger.info(f"[Retrieval] Returning {len(contexts)} context(s) from sources: {unique_sources}")

    return {
        "contexts": contexts,
        "citations": citations,
        "unique_sources": unique_sources
    }
