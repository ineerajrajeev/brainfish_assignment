import re
import platform
from sentence_transformers import SentenceTransformer, util
from config import logger, MLX_MODEL_PATH, HF_MODEL_ID

# --- AI Model Loading (Global Scope) ---
embedding_model = None
mlx_model = None
mlx_tokenizer = None
USE_MLX = False

try:
    logger.info("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer loaded.")
    
    # MLX is only available on macOS with Apple Silicon
    if platform.system() == "Darwin":
        try:
            from mlx_lm import load, generate as mlx_generate
            model_source = HF_MODEL_ID if HF_MODEL_ID else MLX_MODEL_PATH
            logger.info(f"Loading MLX model from '{model_source}'...")
            mlx_model, mlx_tokenizer = load(model_source)
            USE_MLX = True
            logger.info("MLX model loaded successfully.")
        except ImportError:
            logger.warning("MLX not available. Using fallback mode.")
        except Exception as e:
            logger.warning(f"MLX model load failed: {e}. Using fallback mode.")
    else:
        logger.info("Non-macOS platform detected. MLX disabled, using fallback mode.")
        
except Exception as e:
    logger.critical(f"Failed to load embedding model: {e}")
    raise

def get_embedding(text):
    if not text or embedding_model is None:
        return []
    return embedding_model.encode(text).tolist()

def analyze_text_with_mlx(text):
    """
    Classifies input to determine if it is worth indexing.
    Returns classification: NOISE, DOCUMENT, BUG, IDEA, or FEEDBACK.
    """
    logger.info(f"[AI] Analyzing for indexing worthiness: {text[:40]}...")
    
    if not USE_MLX or mlx_model is None:
        # Fallback: simple keyword-based classification
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["bug", "error", "crash", "fix", "broken"]):
            return {"classification": "BUG", "summary": text[:50]}
        if any(kw in text_lower for kw in ["idea", "feature", "suggest", "could we", "what if"]):
            return {"classification": "IDEA", "summary": text[:50]}
        if any(kw in text_lower for kw in ["feedback", "review", "opinion", "think"]):
            return {"classification": "FEEDBACK", "summary": text[:50]}
        if len(text.split()) > 20:
            return {"classification": "DOCUMENT", "summary": text[:50]}
        return {"classification": "NOISE", "summary": text[:50]}
    
    from mlx_lm import generate
    
    system_prompt = (
        "You are a Knowledge Curator. Classify the input into exactly one word: NOISE, DOCUMENT, BUG, IDEA, or FEEDBACK.\n"
        "NOISE: casual chatter, greetings, short acks, scheduling, vague thoughts.\n"
        "DOCUMENT: technical facts, specs, fixes, how-to guides.\n"
        "BUG: bug reports with repro steps or errors.\n"
        "IDEA: feature requests or product suggestions.\n"
        "FEEDBACK: constructive user feedback.\n"
        "Output ONLY the single label word."
    )
    
    prompt = f"<start_of_turn>user\n{system_prompt}\n\nInput: {text}<end_of_turn>\n<start_of_turn>model\n"
    
    response_text = generate(
        mlx_model, 
        mlx_tokenizer, 
        prompt=prompt, 
        max_tokens=10,
        verbose=False
    )
    
    cleaned = response_text.strip().upper()
    cleaned = cleaned.split("<end_of_turn>")[0].strip()
    
    valid_classes = ["NOISE", "DOCUMENT", "BUG", "IDEA", "FEEDBACK"]
    for cls in valid_classes:
        if cls in cleaned:
            return {"classification": cls, "summary": text[:50]}
    
    logger.warning(f"Could not parse classification from: {cleaned[:50]}")
    return {"classification": "NOISE", "summary": text[:50]}

def find_best_matches(query_embedding, docs, top_k=3, min_relevance=0.85):
    """Finds the most relevant documents using cosine similarity."""
    if not docs:
        return []
    
    doc_vectors = [d['vector'] for d in docs if 'vector' in d and d['vector']]
    doc_texts = [d['text'] for d in docs if 'vector' in d and d['vector']]
    
    if not doc_vectors:
        return []

    scores = util.cos_sim(query_embedding, doc_vectors)[0]
    
    score_text_pairs = []
    for i, score in enumerate(scores):
        score_val = score.item()
        if score_val >= min_relevance:
            score_text_pairs.append((score_val, doc_texts[i]))
    
    if not score_text_pairs:
        return []
    
    score_text_pairs.sort(key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in score_text_pairs[:top_k]]

def generate_chat_response(query, context_texts):
    """Generates a response using the model and retrieved context."""
    logger.info(f"[AI] Generating chat response for: {query[:30]}...")

    # Log context details
    if context_texts:
        logger.info(f"[AI] Context received: {len(context_texts)} document(s)")
        for i, ctx in enumerate(context_texts):
            logger.info(f"   Doc {i+1}: {ctx[:80]}...")
    else:
        logger.info("[AI] No context documents provided")
    
    if not USE_MLX or mlx_model is None:
        # Fallback response without LLM - return context directly
        if context_texts:
            combined = "\n\n".join(context_texts[:3])
            return f"Based on the available knowledge:\n\n{combined[:1500]}"
        return "I'm currently running in fallback mode without the language model. Please check the server configuration."
    
    from mlx_lm import generate
    
    if context_texts:
        # Format context clearly with separators
        context_parts = []
        for i, text in enumerate(context_texts):
            # Truncate very long contexts to avoid token limits
            truncated = text[:800] if len(text) > 800 else text
            context_parts.append(f"[Document {i+1}]\n{truncated}")
        context_block = "\n\n".join(context_parts)
        
        user_prompt = (
            f"You are a knowledgeable assistant. Use the documents below to answer the question.\n"
            f"Synthesize information from the documents into a helpful, direct answer.\n"
            f"Do NOT say 'I don't know' - always provide an answer based on what's in the documents.\n\n"
            f"---DOCUMENTS---\n{context_block}\n---END DOCUMENTS---\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    else:
        user_prompt = f"You are a helpful AI assistant. Answer clearly and concisely.\n\nUser: {query}"
    
    prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Log the prompt length for debugging
    logger.info(f"[AI] Prompt length: {len(prompt)} chars")
    
    try:
        response = generate(
            mlx_model, 
            mlx_tokenizer, 
            prompt=prompt, 
            max_tokens=1024,
            stop_strings=["<end_of_turn>"],
            verbose=False
        )
    except TypeError:
        response = generate(
            mlx_model, 
            mlx_tokenizer, 
            prompt=prompt, 
            max_tokens=1024,
            verbose=False
        )
    
    cleaned = response.strip()
    if "<end_of_turn>" in cleaned:
        cleaned = cleaned.split("<end_of_turn>")[0]
    
    # Remove repetition patterns
    cleaned = re.sub(r'(.{30,}?)\1{2,}', r'\1', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # If model still returns empty or "I don't know", return context summary
    if not cleaned or len(cleaned) < 10 or "don't know" in cleaned.lower():
        logger.warning("[AI] Model returned empty/unhelpful response, using context summary")
        if context_texts:
            return f"Based on the available documents:\n\n{context_texts[0][:1000]}"
    
    return cleaned.strip()
