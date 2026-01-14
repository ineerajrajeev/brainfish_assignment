from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import SLACK_APP_TOKEN, logger
from bot import app as slack_app
# Import handlers to ensure they are registered with the app
import handlers

# Flask web application
from flask import Flask, render_template, request, jsonify
from ai_engine import generate_chat_response
from retrieval import retrieve
import traceback
import threading

# Create Flask app
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return f"<h1>Error loading page</h1><p>{str(e)}</p>", 500

@flask_app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

@flask_app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message')
    
    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Simple chat mode: retrieval + generation (internal mode)
        result = retrieve(user_query, mode="internal", top_k=5, min_relevance=0.25)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])

        # Always answer; if no contexts, respond conversationally without citations
        answer = generate_chat_response(user_query, contexts)
        if not contexts:
            citations = []
        return jsonify({"response": answer, "citations": citations})
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@flask_app.route('/qa', methods=['POST'])
def qa():
    """
    Retrieval + generation with citation policy.
    Body: { "query": "...", "mode": "internal" | "customer" }
    """
    data = request.json
    query = data.get("query", "")
    mode = data.get("mode", "internal")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = retrieve(query, mode=mode, top_k=5, min_relevance=0.25)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])

        # Always answer; if no contexts, respond politely and avoid citations
        answer = generate_chat_response(query, contexts)
        if not contexts:
            citations = []
        return jsonify({"answer": answer, "citations": citations})
    except Exception as e:
        logger.error(f"QA error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

@flask_app.route('/api/retrieve', methods=['POST', 'GET'])
def api_retrieve():
    """
    REST API endpoint for retrieval with answer and citations.
    
    POST Body (JSON):
    {
        "query": "your question",
        "mode": "internal" | "customer" (default: "internal"),
        "top_k": 5 (default: 5),
        "min_relevance": 0.25 (default: 0.25)
    }
    
    GET Query Parameters:
    - query: your question (required)
    - mode: "internal" | "customer" (default: "internal")
    - top_k: number of results (default: 5)
    - min_relevance: minimum relevance score (default: 0.25)
    
    Returns:
    {
        "answer": "generated answer",
        "citations": [
            {
                "source": "docs",
                "filename": "file.pdf",
                "ts": "1234567890.123456",
                ...
            }
        ],
        "query": "original query",
        "mode": "internal" | "customer",
        "num_sources": 2
    }
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            query = data.get("query", "")
            mode = data.get("mode", "internal")
            top_k = data.get("top_k", 5)
            min_relevance = data.get("min_relevance", 0.25)
        else:  # GET
            query = request.args.get("query", "")
            mode = request.args.get("mode", "internal")
            top_k = int(request.args.get("top_k", 5))
            min_relevance = float(request.args.get("min_relevance", 0.25))
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        # Retrieve relevant documents
        result = retrieve(query, mode=mode, top_k=top_k, min_relevance=min_relevance)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])
        
        # Generate answer
        answer = generate_chat_response(query, contexts)
        
        # Format response
        response = {
            "answer": answer,
            "citations": citations,
            "query": query,
            "mode": mode,
            "num_sources": len(citations)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Retrieval API error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

def run_flask_app():
    """Run Flask app in a separate thread"""
    logger.info("Starting Flask web application on http://0.0.0.0:8000")
    flask_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

def run_slack_bot():
    """Run Slack bot SocketModeHandler"""
    logger.info("Starting Socket Mode Handler...")
    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()

if __name__ == "__main__":
    logger.info("Starting Brainfish - Slack Bot + Web App")
    
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Run Slack bot in the main thread (blocking)
    run_slack_bot()
