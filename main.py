from slack_bolt.adapter.socket_mode import SocketModeHandler
from config import SLACK_APP_TOKEN, logger
from bot import app as slack_app
import handlers

from flask import Flask, render_template, request, jsonify
from ai_engine import generate_chat_response
from retrieval import retrieve
import traceback
import threading
import sys
import os

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
    return jsonify({"status": "ok"}), 200

@flask_app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message')
    
    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    try:
        result = retrieve(user_query, mode="internal", top_k=3, min_relevance=0.45)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])
        documents = result.get("documents", [])

        answer = generate_chat_response(user_query, contexts)
        if not contexts:
            citations = []
            documents = []
        return jsonify({"response": answer, "citations": citations, "documents": documents})
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@flask_app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    query = data.get("query", "")
    mode = data.get("mode", "internal")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = retrieve(query, mode=mode, top_k=3, min_relevance=0.45)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])
        documents = result.get("documents", [])

        answer = generate_chat_response(query, contexts)
        if not contexts:
            citations = []
            documents = []
        return jsonify({"answer": answer, "citations": citations, "documents": documents})
    except Exception as e:
        logger.error(f"QA error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

@flask_app.route('/api/retrieve', methods=['POST', 'GET'])
def api_retrieve():
    try:
        if request.method == 'POST':
            data = request.json or {}
            query = data.get("query", "")
            mode = data.get("mode", "internal")
            top_k = data.get("top_k", 5)
            min_relevance = data.get("min_relevance", 0.45)
        else:  # GET
            query = request.args.get("query", "")
            mode = request.args.get("mode", "internal")
            top_k = int(request.args.get("top_k", 5))
            min_relevance = float(request.args.get("min_relevance", 0.45))
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        result = retrieve(query, mode=mode, top_k=top_k, min_relevance=min_relevance)
        contexts = result.get("contexts", [])
        citations = result.get("citations", [])
        documents = result.get("documents", [])
        
        answer = generate_chat_response(query, contexts)
        
        response = {
            "answer": answer,
            "citations": citations,
            "documents": documents,
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
    logger.info("Starting Flask web application on http://0.0.0.0:8000")
    flask_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

def run_slack_bot():
    logger.info("Starting Socket Mode Handler...")
    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.start()

def run_tests():
    """Run pytest tests and return True if all tests pass, False otherwise."""
    try:
        import pytest
        logger.info("Running tests before starting application...")
        
        # Run pytest programmatically
        exit_code = pytest.main(["-v", "tests/", "--tb=short"])
        
        # pytest.main returns 0 if all tests pass, non-zero otherwise
        # Exit code 5 means no tests were collected (also treat as failure)
        if exit_code == 0:
            logger.info("All tests passed. Proceeding with application startup.")
            return True
        else:
            logger.error(f"Tests failed with exit code {exit_code}. Aborting application startup.")
            return False
    except ImportError:
        logger.warning("pytest not available. Skipping test check.")
        return True
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting Brainfish - Slack Bot + Web App")
    
    # Run tests before starting the application
    if not run_tests():
        logger.critical("Tests failed. Aborting application startup.")
        sys.exit(1)
    
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Run Slack bot in the main thread (blocking)
    run_slack_bot()
