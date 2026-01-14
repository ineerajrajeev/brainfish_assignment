import os
import logging
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AI_Assistant")

# Load environment variables
load_dotenv()

# Configuration Constants
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "ai_assistant_db")

# Model loading
# If HF_MODEL_ID is set, we load directly from Hugging Face (no quantization).
# Otherwise we load from the local MLX_MODEL_PATH folder.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")  # e.g., "Qwen/Qwen2.5-VL-3B-Instruct"
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH", "mlx_model")

# Channel Mapping
CHANNEL_MAP = {
    "customers_concerns": "C0A808L5Q6R",
    "final_changes": "C0A83LPSB7C",
    "docs": "C0A7N8H907R",
    "ideas": "C0A808R3C9K",
    "social": "C080YMEQK0W",
    "marketing": "C0A8Y0TLT08",
    "sales": "C0A879AHW1J",
    "top_secret": "C0A808JP38D",
    "customer_input": "C0A808R3C9K" 
}

# Validate Credentials
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    logger.critical("Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN in .env")
    exit(1)
