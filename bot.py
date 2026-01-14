import ssl
import certifi
from typing import Optional
from slack_bolt import App, BoltContext
from slack_bolt.authorization import AuthorizeResult
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, logger

# --- SSL Context for macOS ---
# Uses 'certifi' to provide valid root CAs for Python on Mac
try:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    logger.info("SSL Context initialized with Certifi trust store.")
except Exception as e:
    logger.critical(f"Failed to initialize SSL context: {e}")
    exit(1)

# Initialize the global WebClient with the secure SSL context
global_client = WebClient(token=SLACK_BOT_TOKEN, ssl=ssl_context)

def custom_authorize(
    context: BoltContext,
    enterprise_id: Optional[str],
    team_id: Optional[str],
    user_id: Optional[str]) -> AuthorizeResult:
    """
    Explicit Authorization Strategy to fix 'AuthorizeResult not found'.
    Fetches bot identity via auth.test() using the global client.
    """
    try:
        auth_response = global_client.auth_test()
        return AuthorizeResult(
            enterprise_id=enterprise_id,
            team_id=team_id,
            bot_id=auth_response.get("bot_id"),
            bot_user_id=auth_response["user_id"],
            bot_token=SLACK_BOT_TOKEN
        )
    except SlackApiError as e:
        logger.error(f"Authorization failed: {e}")
        raise e

# Initialize App with Custom Authorization and SSL Client
app = App(
    authorize=custom_authorize,
    signing_secret=SLACK_SIGNING_SECRET,
    client=global_client,
    process_before_response=True
)
