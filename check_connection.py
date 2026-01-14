import os
import ssl
import certifi
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SSL Fix for macOS
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Initialize the Slack Client
client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"), ssl=ssl_context)

def list_channels():
    print("Fetching channel list from Slack...\n")

    try:
        response = client.conversations_list(types="public_channel")

        channels = response["channels"]

        print(f"{'CHANNEL NAME':<25} | {'CHANNEL ID':<15}")
        print("-" * 45)

        for channel in channels:
            name = channel["name"]
            id = channel["id"]
            is_member = channel["is_member"]
            member_status = "(Member)" if is_member else "(Not Member)"
            print(f"#{name:<24} | {id} {member_status}")

    except SlackApiError as e:
        print(f"Error fetching channels: {e.response['error']}")

if __name__ == "__main__":
    if not os.environ.get("SLACK_BOT_TOKEN"):
        print("Error: SLACK_BOT_TOKEN is missing from .env")
    else:
        list_channels()