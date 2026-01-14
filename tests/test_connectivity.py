import os
import pytest
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


@pytest.mark.skipif(not os.getenv("MONGO_URI"), reason="MONGO_URI not set")
def test_mongo_connection():
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
    finally:
        client.close()


@pytest.mark.skipif(not os.getenv("SLACK_BOT_TOKEN"), reason="SLACK_BOT_TOKEN not set")
def test_slack_auth():
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    try:
        resp = client.auth_test()
        assert resp.get("ok", False)
    except SlackApiError as e:
        pytest.fail(f"Slack auth failed: {e.response['error']}")


@pytest.mark.skipif(not os.getenv("SLACK_BOT_TOKEN"), reason="SLACK_BOT_TOKEN not set")
def test_list_channels_smoke():
    # Light-weight connectivity check using list channels
    token = os.getenv("SLACK_BOT_TOKEN")
    client = WebClient(token=token)
    try:
        resp = client.conversations_list(limit=1)
        assert resp.get("ok", False)
    except SlackApiError as e:
        pytest.fail(f"Slack conversations_list failed: {e.response['error']}")
