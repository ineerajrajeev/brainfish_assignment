from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from config import MONGO_URI, MONGO_DB_NAME, logger

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping') # Trigger connection check
    db = mongo_client[MONGO_DB_NAME]
    knowledge_col = db["knowledge"]
    ideas_col = db["ideas"]
    logger.info(f"Connected to MongoDB: {MONGO_DB_NAME}")

    def get_all_knowledge_docs():
        """
        Fetches all documents from the knowledge collection.
        Returns a list of dicts containing text, vector, and metadata.
        """
        if knowledge_col is None:
            return []
        # Projection to fetch only necessary fields
        return list(knowledge_col.find({}, {"text": 1, "vector": 1, "metadata": 1, "_id": 0}))

except ConnectionFailure as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None
    knowledge_col = None
    ideas_col = None
