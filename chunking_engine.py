class ChunkingEngine:
    def __init__(self, max_chunk_size=500):
        self.max_chunk_size = max_chunk_size

    def chunk_slack_thread(self, messages):
        """
        Strategy: Concatenate a Slack thread into a single 'Story'.
        Input: A list of Slack message objects sorted by time.
        """
        full_conversation = []
        for msg in messages:
            user = msg.get("user", "Unknown")
            text = msg.get("text", "")
            full_conversation.append(f"User {user}: {text}")
        
        # Join them into one block. 
        # Ideally, this is ONE chunk. If it's huge, we split by logical turns.
        combined_text = "\n".join(full_conversation)
        return [combined_text]

    def chunk_document(self, text):
        """
        Strategy: Split by Paragraphs (\n\n), not just characters.
        This keeps sections together.
        """
        # 1. Split by double newline (paragraphs)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph keeps us under the limit, add it
            if len(current_chunk) + len(para) < self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Chunk is full, save it and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        # Add the last leftover chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
