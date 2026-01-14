import datetime
import os
from io import BytesIO
import requests
from bot import app
from config import CHANNEL_MAP, logger
from database import db, knowledge_col, ideas_col
from ai_engine import get_embedding, analyze_text_with_mlx, generate_chat_response
from retrieval import retrieve
from slack_sdk.errors import SlackApiError
from chunking_engine import ChunkingEngine

# Initialize Chunking Engine
chunker = ChunkingEngine()

# Global cache for recent message IDs to handle rapid retries
# Simple in-memory set (ts + channel). For production, use Redis.
PROCESSED_MESSAGES = set()

# Cache for bot user ID (fetched once, reused)
BOT_USER_ID = None

def get_username(user_id, client):
    try:
        return client.users_info(user=user_id)["user"]["name"]
    except:
        return "unknown"


def fetch_and_extract_file_text(file_info, client):
    """
    Download a Slack file and extract text for PDFs, Word docs, txt, md.
    Returns a list of strings (pages for PDFs, single-element list otherwise).
    """
    # Prefer the direct download URL if present
    url = file_info.get("url_private_download") or file_info.get("url_private")
    name = file_info.get("name", "file")
    mimetype = file_info.get("mimetype", "")
    ext = os.path.splitext(name)[1].lower()

    if not url:
        logger.warning(f"No url_private for file: {name}")
        return ""

    try:
        # Important: The Authorization header must be set correctly.
        # "Bearer xoxb-..." is required for url_private/url_private_download.
        headers = {
            "Authorization": f"Bearer {client.token}",
            "Accept": "application/octet-stream",
        }
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        content = resp.content
        
        # Check if we got redirected to a login page (HTML) instead of the file
        if b"<html" in content[:200].lower() or b"<!doctype html" in content[:200].lower():
             logger.warning(f"Downloaded HTML instead of file content for {name}. Check bot token scopes (need files:read).")
             return ""
             
    except Exception as e:
        logger.warning(f"Failed to download file {name}: {e}")
        return ""

    # If we got HTML instead of the file (e.g., auth redirect), bail out
    snippet = content[:200].lower()
    if b"<html" in snippet or b"<!doctype html" in snippet:
        logger.warning(f"Downloaded HTML instead of file content for {name}. Check bot token scopes.")
        return ""

    # PDF
    if ext == ".pdf" or "pdf" in mimetype:
        try:
            from PyPDF2 import PdfReader
            # strict=False makes PyPDF2 more tolerant of minor PDF errors (e.g., EOF marker issues)
            reader = PdfReader(BytesIO(content), strict=False)
            pages = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                pages.append(txt)
            extracted = [p.strip() for p in pages if p.strip()]
            if extracted:
                return extracted
        except Exception as e:
            logger.warning(f"Failed to parse PDF {name} with PyPDF2: {e}")
            # As a fallback, try to decode as text (may work for simple PDFs)
            try:
                fallback_txt = content.decode("utf-8", errors="ignore")
                return [fallback_txt] if fallback_txt else []
            except Exception:
                return []

    # Word (.docx)
    if ext == ".docx" or "word" in mimetype:
        try:
            from docx import Document
            doc = Document(BytesIO(content))
            combined = "\n\n".join(p.text for p in doc.paragraphs)
            return [combined]
        except Exception as e:
            logger.warning(f"Failed to parse DOCX {name}: {e}")
            return []

    # Fallback: treat as text/markdown
    try:
        return [content.decode("utf-8", errors="ignore")]
    except Exception as e:
        logger.warning(f"Failed to decode text file {name}: {e}")
        return []

# Explicitly handle message_deleted events
@app.event({"type": "message", "subtype": "message_deleted"})
def handle_message_deletion(body, logger):
    """
    Handle message deletion events to clean up the database.
    """
    event = body.get("event", {})
    deleted_ts = event.get("previous_message", {}).get("ts")
    channel_id = event.get("channel")
    
    if deleted_ts and db is not None:
        logger.info(f"Message deleted in {channel_id} (ts: {deleted_ts}). Checking knowledge base...")

        # Check if exists before deleting to avoid redundant logs
        k_exists = knowledge_col.count_documents({"metadata.ts": deleted_ts})
        i_exists = ideas_col.count_documents({"metadata.thread_ts": deleted_ts})

        if k_exists == 0 and i_exists == 0:
             logger.info("   Document already removed or never existed.")
             return

        # Delete from knowledge collection
        res_k = knowledge_col.delete_one({"metadata.ts": deleted_ts})

        # Delete from ideas collection (using thread_ts as the key identifier stored in metadata)
        res_i = ideas_col.delete_one({"metadata.thread_ts": deleted_ts})

        # Also remove from processed messages cache to prevent reprocessing
        # Find all message keys that match this timestamp (could be in different channels)
        keys_to_remove = [key for key in PROCESSED_MESSAGES if key.endswith(f":{deleted_ts}")]
        for key in keys_to_remove:
            PROCESSED_MESSAGES.discard(key)

        if res_k.deleted_count > 0:
            logger.info(f"   Removed {res_k.deleted_count} doc(s) from Knowledge DB.")
        if res_i.deleted_count > 0:
            logger.info(f"   Removed {res_i.deleted_count} doc(s) from Ideas DB.")
        if keys_to_remove:
            logger.info(f"   Cleared {len(keys_to_remove)} entry/entries from processed messages cache.")

# Explicitly handle message_changed events (edited messages)
@app.event({"type": "message", "subtype": "message_changed"})
def handle_message_edit(body, logger, client):
    """
    Handle message edit events to update embeddings and text in the database.
    """
    event = body.get("event", {})
    updated_message = event.get("message", {})
    channel_id = event.get("channel")
    ts = updated_message.get("ts")
    text = updated_message.get("text", "")
    user = updated_message.get("user")
    
    if not ts or db is None:
        return
    
    logger.info(f"Message edited in {channel_id} (ts: {ts}). Updating knowledge base...")

    # Remove from processed cache so it can be re-processed if needed
    message_key = f"{channel_id}:{ts}"
    if message_key in PROCESSED_MESSAGES:
        PROCESSED_MESSAGES.remove(message_key)

    # Check which collections have entries for this ts
    k_count = knowledge_col.count_documents({"metadata.ts": ts})
    i_count = ideas_col.count_documents({"metadata.thread_ts": ts})

    if k_count == 0 and i_count == 0:
        logger.info(f"   No existing entries found for ts: {ts}. Treating as new message.")
        # Let the regular message handler process it
        # Create a no-op say function to avoid issues
        def noop_say(*args, **kwargs):
            pass
            
        # Ensure channel is in the message object to avoid KeyError
        updated_message["channel"] = channel_id
        
        handle_incoming_messages(updated_message, noop_say, client)
        return
    
    # Update knowledge collection entries
    if k_count > 0:
        logger.info(f"   Updating {k_count} entry/entries in Knowledge DB...")
        
        # Get all matching documents to understand their structure
        matching_docs = list(knowledge_col.find({"metadata.ts": ts}))
        
        # Separate file chunks from text-based entries
        file_chunks = [doc for doc in matching_docs if doc.get("metadata", {}).get("filename")]
        text_entries = [doc for doc in matching_docs if not doc.get("metadata", {}).get("filename")]
        
        # For docs channel: Don't update file chunks when message text is edited
        # (File content hasn't changed, only the message caption might have)
        if channel_id == CHANNEL_MAP.get("docs") and file_chunks:
            logger.info(f"   Skipping {len(file_chunks)} file chunk(s) - file content unchanged.")
        
        # Update text-based entries
        for doc in text_entries:
            # Re-analyze text if needed (for final_changes channel)
            if doc.get("metadata", {}).get("source") == "final_changes":
                analysis = analyze_text_with_mlx(text)
                classification = analysis.get("classification")
                
                if classification not in ['BUG', 'IDEA', 'FEEDBACK', 'DOCUMENT']:
                    logger.info(f"   Updated message classified as {classification} (NOISE). Removing from DB.")
                    knowledge_col.delete_one({"_id": doc["_id"]})
                    continue
            
            # Skip if text is empty (might be a file-only message)
            if not text.strip():
                logger.info(f"   Skipping entry - edited message has no text content.")
                continue
            
            # Update text and embedding
            new_embedding = get_embedding(text)
            knowledge_col.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "text": text,
                        "vector": new_embedding,
                        "timestamp": datetime.datetime.utcnow()
                    }
                }
            )
            logger.info(f"   Updated Knowledge DB entry (id: {doc['_id']})")
    
    # Update ideas collection entries (thread-based)
    if i_count > 0:
        logger.info(f"   Updating Ideas DB thread (ts: {ts})...")
        
        # For ideas, we need to re-process the entire thread to get updated context
        thread_ts = updated_message.get("thread_ts", ts)
        
        try:
            history = client.conversations_replies(channel=channel_id, ts=thread_ts)
            messages_list = history["messages"]
        except Exception as e:
            logger.warning(f"   Could not fetch thread history: {e}")
            messages_list = [updated_message]
        
        # Re-chunk the thread
        chunks = chunker.chunk_slack_thread(messages_list)
        
        # Delete old chunks and insert new ones
        ideas_col.delete_many({"metadata.thread_ts": thread_ts})
        
        for chunk_text in chunks:
            analysis = analyze_text_with_mlx(chunk_text)
            if analysis.get("classification") == "NOISE":
                continue
            
            meta = {
                "source": "ideas_channel",
                "thread_ts": thread_ts,
                "type": "thread_context",
                "source_of_truth": False,
                "public": False
            }
            ideas_col.insert_one({
                "text": chunk_text,
                "vector": get_embedding(chunk_text),
                "metadata": meta,
                "timestamp": datetime.datetime.utcnow()
            })
        
        logger.info(f"   Updated Ideas DB thread with {len(chunks)} chunk(s)")

    logger.info(f"   Finished updating message edits for ts: {ts}")

@app.message()
def handle_incoming_messages(message, say, client):
    channel_id = message["channel"]
    text = message.get("text", "")
    ts = message.get("ts")
    user = message.get("user")
    
    # --- 0. IMMEDIATE DEDUPLICATION GATE ---
    # Create a unique key for this specific message event
    message_key = f"{channel_id}:{ts}"
    
    # 1. Check in-memory cache (fastest, catches rapid retries)
    if message_key in PROCESSED_MESSAGES:
        logger.info(f"[Dedupe] Skipping duplicate event in memory: {ts}")
        return

    # 2. Check Database (persistent, catches server restarts)
    if db is not None:
        # Check if we have *any* record with this timestamp in either collection
        exists_k = knowledge_col.count_documents({"metadata.ts": ts})
        exists_i = ideas_col.count_documents({"metadata.thread_ts": ts})
        
        if exists_k > 0 or exists_i > 0:
            logger.info(f"[Dedupe] Message already exists in DB: {ts}")
            # Add to memory cache to skip DB check next time
            PROCESSED_MESSAGES.add(message_key)
            return

    # Mark as processing immediately
    PROCESSED_MESSAGES.add(message_key)
    
    # Log incoming event only after passing deduplication
    logger.info(f"[EVENT] Channel: {channel_id} | User: {user}")

    # --- 1. BOT MENTION INGESTION (Override Priority) ---
    # Dynamically get bot user ID (cached after first fetch)
    global BOT_USER_ID
    if BOT_USER_ID is None:
        try:
            auth_test = client.auth_test()
            BOT_USER_ID = auth_test.get("user_id", "")
            logger.info(f"Bot User ID cached: {BOT_USER_ID}")
        except Exception as e:
            logger.warning(f"Could not fetch bot user ID: {e}")
            BOT_USER_ID = ""

    # Check for bot mentions: either <@BOT_ID> format or @aiOS text
    bot_mention_pattern = f"<@{BOT_USER_ID}>" if BOT_USER_ID else ""
    is_mention = (bot_mention_pattern and bot_mention_pattern in text) or "@aiOS" in text

    if is_mention:
        logger.info(f"Mention detected (Bot ID: {BOT_USER_ID or 'N/A'}).")

        # Check for PUSH command (handles both :PUSH and :ASK patterns)
        if ":PUSH" in text:
            logger.info("PUSH command - Force storing message to knowledge DB (Bypassing checks).")
            if db is not None:
                # Clean text: remove bot mention and PUSH command
                clean_text = text
                if bot_mention_pattern:
                    clean_text = clean_text.replace(f"{bot_mention_pattern}:PUSH", "").replace(f"{bot_mention_pattern} :PUSH", "")
                clean_text = clean_text.replace("@aiOS:PUSH", "").replace("@aiOS :PUSH", "").replace(":PUSH", "").strip()
                
                if not clean_text:
                    clean_text = text  # Fallback to original if cleaning removed everything
                
                meta = {
                    "source": "mention_push",
                    "user": get_username(user, client),
                    "ts": ts,
                    "source_of_truth": True,
                    "public": True
                }
                knowledge_col.insert_one({
                    "text": clean_text,
                    "vector": get_embedding(clean_text),
                    "metadata": meta,
                    "timestamp": datetime.datetime.utcnow()
                })
                # React to confirm
                try:
                    client.reactions_add(channel=channel_id, name="floppy_disk", timestamp=ts)
                except SlackApiError:
                    pass 
            return # STOP PROCESSING - Override any channel specific logic

        elif ":ASK" in text:
            logger.info("ASK command - Responding with internal context.")
            
            # Extract query: remove bot mention and ASK command
            clean_query = text
            if bot_mention_pattern:
                clean_query = clean_query.replace(f"{bot_mention_pattern}:ASK", "").replace(f"{bot_mention_pattern} :ASK", "")
            clean_query = clean_query.replace("@aiOS:ASK", "").replace("@aiOS :ASK", "").replace(":ASK", "").strip()
            
            if not clean_query:
                clean_query = text  # Fallback to original if cleaning removed everything
            
            try:
                # Retrieve from ALL docs (internal mode)
                result = retrieve(clean_query, mode="internal", top_k=5, min_relevance=0.25)
                contexts = result.get("contexts", [])
                citations = result.get("citations", [])
                
                answer = generate_chat_response(clean_query, contexts)
                
                # Format citations
                citation_text = ""
                if citations:
                    citation_text = "\n\n*Sources:*\n"
                    for i, c in enumerate(citations, 1):
                        source = c.get("source", "unknown")
                        fname = c.get("filename", "")
                        citation_text += f"{i}. {source}"
                        if fname: citation_text += f" ({fname})"
                        citation_text += "\n"
                
                full_response = f"{answer}{citation_text}"
                
                # Reply in thread
                client.chat_postMessage(channel=channel_id, thread_ts=ts, text=full_response)
                
            except Exception as e:
                logger.error(f"Failed to process ASK command: {e}")
                try:
                    say("I encountered an error trying to answer that.")
                except:
                    pass
                
            return # STOP PROCESSING

        else:
            logger.info("Mention - storing message to knowledge DB.")
            if db is not None:
                meta = {
                    "source": "mention",
                    "user": get_username(user, client),
                    "ts": ts,
                    "source_of_truth": False,
                    "public": False
                }
                knowledge_col.insert_one({
                    "text": text,
                    "vector": get_embedding(text),
                    "metadata": meta,
                    "timestamp": datetime.datetime.utcnow()
                })
            return

    # --- 1. IGNORE LOGIC ---
    if channel_id in [CHANNEL_MAP["marketing"], CHANNEL_MAP["sales"], CHANNEL_MAP["top_secret"]]:
        return

    # --- 2. KNOWLEDGE INGESTION (Final Changes) ---
    if channel_id == CHANNEL_MAP["final_changes"]:
        logger.info("Processing 'Final Changes' (Source of Truth)")

        # Check for NOISE
        analysis = analyze_text_with_mlx(text)
        classification = analysis.get("classification")
        
        if classification not in ['BUG', 'IDEA', 'FEEDBACK', 'DOCUMENT']:
             logger.info(f"Skipping message classified as {classification} (NOISE/Invalid).")
             # Remove from cache so if they edit it later to be valid, we might process (optional)
             # PROCESSED_MESSAGES.remove(message_key) 
             return

        meta = {
            "source": "final_changes",
            "user": get_username(user, client),
            "ts": ts,
            "source_of_truth": True,
            "public": False
        }
        
        # Store in DB
        if db is not None:
            doc = {
                "text": text, 
                "vector": get_embedding(text), 
                "metadata": meta, 
                "timestamp": datetime.datetime.utcnow()
            }
            # Insert is safe because we already checked for existence at step 0
            knowledge_col.insert_one(doc)
            
            # React to confirm
            try:
                client.reactions_add(channel=channel_id, name="floppy_disk", timestamp=ts)
            except SlackApiError:
                pass # Ignore if already reacted

    # --- 3. DOCS INGESTION ---
    elif channel_id == CHANNEL_MAP["docs"]:
        files = message.get("files", [])
        if files:
            logger.info("File(s) detected in Docs")

            for f in files:
                name = f.get("name", "file")
                logger.info(f"Ingesting file: {name}")
                pages = fetch_and_extract_file_text(f, client)  # list of page texts (or single-element list)
                if not pages:
                    logger.info(f"No text extracted from file: {name}, skipping.")
                    continue

                # Store page-by-page (do not run through Gemma); each page is a chunk.
                if db is not None:
                    existing_chunks = knowledge_col.count_documents({"metadata.filename": name, "metadata.ts": ts})
                    if existing_chunks > 0:
                        logger.info(f"   File {name} (ts: {ts}) already ingested. Skipping DB insert.")
                    else:
                        for i, page_text in enumerate(pages):
                            meta = {
                                "source": "docs",
                                "filename": name,
                                "chunk_id": i,
                                "ts": ts,
                                "source_of_truth": True,
                                "public": True
                            }
                            knowledge_col.insert_one({
                                "text": page_text,
                                "vector": get_embedding(page_text),
                                "metadata": meta,
                                "timestamp": datetime.datetime.utcnow()
                            })
                        logger.info(f"   Ingested {len(pages)} page-chunks for {name}")

            try:
                client.reactions_add(channel=channel_id, name="page_facing_up", timestamp=ts)
            except Exception as e:
                logger.warning(f"Could not react on docs file ingestion: {e}")

    # --- 4. IDEA TRACKING ---
    elif channel_id == CHANNEL_MAP["ideas"]:
        logger.info("Processing 'Ideas' with Thread Awareness")
        
        thread_ts = message.get("thread_ts", message["ts"])
        
        # If this is a reply, we might want to re-process the whole thread context
        # But we must be careful not to duplicate the *parent* message in vector search.
        # Strategy: Delete old vector for this thread and insert new updated one.
        
        try:
            history = client.conversations_replies(channel=channel_id, ts=thread_ts)
            messages_list = history["messages"]
        except:
            messages_list = [message]

        chunks = chunker.chunk_slack_thread(messages_list)
        
        if db is not None:
            # OPTIONAL: Remove old chunks for this thread to avoid "ghost" partial conversations
            ideas_col.delete_many({"metadata.thread_ts": thread_ts})
            
            for chunk_text in chunks:
                analysis = analyze_text_with_mlx(chunk_text)
                if analysis.get("classification") == "NOISE": continue

                meta = {
                    "source": "ideas_channel",
                    "thread_ts": thread_ts,
                    "type": "thread_context",
                    "source_of_truth": False,
                    "public": False
                }
                ideas_col.insert_one({
                    "text": chunk_text, 
                    "vector": get_embedding(chunk_text), 
                    "metadata": meta, 
                    "timestamp": datetime.datetime.utcnow()
                })

    # --- 5. CUSTOMER AGENT (The Brain) ---
    # Triggered by DMs or the specific input channel
    elif channel_id == CHANNEL_MAP["customer_input"] or message.get("channel_type") == "im":

        logger.info("Agent Activated (Customer Input)")

        # User Feedback
        temp_msg = say(f"Analyzing...")
        
        # AI Analysis
        result = analyze_text_with_mlx(text)
        
        # Clean up 'Analyzing' message
        try:
            client.chat_delete(channel=channel_id, ts=temp_msg["ts"])
        except: pass
        
        cat = result.get("classification", "FEEDBACK")
        sent = result.get("sentiment", "NEUTRAL")
        summary = result.get("summary", text)
        
        if cat == "NOISE":
             say("I'm listening, but that didn't seem like a bug or feature request. Let me know if you need help!")
             return

        # Log to Social
        client.chat_postMessage(
            channel=CHANNEL_MAP["social"],
            text=f"Analysis: {cat} | {summary}"
        )

        # Routing
        if cat == "BUG":
            client.chat_postMessage(
                channel=CHANNEL_MAP["customers_concerns"],
                text=f"Bug: {text}"
            )
            say(f"Logged bug.")

        elif cat == "IDEA":
            # Simple Duplicate Check
            existing_idea = None
            if db is not None:
                # Use a simple text regex for demo (Vector search would go here)
                existing_idea = ideas_col.find_one({"text": {"$regex": text, "$options": "i"}})

            if existing_idea:
                client.chat_postMessage(
                    channel=CHANNEL_MAP["ideas"],
                    thread_ts=existing_idea["metadata"]["thread_ts"],
                    text=f"+1 Customer Vote: {summary}"
                )
                say("This idea is already being discussed. I've added your vote!")
            else:
                res = client.chat_postMessage(
                    channel=CHANNEL_MAP["ideas"],
                    text=f"*New Idea*\n{text}"
                )
                # Save the new idea thread reference
                if db is not None:
                     meta = {"source": "agent", "thread_ts": res["ts"]}
                     ideas_col.insert_one({"text": text, "vector": get_embedding(text), "metadata": meta, "timestamp": datetime.datetime.utcnow()})
                say("That's a great idea! I've started a new thread for it.")
        
        else:
            say("Feedback logged.")
