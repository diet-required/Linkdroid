"""
server.py
=========
WebSocket server that bridges the Godot game engine frontend and the
Linkdroid LangGraph agent backend.

Message flow (per turn):
  1. Godot sends JSON: {"user_input": "<text>"}
  2. Server sends an immediate {"speech": "[THINKING]", ...} ACK.
  3. On the first message of a session, the last 5 memory pairs are fetched
     and summarised so the agent has cross-session continuity.
  4. The user message is appended to the in-memory chat history and the
     LangGraph agent is invoked asynchronously.
  5. The final AIMessage text is extracted, saved to long-term memory, and
     the user profile update is kicked off in a background thread.
  6. The response packet is sent back to Godot:
     {"speech": "<text>", "emotion_status": "ok", "hormone_data": {...}}

WebSocket endpoint: ws://localhost:8765
"""

import sqlite3
import array
import json
import asyncio
import websockets
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from agent_graph import linkdroid_core
import agent_graph
from agent_graph import run_user_profile_update

from datetime import datetime
from tools import get_raw_embedding, MEMORY_DB_PATH
from memory_utils import get_session_opening_context

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# In-memory conversation history shared across the session.
# Compressed after each turn to prevent unbounded growth.
godot_chat_history: list[BaseMessage] = []

# Reference to the currently connected Godot WebSocket client.
# Only one client is expected at a time (single-player desktop app).
CONNECTED_GODOT = None

# Flag that prevents fetching the session opening context more than once
# per server session (fetching is expensive – it calls the summariser LLM).
_session_context_loaded = False


# ---------------------------------------------------------------------------
# Chat history compression
# ---------------------------------------------------------------------------

def compress_chat_history(
    history: list[BaseMessage],
    max_turns: int = 10,
) -> list[BaseMessage]:
    """
    Compress the conversation history to keep context manageable.

    Two operations are applied:
      1. Multimodal pack strings and oversized ToolMessage contents are
         replaced with short human-readable summaries to free context tokens.
      2. The list is sliced to at most *max_turns* * 2 messages (user + AI
         per turn) and trimmed further to avoid starting on a dangling
         ToolMessage.

    Args:
        history:   Raw conversation history from the graph's final state.
        max_turns: Maximum number of complete user/AI turns to retain.

    Returns:
        A compressed copy of the history list.
    """
    compressed: list[BaseMessage] = []

    for msg in history:
        # Guard: some messages may store content as non-strings (e.g. lists).
        if not (hasattr(msg, "content") and isinstance(msg.content, str)):
            compressed.append(msg)
            continue

        # Replace multimodal pack strings with a compact filename reference.
        # The full base64 blob is no longer needed after the agent has seen it.
        if "__MULTIMODAL_PACK__" in msg.content:
            parts = msg.content.split("|")
            fname = parts[3] if len(parts) >= 4 else "unknown"
            new_content = f"[Media analysis completed: {fname}]"

            if isinstance(msg, ToolMessage):
                compressed.append(ToolMessage(
                    content=new_content,
                    tool_call_id=msg.tool_call_id,
                    status=msg.status,
                ))
            else:
                compressed.append(msg.__class__(content=new_content))
            continue

        # Truncate large (non-multimodal) ToolMessage outputs.
        if isinstance(msg, ToolMessage) and len(msg.content) > 300:
            compressed.append(ToolMessage(
                content=msg.content[:300] + "...(truncated)",
                tool_call_id=msg.tool_call_id,
                status=msg.status,
            ))
            continue

        compressed.append(msg)

    # ---- Slice to max_turns ------------------------------------------------
    if len(compressed) <= max_turns * 2:
        return compressed

    sliced = compressed[-(max_turns * 2):]

    # Never start the history with a dangling ToolMessage (no prior tool call
    # to attach it to would confuse the LLM).
    while sliced and isinstance(sliced[0], ToolMessage):
        sliced = sliced[1:]

    return sliced


# ---------------------------------------------------------------------------
# Long-term memory persistence
# ---------------------------------------------------------------------------

async def save_chat_to_memory(
    user_query: str,
    ai_response: str,
    emotions_dict: dict,
) -> None:
    """
    Persist the current conversation turn to the long-term memory database.

    Each row stores:
      - Timestamp
      - Raw user query
      - Raw AI response
      - Serialised emotion/hormone state at response time
      - Embedding vector of the combined "User: … / AI: …" text
        (used later for semantic similarity search)

    The embedding call and database write are both offloaded to a thread pool
    via asyncio.to_thread() so they don't block the event loop.

    Args:
        user_query:   The user's raw input text.
        ai_response:  The agent's final text response for this turn.
        emotions_dict: The hormone score dict from agent_graph.last_hormone_result.
    """
    # Skip empty turns (e.g. tool-only exchanges with no meaningful text).
    if not user_query.strip() or not ai_response.strip():
        return

    print("[server] saving conversation turn to long-term memory...")

    try:
        # Embed the combined turn so we can do semantic recall later.
        text_to_embed = f"User: {user_query}\nAI: {ai_response}"
        vector        = await asyncio.to_thread(get_raw_embedding, text_to_embed)
        # Serialise the float list as a compact binary blob (array of doubles).
        buf = array.array("d", vector).tobytes()

        def _db_write() -> None:
            """Synchronous SQLite write – runs in a thread pool worker."""
            conn   = sqlite3.connect(MEMORY_DB_PATH)
            cursor = conn.cursor()

            # Auto-create the table on first run.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    user_query  TEXT    NOT NULL,
                    ai_response TEXT    NOT NULL,
                    emotions    TEXT    NOT NULL,
                    embedding   BLOB    NOT NULL
                )
            """)

            cursor.execute(
                "INSERT INTO long_term_memories "
                "(timestamp, user_query, ai_response, emotions, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_query,
                    ai_response,
                    json.dumps(emotions_dict if emotions_dict else {}),
                    buf,
                ),
            )
            conn.commit()
            conn.close()

        await asyncio.to_thread(_db_write)
        print("[server] memory saved successfully.")

    except Exception as e:
        print(f"[server] memory save error: {e}")


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def godot_websocket_handler(websocket) -> None:
    """
    Handle all messages from a single connected Godot client.

    This coroutine runs for the lifetime of the WebSocket connection.
    On connection close (normal or abnormal) the CONNECTED_GODOT reference
    is cleared so the next client can connect cleanly.

    Protocol:
        Incoming: JSON with key "user_input" (str)
        Outgoing ACK: {"speech": "[THINKING]", "emotion_status": "processing"}
        Outgoing response: {
            "speech":         "<text>",
            "emotion_status": "ok",
            "hormone_data":   {<hormone scores>}
        }

    Args:
        websocket: The websockets library connection object.
    """
    global CONNECTED_GODOT, godot_chat_history, _session_context_loaded
    print("[server] Godot client connected.")
    CONNECTED_GODOT = websocket

    try:
        async for message in websocket:
            data      = json.loads(message)
            user_text = data.get("user_input", "").strip()
            print(f"[server] received: {data}")

            if not user_text:
                continue   # Ignore empty pings.

            # Immediately acknowledge so the Godot UI shows a "thinking" state.
            await websocket.send(json.dumps({
                "speech":         "[THINKING]",
                "emotion_status": "processing",
            }))

            # ---- Session context injection (first message only) ------------
            # Fetch and summarise the last 5 memory pairs so the agent has
            # continuity from the previous session.  We do this only once to
            # avoid an expensive LLM call on every turn.
            session_context = ""
            if not _session_context_loaded:
                _session_context_loaded = True
                try:
                    session_context = await asyncio.to_thread(get_session_opening_context)
                    if session_context:
                        print("[server] previous session context loaded.")
                except Exception as e:
                    print(f"[server] session context load error: {e}")

            # Append the user's message to the shared history.
            godot_chat_history.append(HumanMessage(content=user_text))

            # Guard against the graph not yet being compiled (race condition
            # on startup – should not happen in practice but handled gracefully).
            if linkdroid_core is None:
                await websocket.send(json.dumps({
                    "speech":         "The agent graph is still initializing.",
                    "emotion_status": "waiting",
                }))
                continue

            # Reset the cached hormone result so we can detect if reflection
            # failed to populate it this turn.
            agent_graph.last_hormone_result = None

            # ---- Build initial graph state ---------------------------------
            # The session context is passed via dynamic_prompt so reflection_node
            # can extract and carry it forward into the assembled system prompt.
            initial_state = {
                "messages": godot_chat_history,
                "target_type": "THIRD_PARTY",        # Overwritten by reflection_node.
                "hidden_thought": "",                 # Overwritten by reflection_node.
                "dynamic_prompt": (
                    f"[Previous session summary]\n{session_context}\n\n"
                    if session_context else ""
                ),
            }

            # ---- Invoke the LangGraph agent (async) ------------------------
            final_state = await linkdroid_core.ainvoke(
                initial_state,
                config={"recursion_limit": 15},
            )

            # Compress history and update the shared list for next turn.
            godot_chat_history = compress_chat_history(
                final_state["messages"], max_turns=10
            )

            # ---- Extract the final response text ---------------------------
            # Walk the history in reverse to find the most recent AIMessage
            # that contains actual text (skip tool-call-only messages).
            final_response = "Something went wrong in the response pipeline."
            for msg in reversed(godot_chat_history):
                if isinstance(msg, AIMessage) and msg.content:
                    raw = msg.content

                    if isinstance(raw, list):
                        # Multimodal content list – join text chunks.
                        final_response = "".join(
                            p["text"] if isinstance(p, dict) and "text" in p else str(p)
                            for p in raw
                        ).strip()
                    elif isinstance(raw, dict):
                        final_response = str(raw.get("text", raw)).strip()
                    else:
                        final_response = str(raw).strip()

                    break

            # ---- Persist this turn to long-term memory ---------------------
            await save_chat_to_memory(
                user_query=user_text,
                ai_response=final_response,
                emotions_dict=agent_graph.last_hormone_result,
            )

            # ---- Background: update user profile (non-blocking) ------------
            # This runs in a thread pool so it doesn't delay the response.
            now      = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M")
            asyncio.create_task(
                asyncio.to_thread(run_user_profile_update, godot_chat_history, time_str)
            )

            # ---- Send response packet to Godot -----------------------------
            response_packet = {
                "speech":       final_response,
                "emotion_status": "ok",
                "hormone_data": agent_graph.last_hormone_result,
            }
            await websocket.send(json.dumps(response_packet))
            print("[server] response sent.")

            if agent_graph.last_hormone_result:
                print(f"[server] hormone state: {agent_graph.last_hormone_result}")

    except websockets.exceptions.ConnectionClosed:
        print("[server] Godot client disconnected.")
    finally:
        # Always clean up the connection reference on exit.
        CONNECTED_GODOT = None


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def start_websocket_server() -> None:
    """
    Start the WebSocket server and run it indefinitely.

    Binds to localhost:8765.  Only one client (the Godot frontend) is
    expected to connect at a time; concurrent connections are technically
    supported by the websockets library but not tested.

    This coroutine never returns under normal operation – it awaits an
    asyncio.Future() that is never resolved, keeping the event loop alive.
    """
    async with websockets.serve(godot_websocket_handler, "localhost", 8765):
        print("[server] WebSocket server listening on port 8765.")
        await asyncio.Future()   # Run forever.
