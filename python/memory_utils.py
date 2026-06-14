"""
memory_utils.py
===============
Shared memory retrieval and summarisation utilities.

This module is intentionally kept free of imports from agent_graph.py to
avoid circular dependency issues.  It is safe to import from:
  - tools.py         (get_raw_embedding is imported lazily inside functions)
  - server.py
  - agent_graph.py

Responsibilities:
  1. Retrieve rows from the long-term memory SQLite database (memory.db).
  2. Convert raw conversation rows into readable text.
  3. Summarise conversation windows using a lightweight LLM.
  4. Expose high-level recall helpers used by tools.py and server.py.

Environment variables consumed here:
  GOOGLE_API_KEY  – Google Generative AI API key (required).
  LIGHT_MODEL     – Gemini model used for summarisation
                    (default: "gemini-2.0-flash-lite").
"""

import os
import array
import sqlite3
import asyncio

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the SQLite file that stores all long-term conversation memories.
MEMORY_DB_PATH = "memory.db"

# Safety filter overrides – all categories set to BLOCK_NONE so the
# summariser can handle potentially sensitive conversation content without
# being blocked mid-summary.
_SAFETY = {
    "HARM_CATEGORY_HATE_SPEECH":       "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

# ---------------------------------------------------------------------------
# Summariser LLM (module-level singleton)
# ---------------------------------------------------------------------------

# A lightweight model is used here deliberately – summaries only need to be
# coherent, not creative.  Using a smaller model keeps latency and cost low.
_summarizer_llm = ChatGoogleGenerativeAI(
    model=os.getenv("LIGHT_MODEL", "gemini-2.0-flash-lite"),
    temperature=0.3,           # Low temperature → more deterministic summaries.
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    safety_settings=_SAFETY,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _llm_summarize(raw_block: str) -> str:
    """
    Summarise a raw block of conversation text into 4–5 sentences using the
    lightweight summariser LLM.

    The summary captures: the topic, conversation arc, and emotional tone /
    conclusion.  It is written in plain prose with no markdown or numbering so
    it can be embedded directly in a system prompt.

    Args:
        raw_block: Multi-line string with alternating "User: …" / "AI: …" lines.

    Returns:
        A 4–5 sentence plain-text summary, or a truncated fallback if the
        LLM call fails.
    """
    try:
        result = _summarizer_llm.invoke(
            "Below is a series of consecutive conversation exchanges.\n"
            "Summarize the entire conversation in 4 to 5 natural sentences.\n"
            "Include: what the topic was, how the conversation flowed, "
            "and what the emotional tone or conclusion was.\n"
            "Output the summary only — no numbering, no headers, no markdown:\n\n"
            f"{raw_block}"
        )
        raw = result.content

        # The Gemini SDK may return content as a list of text chunks;
        # join them into a single string.
        if isinstance(raw, list):
            return "".join(
                p["text"] if isinstance(p, dict) and "text" in p else str(p)
                for p in raw
            ).strip()

        return str(raw).strip()

    except Exception as e:
        print(f"[memory_utils] summarizer error: {e}")
        # Graceful fallback: return the first 400 characters of the raw block.
        return raw_block[:400] + "..."


def _rows_to_text(rows: list) -> str:
    """
    Convert a list of database rows into interleaved "User / AI" plain text.

    Expected row layout (from long_term_memories):
        (id, timestamp, user_query, ai_response, emotions, [embedding])

    The embedding column is optional; rows with fewer than 4 columns are
    handled gracefully with empty strings.

    Args:
        rows: List of tuples returned by an sqlite3 cursor.

    Returns:
        A multi-line string suitable for passing to _llm_summarize().
    """
    lines = []
    for row in rows:
        user_q = row[2] if len(row) > 2 else ""
        ai_r   = row[3] if len(row) > 3 else ""
        lines.append(f"User: {user_q}")
        lines.append(f"AI:   {ai_r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Window summarisation
# ---------------------------------------------------------------------------

def summarize_window(rows: list, center_idx: int, window: int = 2) -> str:
    """
    Extract a sliding window of rows centred on *center_idx* and summarise them.

    Default window=2 yields 5 rows total (centre ± 2), giving enough context
    for a meaningful summary without overwhelming the LLM's context window.

    Args:
        rows:       Full list of memory rows (chronological order).
        center_idx: Index of the most relevant row (e.g. top cosine-similarity hit).
        window:     Number of rows to include on each side of the centre.

    Returns:
        A plain-text LLM summary of the selected window.
    """
    start   = max(0, center_idx - window)
    end     = min(len(rows) - 1, center_idx + window)
    segment = rows[start : end + 1]
    raw_block = _rows_to_text(segment)
    return _llm_summarize(raw_block)


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def _load_all_rows() -> list:
    """
    Load every row from long_term_memories ordered chronologically (ASC by id).

    Also creates the table if it does not yet exist, making this function safe
    to call on a fresh installation before any memories have been written.

    Returns:
        List of tuples: (id, timestamp, user_query, ai_response, emotions, embedding).
    """
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()

    # Auto-create schema on first run so callers never have to check existence.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            user_query  TEXT  NOT NULL,
            ai_response TEXT  NOT NULL,
            emotions    TEXT  NOT NULL,
            embedding   BLOB  NOT NULL
        )
    """)

    cursor.execute(
        "SELECT id, timestamp, user_query, ai_response, emotions, embedding "
        "FROM long_term_memories ORDER BY id ASC"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Public recall helpers
# ---------------------------------------------------------------------------

def recall_by_embedding(query: str, date_filter: str = None) -> str:
    """
    Find the most semantically similar memory to *query* and return a summary
    of the surrounding conversation window.

    Algorithm:
        1. Embed *query* using get_raw_embedding().
        2. Load all rows (optionally filtered by *date_filter*).
        3. Compute cosine similarity between the query vector and each row's
           stored embedding vector.
        4. Select the top-1 match (must exceed a 0.3 similarity threshold).
        5. Summarise a ±2-row window around the best match.

    Args:
        query:       Natural language description of the memory to find.
        date_filter: Optional "YYYY-MM-DD" string.  When provided, only rows
                     from that date are considered as candidates.

    Returns:
        A plain-text summary of the most relevant memory window, or a
        "not found" / error message.
    """
    # Lazy import to avoid circular dependency at module load time:
    # tools.py imports memory_utils, so we cannot import tools at the top level.
    from tools import get_raw_embedding

    try:
        query_vector = get_raw_embedding(query)
    except Exception as e:
        return f"Embedding extraction failed: {e}"

    rows = _load_all_rows()
    if not rows:
        return "No memories stored yet."

    # Apply optional date filter before building the similarity matrix.
    if date_filter:
        candidate_rows = [r for r in rows if r[1].startswith(date_filter)]
        if not candidate_rows:
            return f"No memories found for date: {date_filter}"
    else:
        candidate_rows = rows

    # ---- Cosine similarity (vectorised with NumPy) -------------------------
    q_vec = np.array(query_vector, dtype=np.float64)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm   # Normalise query vector in-place.

    # Deserialise each stored embedding blob (written as array("d") bytes).
    db_vecs = []
    for row in candidate_rows:
        try:
            db_vecs.append(np.frombuffer(row[5], dtype=np.float64))
        except Exception:
            # If a blob is corrupt, substitute a zero vector so it scores 0.
            db_vecs.append(np.zeros_like(q_vec))

    db_matrix = np.stack(db_vecs)                             # shape: (N, dim)
    norms     = np.linalg.norm(db_matrix, axis=1, keepdims=True)
    norms     = np.where(norms == 0, 1, norms)               # Avoid divide-by-zero.
    scores    = (db_matrix / norms) @ q_vec                   # Cosine similarities.

    best_idx = int(np.argmax(scores))

    # Reject matches below the minimum similarity threshold.
    if scores[best_idx] < 0.3:
        return f"No memory closely matching '{query}' was found."

    print(
        f"[memory_utils] top match: "
        f"id={candidate_rows[best_idx][0]}, score={scores[best_idx]:.4f}"
    )

    # Find the centre index within the *full* rows list (not just the filtered
    # candidates) so that the context window can include adjacent entries from
    # other dates if relevant.
    best_id     = candidate_rows[best_idx][0]
    full_center = next(
        (i for i, r in enumerate(rows) if r[0] == best_id),
        best_idx,
    )

    return summarize_window(rows, full_center, window=2)


def recall_by_date(date_str: str) -> str:
    """
    Retrieve all conversation memories recorded on a specific date and return
    an LLM-generated summary of the entire day's conversations.

    Uses a SQL LIKE query on the timestamp column (format: "YYYY-MM-DD HH:MM:SS")
    rather than an embedding search, so no API call is needed to retrieve rows.

    Args:
        date_str: Target date in "YYYY-MM-DD" format.

    Returns:
        A labelled summary string, or a "not found" message.
    """
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()

    # LIKE with a trailing % matches any timestamp that starts with the date.
    cursor.execute(
        "SELECT id, timestamp, user_query, ai_response, emotions "
        "FROM long_term_memories WHERE timestamp LIKE ? ORDER BY timestamp ASC",
        (f"{date_str}%",),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"No memories recorded on {date_str}."

    raw_block = _rows_to_text(rows)
    summary   = _llm_summarize(raw_block)
    return f"[Memory summary for {date_str}]\n{summary}"


def get_session_opening_context() -> str:
    """
    Retrieve and summarise the last 5 conversation pairs from memory.db.

    This is called by server.py at the start of each new WebSocket session
    and injected into the system prompt so the agent has continuity across
    sessions without needing to re-read the entire memory database.

    Returns:
        A plain-text summary of recent conversations, or an empty string if
        no memories exist yet (e.g. first launch).
    """
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()

    try:
        # Guard against a fresh installation where the table does not yet exist.
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

        # Grab the 5 most recent rows (DESC), then reverse for chronological order.
        cursor.execute(
            "SELECT id, timestamp, user_query, ai_response, emotions "
            "FROM long_term_memories ORDER BY id DESC LIMIT 5"
        )
        rows = cursor.fetchall()

    except Exception as e:
        print(f"[memory_utils] session context load error: {e}")
        return ""
    finally:
        conn.close()

    if not rows:
        return ""   # No previous sessions exist; nothing to inject.

    # Restore chronological (oldest-first) order before summarising.
    rows      = list(reversed(rows))
    raw_block = _rows_to_text(rows)
    return _llm_summarize(raw_block)
