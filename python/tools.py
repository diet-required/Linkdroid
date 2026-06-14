"""
tools.py
========
LangChain @tool definitions that Linkdroid's agent graph can invoke.

Covers:
  - Local file system management (read / write / list)
  - Web search via DuckDuckGo
  - System hardware status scanning
  - Windows system control (app launch/kill, volume, media keys, sleep)
  - Clipboard history management
  - SQLite-backed schedule database
  - Desktop screenshot capture
  - Long-term memory recall (date-based or semantic embedding search)

Environment variables consumed here:
  GOOGLE_API_KEY    – Google Generative AI API key (required for embeddings)
  EMBEDDING_MODEL   – Gemini embedding model name
                      (default: "gemini-embedding-exp-03-07")
  DESKTOP_PATH      – Override for the Desktop directory path
                      (default: ~/Desktop)
"""

import os
import re
import io
import base64
import sqlite3
import subprocess
import array
import ctypes
import psutil
from pathlib import Path

from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from google import genai
from PIL import ImageGrab
import numpy as np

# ---------------------------------------------------------------------------
# Module-level singletons / shared state
# ---------------------------------------------------------------------------

# DuckDuckGo search wrapper – stateless, safe to share across tool calls.
search_engine = DuckDuckGoSearchAPIWrapper()

# Clipboard history populated by background.py; tools read/clear this list.
clipboard_history: list[str] = []

# Path to the shared SQLite memory database.
MEMORY_DB_PATH = "memory.db"

# Desktop base directory; can be overridden via the DESKTOP_PATH env var
# so the agent works on non-standard Windows or Linux setups.
_DESKTOP_PATH = Path(os.getenv("DESKTOP_PATH", str(Path.home() / "Desktop")))


# ---------------------------------------------------------------------------
# Embedding helper (not a LangChain tool – called directly from memory code)
# ---------------------------------------------------------------------------

def get_raw_embedding(text: str) -> list[float]:
    """
    Generate a dense vector embedding for *text* using the Gemini embedding API.

    The model is configured via the EMBEDDING_MODEL environment variable so
    it can be swapped without touching source code.

    Args:
        text: The string to embed. Can be a single word or several sentences.

    Returns:
        A list of floats representing the embedding vector.

    Raises:
        RuntimeError: If the API response cannot be parsed into a vector.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    # Read the embedding model name from the environment; fall back to the
    # experimental model that was in use when this module was first written.
    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-exp-03-07")

    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=embedding_model,
        contents=text,
    )

    # The Gemini SDK wraps the embedding in a list of Embedding objects.
    # Each object exposes its values either via .values or as a direct iterable.
    if hasattr(response, "embeddings") and response.embeddings:
        emb = response.embeddings[0]
        return list(emb.values) if hasattr(emb, "values") else list(emb)

    raise RuntimeError(
        f"Embedding response parse failed. Response structure: {response}"
    )


# ---------------------------------------------------------------------------
# Tool: Local file system management
# ---------------------------------------------------------------------------

@tool
def manage_local_files(command: str, path: str, content: str = "") -> str:
    """
    Read, write, and list files on the local file system.

    Supported commands:
        "list"  – Return a directory listing for *path*.
        "read"  – Return file contents (text) or a base64 multimodal pack
                  (images / audio / video) ready for the vision node.
        "write" – Write *content* to *path* (creates parent dirs as needed).

    Path resolution rules:
        - Absolute paths (containing a drive letter ':') are used as-is.
        - Paths mentioning "Desktop" are resolved literally.
        - All other relative paths are resolved under _DESKTOP_PATH.

    Multimodal return format (images/audio/video):
        "__MULTIMODAL_PACK__|<mime>|<base64>|<filename>"
        agent_graph.py's unpack_multimodal_tool_messages() strips this tag and
        injects the binary content directly into the vision LLM message.

    Args:
        command: One of "list", "read", "write".
        path:    Target path (absolute or relative to Desktop).
        content: Text content to write (only used with "write").

    Returns:
        A string describing the result, or an error message.
    """
    print(f"[tool:manage_local_files] command={command}, path={path}")

    # --- Path resolution ----------------------------------------------------
    # Accept absolute Windows-style paths, explicit Desktop mentions, or plain
    # relative paths that we anchor to the configured Desktop directory.
    if "Desktop" in path or ":" in path:
        target_path = Path(path.replace("\\", "/"))
    else:
        target_path = _DESKTOP_PATH / path.lstrip("/")

    try:
        # ---- LIST ----------------------------------------------------------
        if command == "list":
            if not target_path.exists():
                return f"Path not found: {target_path}"
            entries = "\n".join(os.listdir(target_path))
            return f"[{target_path}] contents:\n{entries}"

        # ---- READ ----------------------------------------------------------
        elif command == "read":
            if not target_path.exists():
                return f"File not found: {target_path}"

            suffix = target_path.suffix.lower()

            # Plain-text formats – return as a UTF-8 string.
            text_exts = {".txt", ".py", ".json", ".gd", ".md", ".html", ".css"}
            if suffix in text_exts:
                with open(target_path, "r", encoding="utf-8") as f:
                    return f"[{target_path.name}]\n{f.read()}"

            # Binary media formats – encode as base64 and tag for the vision node.
            media_exts = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".mp3": "audio/mp3",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
            }
            if suffix in media_exts:
                with open(target_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                mime_type = media_exts[suffix]
                print(f"[tool:manage_local_files] media read: {target_path.name}")
                # Return a specially tagged string; agent_graph.py unpacks it.
                return f"__MULTIMODAL_PACK__|{mime_type}|{base64_data}|{target_path.name}"

            return f"Unsupported file extension: {suffix}"

        # ---- WRITE ---------------------------------------------------------
        elif command == "write":
            # Ensure parent directories exist before writing.
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {target_path} ({len(content)} chars)"

    except Exception as e:
        return f"File system error: {e}"


# ---------------------------------------------------------------------------
# Tool: Web search
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """
    Perform a web search via DuckDuckGo and return the top results as text.

    Uses the LangChain DuckDuckGoSearchAPIWrapper which returns a string
    containing the most relevant snippets.

    Args:
        query: Natural language or keyword search query.

    Returns:
        A string with search result snippets, or an error message.
    """
    print(f"[tool:web_search] query={query}")
    try:
        return search_engine.run(query)
    except Exception as e:
        return f"Search error: {e}"


# ---------------------------------------------------------------------------
# Tool: System hardware status
# ---------------------------------------------------------------------------

@tool
def scan_system_status() -> str:
    """
    Scan and report the current CPU and RAM utilisation via psutil.

    Samples CPU usage over a short 0.5-second window to avoid returning
    a misleadingly low instantaneous reading.

    Returns:
        A multi-line string with CPU %, RAM %, and available RAM in GB.
    """
    print("[tool:scan_system_status] scanning hardware")
    try:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        ram_info = psutil.virtual_memory()
        return (
            f"CPU usage: {cpu_usage}%\n"
            f"RAM usage: {ram_info.percent}%\n"
            f"Available RAM: {round(ram_info.available / (1024 ** 3), 2)} GB"
        )
    except Exception as e:
        return f"System scan error: {e}"


# ---------------------------------------------------------------------------
# Tool: Windows system control
# ---------------------------------------------------------------------------

@tool
def control_windows_system(action_type: str, target_value: str = "", count: int = 1) -> str:
    """
    Control Windows system-level actions: launch / kill apps, adjust volume,
    control media playback, or put the PC to sleep.

    Supported action_type values:
        "launch_app"      – Launch an application by name (looks up exeindexer.db).
        "kill_app"        – Force-terminate a process by name (taskkill /f).
        "sleep"           – Enter system sleep mode.
        "volume_up"       – Raise volume by (count * 2)%.
        "volume_down"     – Lower volume by (count * 2)%.
        "mute"            – Toggle mute.
        "media_play_pause"– Toggle play / pause on the active media player.
        "media_next"      – Skip to the next media track.
        "media_prev"      – Return to the previous media track.

    App launch uses a local SQLite database (exeindexer.db) that maps program
    names to full executable paths.  Exact match is tried first; a LIKE query
    is used as a fallback to handle partial name matches.

    Args:
        action_type:  One of the action strings listed above.
        target_value: Application name (required for launch_app / kill_app).
        count:        Repetition count for volume_up / volume_down.

    Returns:
        A human-readable result string, or an error message.
    """
    print(f"[tool:control_windows_system] action={action_type}, target={target_value}")

    try:
        # ---- Application launch --------------------------------------------
        if action_type == "launch_app":
            if not target_value:
                return "No program name provided."

            # Strip extension and normalise to lowercase for consistent matching.
            program_keyword = Path(target_value.lower()).stem

            conn = sqlite3.connect("exeindexer.db")
            cursor = conn.cursor()

            # First attempt: exact match on program_name or file_name.
            cursor.execute(
                "SELECT file_name, full_path FROM exe_registry "
                "WHERE program_name = ? OR file_name = ? LIMIT 1",
                (program_keyword, program_keyword + ".exe"),
            )
            result = cursor.fetchone()

            # Fallback: substring match, preferring shorter (more specific) names.
            if not result:
                cursor.execute(
                    "SELECT file_name, full_path FROM exe_registry "
                    "WHERE program_name LIKE ? OR file_name LIKE ? "
                    "ORDER BY LENGTH(file_name) ASC LIMIT 1",
                    (f"%{program_keyword}%", f"%{program_keyword}%"),
                )
                result = cursor.fetchone()

            conn.close()

            if not result:
                return f"No executable found matching '{program_keyword}'."

            # Launch the process detached so it survives the Python process.
            subprocess.Popen(
                f'"{result[1]}"',
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            return f"Launched: {result[0]}"

        # ---- Application kill ----------------------------------------------
        elif action_type == "kill_app":
            if not target_value:
                return "No program name provided."

            program_keyword = Path(target_value.lower()).stem
            # Use a wildcard so partial process names (e.g. "note" → "notepad.exe") work.
            exit_code = os.system(f'taskkill /f /im "{program_keyword}*"')
            if exit_code == 0:
                return f"Process terminated: {program_keyword}"
            return "Process not found or insufficient permissions."

        # ---- Virtual key helper (inner function) ---------------------------
        def press_key(vk_code: int) -> None:
            """Send a single virtual key press + release via the Win32 API."""
            ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)   # key down
            ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)   # key up

        # ---- Sleep ---------------------------------------------------------
        if action_type == "sleep":
            # SetSuspendState(Hibernate=0, ForceCritical=1, DisableWakeEvent=0)
            ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)
            return "System entering sleep mode."

        # ---- Volume controls -----------------------------------------------
        elif action_type == "volume_up":
            for _ in range(count):
                press_key(0xAF)   # VK_VOLUME_UP
            return f"Volume increased by {count * 2}%."

        elif action_type == "volume_down":
            for _ in range(count):
                press_key(0xAE)   # VK_VOLUME_DOWN
            return f"Volume decreased by {count * 2}%."

        elif action_type == "mute":
            press_key(0xAD)       # VK_VOLUME_MUTE
            return "Mute toggled."

        # ---- Media controls ------------------------------------------------
        elif action_type == "media_play_pause":
            press_key(0xB3)       # VK_MEDIA_PLAY_PAUSE
            return "Play/pause toggled."

        elif action_type == "media_next":
            press_key(0xB0)       # VK_MEDIA_NEXT_TRACK
            return "Skipped to next track."

        elif action_type == "media_prev":
            press_key(0xB1)       # VK_MEDIA_PREV_TRACK
            return "Returned to previous track."

        return f"Unknown action: {action_type}"

    except Exception as e:
        return f"System control error: {e}"


# ---------------------------------------------------------------------------
# Tool: Clipboard history management
# ---------------------------------------------------------------------------

@tool
def manage_clipboard_guardian(command: str) -> str:
    """
    View or clear the in-memory clipboard history maintained by background.py.

    The clipboard guardian (background.py) polls the system clipboard every
    second and appends new entries to the shared `clipboard_history` list.
    This tool exposes that list to the agent.

    Supported commands:
        "show"  – Display all captured clipboard entries (last 30).
        "clear" – Erase the entire clipboard history.

    Args:
        command: "show" or "clear".

    Returns:
        A formatted string with clipboard contents, a confirmation message,
        or an error string.
    """
    global clipboard_history
    print(f"[tool:manage_clipboard_guardian] command={command}")

    try:
        if command == "show":
            if not clipboard_history:
                return "Clipboard history is empty."

            # Show a truncated preview per entry, with the full text in brackets.
            lines = [
                f"[{idx}] {text[:40]}{'...' if len(text) > 40 else ''} (full: {text})"
                for idx, text in enumerate(clipboard_history, 1)
            ]
            return "Clipboard history:\n" + "\n".join(lines)

        elif command == "clear":
            clipboard_history.clear()
            return "Clipboard history cleared."

        return f"Unknown command: {command}"

    except Exception as e:
        return f"Clipboard error: {e}"


# ---------------------------------------------------------------------------
# Tool: Schedule database
# ---------------------------------------------------------------------------

@tool
def manage_schedule_db(
    command: str,
    date_str: str,
    title: str = "",
    category: str = "general",
) -> str:
    """
    Create, read, and delete schedule entries stored in a local SQLite DB
    (schedule.db in the working directory).

    Supported commands:
        "add"    – Insert a new schedule entry for the given date.
        "show"   – List all entries for the given date.
        "delete" – Remove an entry by its numeric id or title keyword.

    The schedules table schema (auto-created if absent):
        id        INTEGER PRIMARY KEY AUTOINCREMENT
        date      TEXT     (YYYY-MM-DD)
        title     TEXT
        category  TEXT     (default: "general")

    Args:
        command:  "add", "show", or "delete".
        date_str: Target date in YYYY-MM-DD format.
        title:    Event title (required for "add"; id or keyword for "delete").
        category: Category label (used only with "add").

    Returns:
        A human-readable result or error string.
    """
    db_path = Path("schedule.db")
    print(f"[tool:manage_schedule_db] command={command}, date={date_str}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Ensure the table exists on first use.
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS schedules "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
            " date TEXT NOT NULL, title TEXT, category TEXT)"
        )
        conn.commit()

        # ---- ADD -----------------------------------------------------------
        if command == "add":
            if not title:
                conn.close()
                return "No title provided for the schedule entry."

            cursor.execute(
                "INSERT INTO schedules (date, title, category) VALUES (?, ?, ?)",
                (date_str, title, category),
            )
            conn.commit()
            conn.close()
            return f"Schedule added: [{category}] '{title}' on {date_str}"

        # ---- SHOW ----------------------------------------------------------
        elif command == "show":
            cursor.execute(
                "SELECT id, title, category FROM schedules WHERE date = ?",
                (date_str,),
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return f"No schedules found for {date_str}."

            lines = "\n".join(f"- id:{r[0]} [{r[2]}] {r[1]}" for r in rows)
            return f"[{date_str} schedule]\n{lines}"

        # ---- DELETE --------------------------------------------------------
        elif command == "delete":
            if not title:
                conn.close()
                return "Specify the schedule id or title to delete."

            if title.isdigit():
                # Numeric value – treat as the row id.
                cursor.execute(
                    "DELETE FROM schedules WHERE id = ? AND date = ?",
                    (int(title), date_str),
                )
            else:
                # String value – perform a LIKE search on the title column.
                cursor.execute(
                    "DELETE FROM schedules WHERE title LIKE ? AND date = ?",
                    (f"%{title}%", date_str),
                )

            conn.commit()
            conn.close()
            return f"Schedule deleted: '{title}' on {date_str}"

        conn.close()
        return f"Unknown command: {command}"

    except Exception as e:
        return f"Schedule DB error: {e}"


# ---------------------------------------------------------------------------
# Tool: Desktop screenshot capture
# ---------------------------------------------------------------------------

@tool
def capture_and_send_desktop() -> str:
    """
    Capture the current desktop as a screenshot, scale it to 50% of the
    original resolution (to reduce token cost), and return it as a base64-
    encoded PNG wrapped in the multimodal pack format.

    The returned string is parsed by agent_graph.unpack_multimodal_tool_messages
    and injected as an inline image into the vision LLM message.

    Returns:
        "__MULTIMODAL_PACK__|image/png|<base64>|screenshot.png" on success,
        or an error string on failure.
    """
    print("[tool:capture_and_send_desktop] capturing screen")
    try:
        screenshot = ImageGrab.grab()

        # Downscale to half resolution to reduce base64 size and LLM input tokens.
        new_size = (screenshot.width // 2, screenshot.height // 2)
        resized = screenshot.resize(new_size)

        # Encode to PNG bytes, then to base64 string.
        img_byte_arr = io.BytesIO()
        resized.save(img_byte_arr, format="PNG")
        base64_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

        return f"__MULTIMODAL_PACK__|image/png|{base64_data}|screenshot.png"

    except Exception as e:
        return f"Screenshot error: {e}"


# ---------------------------------------------------------------------------
# Tool: Long-term memory recall
# ---------------------------------------------------------------------------

@tool
def recall_past_memory(query: str) -> str:
    """
    Retrieve past conversation memories from the long-term memory database.

    Dispatch strategy:
        - If *query* contains a date in YYYY-MM-DD format → date-based SQL
          search via memory_utils.recall_by_date().
        - Otherwise → semantic embedding similarity search via
          memory_utils.recall_by_embedding().

    Both paths return a short LLM-generated summary of the most relevant
    memory window rather than raw database rows, keeping the context length
    manageable.

    Args:
        query: A date string (e.g. "2026-05-20") or a natural language phrase
               describing what you want to remember (e.g. "when I was upset
               about work").

    Returns:
        A human-readable memory summary, or a "not found" message.
    """
    print(f"[tool:recall_past_memory] query='{query}'")

    # Import here to avoid circular dependency at module load time:
    # memory_utils imports get_raw_embedding from this file.
    from memory_utils import recall_by_date, recall_by_embedding

    # Check whether the query contains an ISO date string.
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)

    if date_match:
        target_date = date_match.group(1)
        print(f"[tool:recall_past_memory] date mode: {target_date}")
        return recall_by_date(target_date)
    else:
        print("[tool:recall_past_memory] embedding mode")
        return recall_by_embedding(query)


# ---------------------------------------------------------------------------
# Exported tool list (consumed by agent_graph.py)
# ---------------------------------------------------------------------------

# All tools that the LangGraph agent is allowed to invoke.
# Order does not affect routing; it only determines how the tools are described
# to the LLM via bind_tools().
get_all_tools = [
    web_search,
    manage_local_files,
    scan_system_status,
    control_windows_system,
    manage_clipboard_guardian,
    manage_schedule_db,
    capture_and_send_desktop,
    recall_past_memory,
]
