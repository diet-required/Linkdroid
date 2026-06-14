"""
background.py
=============
Background task runner for Linkdroid.

Currently contains a single long-running background task:

  Clipboard Guardian
  ------------------
  Polls the system clipboard every second using pyperclip.
  Whenever new text is detected, it is appended to the shared
  `clipboard_history` list defined in tools.py.

  The agent can then inspect or clear clipboard history through the
  manage_clipboard_guardian tool.

  History is capped at 30 entries (oldest entries are evicted) to
  prevent unbounded memory growth during long sessions.

Usage:
    Import and call start_background_tasks() once at startup.
    All tasks are launched as daemon threads, so they are automatically
    terminated when the main process exits.
"""

import threading
import time

import pyperclip

from tools import clipboard_history


# ---------------------------------------------------------------------------
# Clipboard guardian
# ---------------------------------------------------------------------------

def watch_clipboard_loop() -> None:
    """
    Infinite loop that monitors the system clipboard for new text content.

    Behaviour:
      - Checks the clipboard every 1.0 second via pyperclip.paste().
      - Appends to `clipboard_history` only when the content has changed
        since the last check (deduplication by string equality).
      - Silently ignores any exceptions (e.g. clipboard access errors on
        headless environments) and continues polling.
      - Evicts the oldest entry when the history exceeds 30 items so memory
        usage stays bounded.

    This function is intended to run inside a daemon thread and never returns.
    """
    print("[background] clipboard guardian started.")
    last_text = ""   # Track the last seen clipboard content to detect changes.

    while True:
        try:
            current_text = pyperclip.paste()

            # Only record genuinely new content to avoid duplicate entries
            # when the clipboard hasn't changed between polls.
            if current_text and current_text != last_text:
                clipboard_history.append(current_text)
                last_text = current_text
                print(f"[background] clipboard captured: '{current_text[:30]}...'")

                # Cap history at 30 entries; remove the oldest (index 0) when
                # the limit is exceeded.  A deque would be more efficient here
                # but a plain list keeps the shared type simple for tools.py.
                if len(clipboard_history) > 30:
                    clipboard_history.pop(0)

        except Exception:
            # Swallow all exceptions so a transient clipboard error
            # (e.g. another app holding the clipboard lock on Windows)
            # doesn't crash the background thread.
            pass

        # Poll interval: 1 second is fast enough to feel real-time while
        # keeping CPU usage negligible.
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_background_tasks() -> None:
    """
    Spawn all background daemon threads.

    Called once from main.py before the WebSocket server starts.
    Each thread is a daemon so it will be killed automatically when the
    main Python process exits – no explicit cleanup is needed.
    """
    threading.Thread(
        target=watch_clipboard_loop,
        daemon=True,
        name="ClipboardGuardian",
    ).start()
