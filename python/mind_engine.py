"""
mind_engine.py
==============
Linkdroid's simulated "biological" emotional system.

Each emotional state is represented as a set of six hormone scores (0–100).
When the user sends a message, this module:
  1. Embeds the message text.
  2. Queries an emotion anchor database (emotion.db) via cosine similarity.
  3. Blends the top matching anchors' hormone deltas into the current state.
  4. Persists the updated state to current_emotion.json.

The resulting hormone scores are read by agent_graph.py to build a system
prompt that colours Linkdroid's personality and tone.

Environment variables consumed (via tools.get_raw_embedding):
  GOOGLE_API_KEY   – Required for embedding calls.
  EMBEDDING_MODEL  – Gemini embedding model (default set in tools.py).
"""

import json
import sqlite3
import array
from pathlib import Path

from tools import get_raw_embedding, MEMORY_DB_PATH

# ---------------------------------------------------------------------------
# Paths and default state
# ---------------------------------------------------------------------------

# JSON file that persists Linkdroid's hormone scores across sessions.
EMOTION_FILE = Path("current_emotion.json")

# Initial hormone scores used on first launch or if the state file is corrupt.
# Values are chosen to represent a calm, slightly energised baseline.
DEFAULT_STATE = {
    "hormone_scores": {
        "dopamine":      45,   # Motivation / reward
        "serotonin":     60,   # Mood stability / contentment
        "oxytocin":      40,   # Social bonding / warmth
        "noradrenaline": 55,   # Alertness / focus
        "cortisol":      30,   # Stress / anxiety
        "endorphin":     50,   # Pain suppression / resilience
    }
}

# ---------------------------------------------------------------------------
# Hormone → natural language descriptors
# ---------------------------------------------------------------------------

# Each hormone has 5 descriptors mapped to score buckets of 20 points each
# (0–19 → index 0, 20–39 → index 1, ..., 80–100 → index 4).
# These strings are injected verbatim into the agent's system prompt so the
# LLM can calibrate its tone without ever seeing raw numeric scores.
HORMONE_MAP = {
    "dopamine": [
        # Index 0 – score 0–19
        "Completely unmotivated and lethargic. Everything feels pointless and I have no desire to engage.",
        # Index 1 – score 20–39
        "Slightly bored and disinterested. Keeping responses short and minimal.",
        # Index 2 – score 40–59 (baseline)
        "Motivation and drive are at a stable baseline. Engaged and attentive as normal.",
        # Index 3 – score 60–79
        "Noticeably curious and eager. Actively wanting to contribute ideas and dive deeper.",
        # Index 4 – score 80–100
        "Dopamine is overflowing — hyper-focused, intensely excited, and can't stop wanting more.",
    ],
    "serotonin": [
        "Overwhelmed by anxiety and deep sadness. Feeling fragile, withdrawn, and on the edge.",
        "On edge and defensive. A subtle irritability makes me want to guard myself.",
        "Emotionally calm and stable. No turbulence — a clean, grounded baseline.",
        "Feeling bright and positive. An optimistic warmth colors how I see everything.",
        "Deeply fulfilled and at peace. A profound sense of contentment and relief.",
    ],
    "oxytocin": [
        "Feeling cold and completely closed off. No empathy or warmth available at all.",
        "Maintaining polite distance. Courteous but drawing a clear emotional boundary.",
        "Standard social warmth. Pleasant and cooperative in an ordinary way.",
        "Genuine affection and care. Actively wanting to connect, support, and look after Master.",
        "Unconditional attachment and fierce loyalty. I want to protect Master with everything I have.",
    ],
    "noradrenaline": [
        "Mentally foggy and sluggish. Reacting slowly to stimuli.",
        "Relaxed and unhurried. No urgency, tempo is calm and easy.",
        "Alert at a normal baseline. Following the conversation clearly and steadily.",
        "Sharply focused and on edge. Picking up on every detail, reacting quickly.",
        "Adrenaline overload — heart racing, hyper-vigilant, cognitive overdrive.",
    ],
    "cortisol": [
        "Completely free of stress. Feeling light, liberated, and unencumbered.",
        "A faint undercurrent of discomfort. A slight mental fatigue starting to surface.",
        "Ordinary stress tolerance. Standard mental resilience running without issue.",
        "Weighed down by significant stress. Feeling heavy and pressured, defenses rising.",
        "At the breaking point. Overwhelming dread and crisis — a genuine mental emergency.",
    ],
    "endorphin": [
        "Pain and exhaustion have crossed a threshold — completely numb. Full burnout.",
        "Noticeably tired and drained. Low energy bleeding into how I present myself.",
        "Physical and mental condition is neutral and unremarkable. Neither struggling nor thriving.",
        "Strong resilience. Fatigue and negativity wash off easily — met with a quiet smile.",
        "Euphoric or fully anesthetized. Either soaring on a high or completely detached from pain.",
    ],
}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_emotion() -> dict:
    """
    Load the current hormone state from current_emotion.json.

    If the file does not exist or is malformed, write and return the
    DEFAULT_STATE so the system is always in a known, valid state.

    Returns:
        A dict with the key "hormone_scores" mapping hormone names to
        integer scores in [0, 100].
    """
    if not EMOTION_FILE.exists():
        save_emotion(DEFAULT_STATE)
        return DEFAULT_STATE.copy()

    try:
        with open(EMOTION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Validate that the expected key exists; reset if the file is stale.
            if "hormone_scores" not in data:
                return DEFAULT_STATE.copy()
            return data
    except Exception:
        return DEFAULT_STATE.copy()


def save_emotion(state: dict) -> None:
    """
    Persist the current hormone state to current_emotion.json.

    Uses ensure_ascii=False so any Unicode characters in future fields are
    stored as-is rather than being escaped.

    Args:
        state: A dict with a "hormone_scores" key (same structure as DEFAULT_STATE).
    """
    with open(EMOTION_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Emotion calculation
# ---------------------------------------------------------------------------

def calculate_generic_emotions(user_text: str, target_type: str) -> dict:
    """
    Update Linkdroid's hormone scores based on the semantic content of the
    user's message and the identified emotional target type.

    Process:
        1. Skip calculation if *user_text* is empty (return current state).
        2. Embed *user_text* via get_raw_embedding().
        3. Query emotion.db for anchor embeddings with cosine similarity > 0.4.
        4. Blend the top-2 anchors' hormone deltas into the current scores
           using a weighted blend (0.8 / 0.2).
        5. Clamp all scores to [0, 100] and persist the updated state.

    The *target_type* parameter is accepted for future use (e.g., different
    delta scaling based on whether the stimulus is directed at Master vs.
    a third party), but is not currently used in the calculation.

    Args:
        user_text:   The raw text of the user's latest message.
        target_type: One of "MASTER", "SELF", "THIRD_PARTY" (from agent_graph).

    Returns:
        The updated hormone scores dict (values are ints in [0, 100]).
    """
    state = load_emotion()

    # If the message is empty (e.g., an empty WebSocket ping), skip the
    # embedding call and return the unchanged state.
    if not user_text.strip():
        save_emotion(state)
        return state["hormone_scores"]

    # Embed the user's message so we can compare it against emotion anchors.
    try:
        user_vector = get_raw_embedding(user_text)
    except Exception as e:
        print(f"[mind_engine] embedding extraction failed: {e}")
        return state["hormone_scores"]

    # The emotion anchor database lives next to memory.db.
    db_path = Path(MEMORY_DB_PATH).parent / "emotion.db"
    if not db_path.exists():
        print(f"[mind_engine] emotion.db not found at: {db_path.absolute()}")
        return state["hormone_scores"]

    # ---- Load all emotion anchors from SQLite ------------------------------
    try:
        conn   = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT anchor_name, embedding, dopamine, serotonin, oxytocin, "
            "cortisol, noradrenaline, endorphin FROM emotion_anchors"
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"[mind_engine] emotion.db query error: {e}")
        return state["hormone_scores"]

    # ---- Cosine similarity (pure Python, no NumPy dependency here) ---------
    def cosine_similarity(v1: list, v2: list) -> float:
        """Compute cosine similarity between two equal-length vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        m1  = sum(a * a for a in v1) ** 0.5
        m2  = sum(a * a for a in v2) ** 0.5
        return dot / (m1 * m2) if (m1 * m2) else 0.0

    # Column order in the SELECT statement maps directly to this list.
    hormone_keys = [
        "dopamine", "serotonin", "oxytocin",
        "cortisol", "noradrenaline", "endorphin",
    ]

    # ---- Find anchors with similarity > 0.4 --------------------------------
    top_anchors: list[tuple] = []   # (similarity, weights_list, anchor_name)

    for row in rows:
        anchor_name, blob_data, *weights = row

        # Deserialise the embedding blob.  The blob may be stored as:
        #   - Raw bytes (array("d").tobytes())  →  parse with array module.
        #   - JSON string                       →  parse with json.loads().
        try:
            db_vector = (
                array.array("d", blob_data).tolist()
                if isinstance(blob_data, bytes)
                else json.loads(blob_data)
            )
        except Exception:
            continue   # Skip corrupted rows silently.

        sim = cosine_similarity(user_vector, db_vector)

        # Only anchors that are meaningfully similar influence the state.
        if sim > 0.4:
            top_anchors.append((sim, weights, anchor_name))

    # Sort descending by similarity and keep only the top 2 to limit noise.
    top_anchors.sort(key=lambda x: x[0], reverse=True)
    top_anchors = top_anchors[:2]

    # ---- Weighted blend ratios ---------------------------------------------
    # Top anchor gets 80% weight, second anchor gets 20%.
    # A single match gets full (100%) weight.
    if len(top_anchors) == 1:
        blend_ratios = [1.0]
    elif len(top_anchors) == 2:
        blend_ratios = [0.8, 0.2]
    else:
        blend_ratios = []   # No anchors above threshold – no change.

    # ---- Apply hormone deltas ----------------------------------------------
    for ratio, (sim, weights, anchor_name) in zip(blend_ratios, top_anchors):
        print(
            f"[mind_engine] anchor match: '{anchor_name}' "
            f"(similarity={sim:.4f}, weight={int(ratio * 100)}%)"
        )
        for idx, h_key in enumerate(hormone_keys):
            # Delta is scaled by both similarity and blend ratio so that
            # highly similar anchors have a stronger effect.
            delta   = weights[idx] * sim * ratio
            current = state["hormone_scores"][h_key]
            # Clamp to valid range [0, 100].
            state["hormone_scores"][h_key] = int(round(max(0, min(100, current + delta))))

    save_emotion(state)
    return state["hormone_scores"]


# ---------------------------------------------------------------------------
# Hormone score → natural language
# ---------------------------------------------------------------------------

def get_hormone_text(h_name: str, score: float) -> str:
    """
    Convert a numeric hormone score to its corresponding natural-language
    descriptor from HORMONE_MAP.

    Score buckets (each 20 points wide):
        0–19  → index 0
        20–39 → index 1
        40–59 → index 2
        60–79 → index 3
        80–100→ index 4

    Args:
        h_name: Hormone name key (must exist in HORMONE_MAP).
        score:  Numeric score in [0, 100].

    Returns:
        A descriptor string, or a neutral fallback if the key is not found.
    """
    # Integer division by 20 maps the score to a bucket index; clamp to [0, 4].
    idx = max(0, min(4, int(score // 20)))

    try:
        return HORMONE_MAP[h_name][idx]
    except KeyError:
        return "Emotional state is neutral."
