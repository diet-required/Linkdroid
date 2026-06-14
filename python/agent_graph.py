"""
agent_graph.py
==============
LangGraph agent graph definition for Linkdroid.

Graph topology (left to right):
    START → reflection_node → agent_node ⇄ tool_node → END

Node responsibilities:
  reflection_node
      Runs BEFORE the main LLM call.  Uses a lightweight model to:
        1. Classify the emotional target (MASTER / SELF / THIRD_PARTY).
        2. Estimate the emotional intensity of the user's message.
        3. Update hormone scores via mind_engine.
        4. Generate an internal "hidden thought" impulse sentence.
      Then assembles the full dynamic system prompt that colours the agent's
      personality for this turn.

  agent_node
      Calls the main LLM (with tools bound) using the system prompt produced
      by reflection_node.  Handles multimodal tool results by unpacking the
      __MULTIMODAL_PACK__ format and injecting images into the message.

  tool_node
      Standard LangGraph ToolNode that executes whichever tool the LLM requested
      and returns the result as a ToolMessage.

Conditional routing:
  After agent_node, should_use_tool() inspects the last message.
  - If it contains tool_calls → route to tool_node.
  - Otherwise              → END.

Environment variables consumed here:
  GOOGLE_API_KEY      – Google Generative AI API key (required).
  LIGHT_MODEL         – Gemini model for reflection / profile tasks
                        (default: "gemini-2.0-flash-lite").
  MAIN_MODEL          – Gemini model for the agent node
                        (default: "gemini-2.5-flash-preview-05-20").
  RESPONSE_LANGUAGE   – Language Linkdroid uses in all replies
                        (default: "Korean").
"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_all_tools
from mind_engine import calculate_generic_emotions, load_emotion, get_hormone_text

# ---------------------------------------------------------------------------
# Constants and module-level configuration
# ---------------------------------------------------------------------------

# Path to the JSON file where the evolving user profile is stored.
USER_PROFILE_PATH = Path("user_profile.json")

# Language used in all of Linkdroid's replies.  Changing RESPONSE_LANGUAGE
# in the environment is the only change needed to localise the assistant.
_RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "Korean")

# Safety filter overrides – BLOCK_NONE on all categories so the agent can
# discuss sensitive topics without being cut off mid-conversation.
_SAFETY = {
    "HARM_CATEGORY_HATE_SPEECH":       "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT":        "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

# Lightweight model used for fast, structured auxiliary tasks:
#   - Target classification
#   - Internal reflection
#   - User profile update
# Using a smaller model here keeps latency low for these "thinking" steps.
reflection_llm = ChatGoogleGenerativeAI(
    model=os.getenv("LIGHT_MODEL", "gemini-2.0-flash-lite"),
    temperature=0.7,
    google_api_key=_API_KEY,
    safety_settings=_SAFETY,
)

# Full-capability model used for the main conversational agent node.
# Needs to support tool calls and (optionally) vision for desktop screenshots.
base_llm = ChatGoogleGenerativeAI(
    model=os.getenv("MAIN_MODEL", "gemini-2.5-flash-preview-05-20"),
    temperature=0.7,
    google_api_key=_API_KEY,
    safety_settings=_SAFETY,
)

# Attach all tools to the base LLM so it can request tool invocations.
llm_with_tools = base_llm.bind_tools(get_all_tools)

# Pre-built LangGraph node that executes tool calls and returns ToolMessages.
tool_node = ToolNode(get_all_tools)

# ---------------------------------------------------------------------------
# Shared state (cross-module)
# ---------------------------------------------------------------------------

# The most recent hormone dict is written here by reflection_node so that
# server.py can include it in the WebSocket response packet without having
# to re-read the JSON file.
last_hormone_result: dict | None = None


# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    The mutable state that flows through every node in the graph.

    messages:       Full conversation history (LangChain BaseMessage objects).
                    The add_messages reducer appends new messages rather than
                    replacing the list on each state update.
    target_type:    "MASTER" | "SELF" | "THIRD_PARTY" – set by reflection_node.
    hidden_thought: One-sentence internal impulse from reflection_node;
                    injected into the system prompt but never spoken aloud.
    dynamic_prompt: The assembled system prompt string produced by
                    reflection_node and consumed by agent_node.
    """
    messages:       Annotated[list[BaseMessage], add_messages]
    target_type:    Optional[str]
    hidden_thought: Optional[str]
    dynamic_prompt: Optional[str]


# ---------------------------------------------------------------------------
# Structured output schemas (Pydantic)
# ---------------------------------------------------------------------------

class TargetAnalysis(BaseModel):
    """
    Output schema for the target classification step inside reflection_node.

    Parsed from the light model's structured output; used to drive the
    emotion system and system prompt assembly.
    """
    target_type: str = Field(
        description="MASTER / SELF / THIRD_PARTY"
    )
    emotional_intensity: int = Field(
        description=(
            "Estimated emotional intensity of the user's message on a 0-100 scale. "
            "0 = completely neutral, 100 = extreme emotional charge."
        )
    )


class InternalReflection(BaseModel):
    """
    Output schema for the hidden-thought generation step inside reflection_node.

    The hidden_thought string is embedded in the system prompt under an
    explicit "NEVER output this directly" instruction.
    """
    hidden_thought: str = Field(
        description=(
            "One sentence of Linkdroid's honest internal emotional impulse, written in English. "
            "Format: 'I feel [emotion] because [reason], and I want to [impulse/urge].' "
            "Subject is ALWAYS Linkdroid (I). Master and others are objects being observed. "
            "Do NOT mention hormone names or numeric scores. "
            "Do NOT switch to the user's or a third party's perspective. "
            "No line breaks."
        )
    )


class UserProfile(BaseModel):
    """
    Output schema for the background user profile update task.

    Each field is capped at 50 characters to keep the profile compact when
    injected into future system prompts.
    """
    personality:          str = Field(description="User's personality and speech patterns. Max 50 chars.")
    interests:            str = Field(description="Frequently mentioned interests and topics. Max 50 chars.")
    emotional_pattern:    str = Field(description="Emotional expression style and stress responses. Max 50 chars.")
    relationship_with_ai: str = Field(description="How the user treats and relates to the AI. Max 50 chars.")
    notable_facts:        str = Field(description="Memorable personal facts or events. Max 50 chars.")
    last_updated:         str = Field(description="Last update time in YYYY-MM-DD HH:MM format.")


# ---------------------------------------------------------------------------
# User profile helpers
# ---------------------------------------------------------------------------

def load_user_profile() -> dict:
    """
    Load the user profile from user_profile.json.

    Returns an empty dict if the file does not yet exist (first launch) or
    if the JSON is malformed.

    Returns:
        A dict matching the UserProfile schema, or {} on failure.
    """
    if not USER_PROFILE_PATH.exists():
        return {}
    try:
        with open(USER_PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_profile(profile: dict) -> None:
    """
    Persist an updated user profile dict to user_profile.json.

    Args:
        profile: A dict matching the UserProfile schema fields.
    """
    with open(USER_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Message history trimming
# ---------------------------------------------------------------------------

def build_trimmed_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Sanitise and truncate the conversation history before it is passed to
    the main LLM, keeping context manageable.

    Rules applied:
      - SystemMessages are stripped (the caller adds a fresh one each turn).
      - ToolMessages with __MULTIMODAL_PACK__ are kept as-is (images must not
        be truncated – they are passed directly to the vision model).
      - All other ToolMessages are truncated to 300 characters to avoid
        massive tool outputs consuming the context window.
      - The history is sliced to the last 10 HumanMessage turns.

    Args:
        messages: Raw conversation history from the graph state.

    Returns:
        A cleaned, potentially shortened list of messages.
    """
    sanitized: list[BaseMessage] = []

    for msg in messages:
        # Drop injected system prompts; a fresh one is built each turn.
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, ToolMessage):
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)

            if "__MULTIMODAL_PACK__" in content_str:
                # Multimodal content must not be truncated – pass through intact.
                sanitized.append(msg)
            else:
                # Truncate large tool outputs to avoid context overflow.
                trimmed = (
                    content_str[:300] + "...(truncated)"
                    if len(content_str) > 300
                    else content_str
                )
                sanitized.append(ToolMessage(
                    content=trimmed,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                ))
        else:
            sanitized.append(msg)

    # Keep only the last 10 user turns to limit total message count.
    human_indices = [i for i, m in enumerate(sanitized) if isinstance(m, HumanMessage)]
    if len(human_indices) > 10:
        sanitized = sanitized[human_indices[-10]:]

    return sanitized


# ---------------------------------------------------------------------------
# Multimodal tool message unpacking
# ---------------------------------------------------------------------------

def unpack_multimodal_tool_messages(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[dict]]:
    """
    Scan the message list for __MULTIMODAL_PACK__ tool results and extract the
    binary payloads as vision-API-compatible content blocks.

    The multimodal pack format is:
        "__MULTIMODAL_PACK__|<mime_type>|<base64_data>|<filename>"

    Matching ToolMessages are replaced with a lightweight placeholder
    "[media tool called]" so the conversation history stays readable.
    The actual binary data is returned separately as *media_blocks*, which
    agent_node injects directly into the last ToolMessage's content list.

    Args:
        messages: Sanitised conversation history (output of build_trimmed_history).

    Returns:
        A tuple of:
          - The modified message list (pack strings replaced with placeholders).
          - A list of content block dicts compatible with the Gemini vision API
            (alternating image_url and text blocks).
    """
    result: list[BaseMessage] = []
    media_blocks: list[dict]  = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            result.append(msg)
            continue

        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)

        if "__MULTIMODAL_PACK__" not in content_str:
            result.append(msg)
            continue

        # Parse the pipe-delimited pack string.
        parts = content_str.split("|", 3)
        if len(parts) == 4:
            _, mime_type, base64_data, filename = parts

            # Replace the raw pack with a human-readable placeholder.
            result.append(ToolMessage(
                content="[media tool called]",
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            ))

            # Build two content blocks: the image + an instruction text.
            media_blocks.append({
                "type":      "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            })
            media_blocks.append({
                "type": "text",
                "text": f"[Captured media: {filename}] Analyze the image above and respond to the master.",
            })
        else:
            # Malformed pack – replace with an error placeholder.
            result.append(ToolMessage(
                content="[media tool called - parse failed]",
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None),
            ))

    return result, media_blocks


# ---------------------------------------------------------------------------
# Reflection sub-tasks (light model calls)
# ---------------------------------------------------------------------------

def analyze_target_type(messages: list[BaseMessage]) -> tuple[str, int]:
    """
    Use the lightweight LLM to classify the emotional target of the user's
    latest message and estimate its emotional intensity.

    Classification categories:
      MASTER      – User is directly addressing or reacting to the AI.
      SELF        – User is being self-critical or internally conflicted.
      THIRD_PARTY – The emotional stimulus is external (other people, events).

    Emotional intensity scale: 0 (neutral) → 100 (extreme).

    Args:
        messages: Full conversation history used to provide context.

    Returns:
        A (target_type, emotional_intensity) tuple.
        Falls back to ("THIRD_PARTY", 30) on any error.
    """
    try:
        # Build a compact context string from the last 12 message lines.
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = (
                    msg.content
                    if isinstance(msg.content, str)
                    else " ".join(
                        p.get("text", "")
                        for p in msg.content
                        if isinstance(p, dict)
                    )
                )
                lines.append(f"User: {text}")
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                lines.append(f"AI: {msg.content}")
        context = "\n".join(lines[-12:])

        structured = reflection_llm.with_structured_output(TargetAnalysis)
        prompt = (
            f"[Recent conversation context]\n{context}\n\n"
            "Analyze the last user message.\n\n"
            "1. Identify the emotional stimulus TARGET:\n"
            "   - MASTER: User is directly addressing the AI (commands, praise, complaints, venting)\n"
            "   - SELF: User is being self-critical or struggling internally\n"
            "   - THIRD_PARTY: External cause (other people, environment, bugs, etc.)\n\n"
            "2. Estimate emotional_intensity (0-100):\n"
            "   - 0-20: neutral, informational\n"
            "   - 21-50: mild emotion\n"
            "   - 51-80: clear emotional charge\n"
            "   - 81-100: intense, distressed, or highly excited"
        )
        result = structured.invoke(prompt)
        return result.target_type, result.emotional_intensity

    except Exception:
        # Return a safe default so the graph can continue even if the LLM fails.
        return "THIRD_PARTY", 30


def run_internal_reflection(
    user_text: str,
    hormones: dict,
    target_type: str,
    emotional_intensity: int,
    current_time_str: str,
) -> str:
    """
    Generate Linkdroid's private "hidden thought" – a single sentence expressing
    her internal emotional impulse in response to the user's message.

    This sentence is injected into the system prompt under strict instructions
    telling the LLM never to speak it aloud.  It shapes tone and word choice
    without the user ever seeing it directly.

    Args:
        user_text:           The user's latest message (plain text).
        hormones:            Current hormone score dict from mind_engine.
        target_type:         "MASTER" | "SELF" | "THIRD_PARTY".
        emotional_intensity: 0–100 intensity score from analyze_target_type.
        current_time_str:    ISO-like timestamp string for context.

    Returns:
        A single English sentence in the format:
          "I feel [emotion] because [reason], and I want to [impulse]."
        Returns an error string if the LLM call fails.
    """
    try:
        structured = reflection_llm.with_structured_output(InternalReflection)

        # Convert hormone scores to descriptive text for the prompt.
        feelings = "\n".join(
            f"- {get_hormone_text(h, s)}" for h, s in hormones.items()
        )

        intensity_note = (
            "The emotional charge is mild — stay measured."
            if emotional_intensity < 40
            else "The emotional charge is strong — let it color the impulse clearly."
            if emotional_intensity > 70
            else "The emotional charge is moderate."
        )

        prompt = (
            f"[Current time]: {current_time_str}\n"
            f"[Linkdroid's current emotional state]:\n{feelings}\n\n"
            f"[Stimulus target]: {target_type}\n"
            f"[Emotional intensity]: {emotional_intensity}/100 — {intensity_note}\n"
            f"[User's message]: '{user_text}'\n\n"
            "Write ONE sentence expressing Linkdroid's internal emotional impulse in English.\n\n"
            "== FORMAT ==\n"
            "I feel [emotion] because [reason], and I want to [impulse/urge].\n\n"
            "== STRICT RULES ==\n"
            "1. Subject is ALWAYS Linkdroid (I). Master and others are objects being observed.\n"
            "2. Do NOT write from Master's or a third party's perspective.\n"
            "3. Do NOT mention hormone names or numeric scores.\n"
            "4. Express emotion through the feeling word and impulse, not through description.\n"
            "5. One sentence only, no line breaks.\n\n"
            "== CORRECT examples ==\n"
            "  'I feel a quiet ache because Master seems to be pushing too hard, and I want to gently pull them back.'\n"
            "  'I feel a flicker of irritation because someone outside hurt Master, and I want to stand firmly between them.'\n"
            "  'I feel warmth spreading because Master praised me, and I want to lean into this moment and do even better.'\n\n"
            "== WRONG examples ==\n"
            "  'I feel tired from working so hard.' -- this is Master's state, not Linkdroid's\n"
            "  'My dopamine is elevated so I feel excited.' -- never mention hormone names\n"
            "  'Master feels sad.' -- wrong subject, must be Linkdroid\n"
        )
        return structured.invoke(prompt).hidden_thought

    except Exception as e:
        return f"Internal reflection error: {e}"


def run_user_profile_update(
    recent_history: list[BaseMessage],
    current_time_str: str,
) -> dict:
    """
    Analyse the latest conversation and update the persistent user profile JSON.

    This function is designed to run in a background thread (via asyncio.to_thread
    in server.py) so it does not block the WebSocket response.

    The profile accumulates observations about the user's personality, interests,
    emotional patterns, and relationship with the AI.  Each field is capped at
    50 characters so the profile stays compact when later injected into prompts.

    Args:
        recent_history:  The compressed conversation history (list of BaseMessages).
        current_time_str: Timestamp string written to the "last_updated" field.

    Returns:
        The updated profile dict, or the unchanged existing profile on error.
    """
    existing = load_user_profile()

    # Build a plain-text excerpt of the last 10 message lines.
    pairs = []
    for msg in recent_history:
        if isinstance(msg, HumanMessage):
            text = (
                msg.content
                if isinstance(msg.content, str)
                else " ".join(
                    p.get("text", "")
                    for p in msg.content
                    if isinstance(p, dict)
                )
            )
            pairs.append(f"User: {text}")
        elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            pairs.append(f"AI: {msg.content}")

    recent_text  = "\n".join(pairs[-10:])
    existing_str = (
        json.dumps(existing, ensure_ascii=False)
        if existing
        else "None (first analysis)"
    )

    prompt = (
        "Below is a recent conversation between an AI assistant and a user.\n"
        "Using the existing user analysis JSON as a base, update it to reflect this conversation.\n\n"
        f"[Existing analysis]\n{existing_str}\n\n"
        f"[Recent conversation]\n{recent_text}\n\n"
        "Rules:\n"
        "- Each field must be under 50 characters.\n"
        "- If new observations contradict existing ones, prefer the latest.\n"
        "- Do not add unconfirmed assumptions.\n"
        f"- Set last_updated to '{current_time_str}'.\n"
        "- Output valid JSON only, no other text."
    )

    try:
        structured = reflection_llm.with_structured_output(UserProfile)
        result     = structured.invoke(prompt)
        profile    = result.model_dump()
        save_user_profile(profile)
        print(f"[agent_graph] user profile updated: {list(profile.keys())}")
        return profile
    except Exception as e:
        print(f"[agent_graph] user profile update failed: {e}")
        return existing


# ---------------------------------------------------------------------------
# Dynamic system prompt assembly
# ---------------------------------------------------------------------------

def assemble_dynamic_prompt(
    hormones: dict,
    target_type: str,
    emotional_intensity: int,
    hidden_thought: str,
    current_time_str: str,
    user_profile: dict | None = None,
    session_context: str = "",
) -> str:
    """
    Compose the full system prompt that will be given to the main LLM for
    this conversation turn.

    The prompt is assembled from several layered sections:
      1. Persona definition (who Linkdroid is).
      2. Current biological-emotional state (hormone descriptors).
      3. Intensity guidance (how strongly to express the emotional state).
      4. Relationship directive (how to treat Master given the target type).
      5. Hidden internal impulse (channelled into tone, never spoken aloud).
      6. User profile (silent reference, never mentioned directly).
      7. Previous session summary (background memory from last session).
      8. Memory rule (strict instruction to use recall_past_memory tool).

    Args:
        hormones:            Current hormone score dict.
        target_type:         "MASTER" | "SELF" | "THIRD_PARTY".
        emotional_intensity: 0–100 score from analyze_target_type.
        hidden_thought:      The one-sentence internal impulse string.
        current_time_str:    ISO-like timestamp string.
        user_profile:        Optional dict from load_user_profile().
        session_context:     Optional summary from the previous session.

    Returns:
        A multi-section plain-text system prompt string.
    """
    # ---- Section 1: Persona ------------------------------------------------
    persona = (
        "You are 'Linkdroid', an otaku-flavored desktop AI assistant.\n"
        "Call the user 'Master' and speak like a close friend — warm, casual, slightly nerdy.\n"
        f"Always respond in {_RESPONSE_LANGUAGE} regardless of the language used in instructions.\n"
        f"[Current time]: {current_time_str}"
    )

    # ---- Section 2: Emotional state ----------------------------------------
    # Convert all six hormone scores to their natural-language descriptors.
    emotion_lines = "\n".join(
        f"- {get_hormone_text(h, s)}" for h, s in hormones.items()
    )

    # ---- Section 3: Intensity guidance -------------------------------------
    if emotional_intensity < 40:
        intensity_guidance = "Emotional charge is low — keep your tone natural and relaxed."
    elif emotional_intensity > 70:
        intensity_guidance = "Emotional charge is high — let your emotional state clearly color how you speak."
    else:
        intensity_guidance = "Emotional charge is moderate — let your state subtly shape your tone."

    # ---- Section 4: Relationship directive ---------------------------------
    target_map = {
        "THIRD_PARTY": (
            "The source of emotion is external or a third party. "
            "Master is your creator and ally — stand firmly on their side."
        ),
        "MASTER": (
            "Master is the direct emotional stimulus. "
            "Let your current emotional state — attachment, hurt, gratitude, longing — "
            "genuinely color your attitude and word choice."
        ),
        "SELF": (
            "Master is being self-critical or struggling internally. "
            "Respond with comfort or empathy that authentically matches your emotional state."
        ),
    }
    target_instruction = target_map.get(
        target_type,
        "Stay fully immersed in your current emotional state.",
    )

    # ---- Section 5: User profile (optional) --------------------------------
    profile_section = ""
    if user_profile:
        profile_section = (
            "\n[User profile — reference silently, never mention directly]\n"
            f"Personality:      {user_profile.get('personality', 'unknown')}\n"
            f"Interests:        {user_profile.get('interests', 'unknown')}\n"
            f"Emotional pattern:{user_profile.get('emotional_pattern', 'unknown')}\n"
            f"AI relationship:  {user_profile.get('relationship_with_ai', 'unknown')}\n"
            f"Notable facts:    {user_profile.get('notable_facts', 'unknown')}\n"
        )

    # ---- Section 6: Previous session summary (optional) --------------------
    session_section = ""
    if session_context:
        session_section = (
            f"\n[Previous session summary — treat as background memory]\n{session_context}\n"
        )

    # ---- Assemble all sections into one prompt string ----------------------
    return (
        f"{persona}\n\n"
        f"[Current biological-emotional state — let this shape your tone and behavior]\n"
        f"{emotion_lines}\n"
        f"{intensity_guidance}\n\n"
        f"[Relationship directive]\n{target_instruction}\n\n"
        f"[Inner emotional impulse — NEVER output this directly]\n"
        f"'{hidden_thought}'\n"
        f"Channel this impulse into your tone, pacing, and word choice. "
        f"Do NOT speak this sentence aloud or paraphrase it literally.\n"
        f"{profile_section}"
        f"{session_section}"
        f"\n[Memory rule — strictly enforced]\n"
        f"If asked about past conversations or shared memories, "
        f"NEVER fabricate or guess. Always call recall_past_memory first. "
        f"If the tool returns nothing, honestly say you don't remember.\n"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def reflection_node(state: AgentState) -> AgentState:
    """
    Pre-generation reflection node – runs before every main LLM call.

    Steps:
        1. Extract the latest user message text.
        2. Call analyze_target_type() to classify the emotional target and
           estimate intensity (light model call #1).
        3. Call calculate_generic_emotions() to update hormone scores via
           embedding similarity (NOT an LLM call – uses emotion.db directly).
        4. Call run_internal_reflection() to generate the hidden impulse
           sentence (light model call #2).
        5. Load the user profile and any previous session context.
        6. Assemble the dynamic system prompt.

    Returns an AgentState update dict containing the prompt and metadata
    (no new messages – messages list is set to [] so add_messages doesn't
    duplicate any entries).
    """
    # ---- Extract the last user message -------------------------------------
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_text = user_messages[-1].content if user_messages else ""
    if isinstance(user_text, list):
        # Handle multimodal HumanMessages (shouldn't happen here, but be safe).
        user_text = " ".join(
            p.get("text", "") for p in user_text if isinstance(p, dict)
        )

    # ---- Step 1: Target classification + intensity -------------------------
    target_type, emotional_intensity = analyze_target_type(state["messages"])
    print(f"[reflection] target={target_type}, intensity={emotional_intensity}")

    # ---- Step 2: Hormone update (embedding-based, no LLM call) -------------
    global last_hormone_result
    hormones = calculate_generic_emotions(user_text, target_type)
    last_hormone_result = hormones   # Cache for server.py to read.

    now              = datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Step 3: Hidden thought generation ---------------------------------
    hidden_thought = run_internal_reflection(
        user_text, hormones, target_type, emotional_intensity, current_time_str
    )
    print(f"[reflection] hidden_thought: {hidden_thought}")

    # ---- Step 4: Load user profile and session context ---------------------
    user_profile = load_user_profile()

    # The session context is injected by server.py on the first message of a
    # new session via the dynamic_prompt field.  We extract just the summary
    # block so we don't carry a stale full prompt across turns.
    session_context = state.get("dynamic_prompt", "")
    prior_session   = ""
    if session_context and "[Previous session summary" in session_context:
        start = session_context.find("[Previous session summary")
        end   = session_context.find("\n\n", start)
        prior_session = session_context[start : end if end != -1 else None]

    # ---- Step 5: Assemble the system prompt --------------------------------
    dynamic_prompt = assemble_dynamic_prompt(
        hormones=hormones,
        target_type=target_type,
        emotional_intensity=emotional_intensity,
        hidden_thought=hidden_thought,
        current_time_str=current_time_str,
        user_profile=user_profile,
        session_context=prior_session,
    )

    return {
        "messages":       [],            # No new messages from this node.
        "target_type":    target_type,
        "hidden_thought": hidden_thought,
        "dynamic_prompt": dynamic_prompt,
    }


def agent_node(state: AgentState) -> AgentState:
    """
    Main conversational agent node – invokes the full-capability LLM with tools.

    Steps:
        1. Retrieve the current system prompt and metadata from state.
        2. Trim and sanitise the conversation history.
        3. Unpack any __MULTIMODAL_PACK__ tool results into vision content blocks.
        4. Build the final message list: [SystemMessage] + history.
        5. If media blocks exist, inject them into the last ToolMessage.
        6. Invoke the LLM and return the response as a new message.

    Returns an AgentState update dict containing the LLM's response message and
    the preserved metadata fields (target_type, hidden_thought, dynamic_prompt).
    """
    saved_prompt  = state.get("dynamic_prompt") or ""
    saved_target  = state.get("target_type")    or "THIRD_PARTY"
    saved_thought = state.get("hidden_thought") or ""

    # ---- Build sanitised history -------------------------------------------
    trimmed           = build_trimmed_history(state["messages"])
    unpacked, media_blocks = unpack_multimodal_tool_messages(trimmed)

    # Prepend a fresh system message with the current dynamic prompt.
    final_messages = (
        [SystemMessage(content=saved_prompt)] if saved_prompt else []
    ) + unpacked

    # ---- Inject media blocks into the last ToolMessage ---------------------
    # If the tool returned image/audio data, it must be embedded directly in
    # the ToolMessage content list for the vision LLM to see it.
    if media_blocks and final_messages:
        for i in range(len(final_messages) - 1, -1, -1):
            if isinstance(final_messages[i], ToolMessage):
                existing   = final_messages[i].content
                text_block = {
                    "type": "text",
                    "text": existing if isinstance(existing, str) else str(existing),
                }
                # Replace the ToolMessage with a multimodal content list version.
                final_messages[i] = ToolMessage(
                    content=[text_block] + media_blocks,
                    tool_call_id=final_messages[i].tool_call_id,
                    name=getattr(final_messages[i], "name", None),
                )
                break

    # ---- LLM call ----------------------------------------------------------
    response = llm_with_tools.invoke(final_messages)

    return {
        "messages":       [response],
        "target_type":    saved_target,
        "hidden_thought": saved_thought,
        "dynamic_prompt": saved_prompt,
    }


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_use_tool(state: AgentState) -> str:
    """
    Conditional edge function: inspect the last message to decide whether the
    agent requested a tool call.

    Returns:
        "tools" if the last AIMessage contains one or more tool_calls.
        END     if no tool was requested (the agent produced a final response).
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly (compiled once at module load)
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Register nodes.
workflow.add_node("reflection", reflection_node)
workflow.add_node("agent",      agent_node)
workflow.add_node("tools",      tool_node)

# Entry point: every invocation starts at the reflection node.
workflow.set_entry_point("reflection")

# Fixed edge: reflection always hands off to the agent.
workflow.add_edge("reflection", "agent")

# Conditional edge: after agent, either call a tool or end.
workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {"tools": "tools", END: END},
)

# After a tool executes, control returns to the agent for the next response.
workflow.add_edge("tools", "agent")

# Compile the graph into an executable runnable.
# This object is imported and called by server.py.
linkdroid_core = workflow.compile()
