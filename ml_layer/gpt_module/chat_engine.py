"""
Conversational coach powered by Gemini 3.5 Flash, grounded with a small RAG
knowledge base and short-term conversation memory. Uses Gemini's structured
JSON output mode so responses are reliably parsed as short bullet points
plus follow-up question suggestions, rather than free-form paragraphs.
"""

from google import genai
from google.genai import types
from django.conf import settings
from ml_layer.gpt_module.knowledge_base import get_relevant_context
from ml_layer.gpt_module.usage_tracker import record_usage
import json

_client = None

# Hardcoded shortcuts: context-independent facts, safe to skip the API for.
# Each maps to a pre-written bullet answer plus relevant follow-up suggestions.
HARDCODED_RESPONSES = {
    "what is bmi": {
        "answer_points": [
            "BMI = weight (kg) ÷ height (m)²",
            "It's a rough screening tool, not a diagnosis",
            "Doesn't distinguish muscle from fat — can mislead very muscular or very lean people",
        ],
        "follow_up_questions": [
            "What is my BMI?",
            "How can I improve my BMI?",
            "Is BMI a reliable measure?",
        ],
    },
    "how much water should i drink": {
        "answer_points": [
            "Common guideline: 2-3 liters (8-12 glasses) per day",
            "Adjust for activity level and climate",
            "Thirst and urine color are good everyday indicators",
        ],
        "follow_up_questions": [
            "Does water intake affect weight loss?",
            "How many calories should I eat?",
            "What is BMI?",
        ],
    },
}

MAX_HISTORY_TURNS = 6

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2 to 4 short, plain-language bullet points answering the user's message. Each under 20 words.",
        },
        "follow_up_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 short, natural follow-up questions the user might realistically ask next, based on this answer.",
        },
    },
    "required": ["answer_points", "follow_up_questions"],
}


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _check_hardcoded(user_message):
    normalized = user_message.strip().lower().rstrip("?")
    return HARDCODED_RESPONSES.get(normalized)


def _format_history(conversation_history):
    if not conversation_history:
        return "(no earlier messages in this conversation)"
    recent = conversation_history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        speaker = "User" if turn["role"] == "user" else "Coach"
        lines.append(f"{speaker}: {turn['message']}")
    return "\n".join(lines)


def get_coach_response(user, user_message, conversation_history=None):
    """
    Returns a dict: {"answer_points": [...], "follow_up_questions": [...]}
    """
    hardcoded = _check_hardcoded(user_message)
    if hardcoded:
        record_usage("chat", response=None, was_hardcoded=True)
        return hardcoded

    client = _get_client()

    relevant_facts = get_relevant_context(user_message, top_k=3)
    facts_text = "\n".join(f"- {fact}" for fact in relevant_facts)

    profile_text = (
        f"Goal: {user.goal}, Activity level: {user.activity_level}, "
        f"Height: {user.height_cm or 'unknown'} cm, "
        f"Current weight: {user.current_weight_kg or 'unknown'} kg, "
        f"Target weight: {user.target_weight_kg or 'unknown'} kg."
    )

    history_text = _format_history(conversation_history)

    prompt = f"""You are a supportive, evidence-based weight management coach.
Use the reference facts below to ground your answer - do not invent numbers
or claims beyond them. You may perform simple, correct arithmetic directly
from the user profile (e.g. BMI from height and weight) without asking for
data already provided. Use the conversation history to maintain context.
Keep answers SHORT and scannable - bullet points, not paragraphs.

User profile:
{profile_text}

Reference facts:
{facts_text}

Conversation so far:
{history_text}

User message:
{user_message}"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        record_usage("chat", response=response)
        parsed = json.loads(response.text)
        return {
            "answer_points": parsed.get("answer_points", ["Sorry, I couldn't format that properly."]),
            "follow_up_questions": parsed.get("follow_up_questions", []),
        }
    except Exception as e:
        return {
            "answer_points": [f"Sorry, I could not process that right now. ({str(e)})"],
            "follow_up_questions": [],
        }
