from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from ml_layer.gpt_module.chat_engine import get_coach_response
from .models import ChatMessage
import json

HISTORY_LOAD_LIMIT = 30
HISTORY_CONTEXT_LIMIT = 6

STARTER_SUGGESTIONS = [
    "What is BMI?",
    "How many calories should I eat?",
    "What can I do about a weight plateau?",
]


@login_required
def chat_view(request):
    messages_qs = ChatMessage.objects.filter(user=request.user).order_by("created_at")[:HISTORY_LOAD_LIMIT]
    return render(request, "coach/chat.html", {
        "chat_history": messages_qs,
        "starter_suggestions": STARTER_SUGGESTIONS,
    })


@login_required
def chat_api(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "").strip()

        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)

        recent_messages = ChatMessage.objects.filter(user=request.user).order_by("-created_at")[:HISTORY_CONTEXT_LIMIT]
        conversation_history = [
            {"role": m.role, "message": m.message} for m in reversed(recent_messages)
        ]

        result = get_coach_response(request.user, user_message, conversation_history)
        answer_points = result["answer_points"]
        follow_up_questions = result["follow_up_questions"]

        # Store as bullet-prefixed text so history replay/context stays consistent
        stored_answer_text = "\n".join(f"- {point}" for point in answer_points)

        ChatMessage.objects.create(user=request.user, role="user", message=user_message)
        ChatMessage.objects.create(user=request.user, role="assistant", message=stored_answer_text)

        return JsonResponse({
            "answer_points": answer_points,
            "follow_up_questions": follow_up_questions,
        })

    return JsonResponse({"error": "Invalid request method"}, status=405)
