"""
The agent: observes recent user data, applies rule-based logic to decide
whether to adjust the calorie target. Gemini is used ONLY for the
human-readable explanation - and only when the decision is nuanced enough
to need it. Simple "maintain" decisions use a hardcoded template instead,
which cuts a meaningful share of API calls since "maintain" is the most
common outcome week to week.
"""

from google import genai
from django.conf import settings
from django.utils import timezone
from datetime import timedelta

from logging_app.models import WeightEntry, DailyCheckIn
from agent.models import AgentDecision
from ml_layer.gpt_module.usage_tracker import make_cache_key, get_cached_or_none, set_cache, record_usage

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _estimate_tdee(user):
    weight = user.current_weight_kg or 70
    height = user.height_cm or 170
    age = user.age or 25

    if user.gender == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5

    activity_multipliers = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725,
    }
    multiplier = activity_multipliers.get(user.activity_level, 1.2)
    return bmr * multiplier


def _hardcoded_maintain_explanation(user, previous_target):
    return (
        f"- Your progress over the past two weeks looks on track for your {user.goal} goal.\n"
        f"- We are keeping your calorie target at {previous_target} kcal/day.\n"
        "- Keep logging consistently so we can catch any needed adjustments early."
    )


def run_weekly_agent(user):
    two_weeks_ago = timezone.now() - timedelta(days=14)
    one_week_ago = timezone.now() - timedelta(days=7)

    recent_weights = list(WeightEntry.objects.filter(user=user, recorded_at__gte=two_weeks_ago).order_by("recorded_at"))
    recent_checkins = list(DailyCheckIn.objects.filter(user=user, date__gte=one_week_ago))

    tdee = _estimate_tdee(user)
    previous_target = user.current_calorie_target or round(tdee)

    weight_trend = None
    if len(recent_weights) >= 2:
        weight_trend = recent_weights[-1].weight_kg - recent_weights[0].weight_kg

    avg_adherence = None
    if recent_checkins:
        scores = [c.adherence_score for c in recent_checkins if c.adherence_score is not None]
        avg_adherence = sum(scores) / len(scores) if scores else None

    decision_type = "maintain_calories"
    new_target = previous_target

    if weight_trend is not None:
        if user.goal == "lose" and weight_trend > -0.2:
            decision_type = "decrease_calories"
            new_target = round(previous_target * 0.9)
        elif user.goal == "gain" and weight_trend < 0.2:
            decision_type = "increase_calories"
            new_target = round(previous_target * 1.1)
        elif user.goal == "lose" and weight_trend < -1.5:
            decision_type = "increase_calories"
            new_target = round(previous_target * 1.05)

    if avg_adherence is not None and avg_adherence <= 2:
        decision_type = "flag_for_review"

    observed_data = {
        "weight_trend_kg": weight_trend,
        "avg_adherence": avg_adherence,
        "estimated_tdee": round(tdee),
        "num_weight_entries": len(recent_weights),
        "num_checkins": len(recent_checkins),
    }

    # Hardcoded shortcut: "maintain" decisions don't need LLM nuance - use a template.
    if decision_type == "maintain_calories":
        reasoning_text = _hardcoded_maintain_explanation(user, previous_target)
        record_usage("agent_explanation", response=None, was_hardcoded=True)
    else:
        # Cache lookup: similar decisions (same type + rounded targets) reuse an explanation
        cache_key = make_cache_key(
            "agent_explanation", decision_type, previous_target, new_target, user.goal,
            round(weight_trend, 1) if weight_trend is not None else "none",
        )
        cached = get_cached_or_none(cache_key)

        if cached is not None:
            reasoning_text = cached
            record_usage("agent_explanation", response=None, was_cached=True)
        else:
            client = _get_client()
            explanation_prompt = f"""You are a weight-management coach assistant. Explain the following
automated decision to the user as exactly 3 short bullet points, each starting with "- " on its own line.
Do not invent data beyond what is given. No heading, no intro sentence - just the 3 bullets.

Decision: {decision_type}
Previous calorie target: {previous_target}
New calorie target: {new_target}
Observed data: {observed_data}
User goal: {user.goal}

Bullet points:"""

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=explanation_prompt,
                )
                reasoning_text = response.text
                record_usage("agent_explanation", response=response)
                set_cache(cache_key, reasoning_text)
            except Exception as e:
                reasoning_text = f"(Explanation unavailable: {str(e)}) Decision was {decision_type} based on recent trends."

    decision = AgentDecision.objects.create(
        user=user,
        decision_type=decision_type,
        previous_calorie_target=previous_target,
        new_calorie_target=new_target,
        observed_data=observed_data,
        reasoning_text=reasoning_text,
    )

    if decision_type != "flag_for_review":
        user.current_calorie_target = new_target
        user.save()

    return decision

