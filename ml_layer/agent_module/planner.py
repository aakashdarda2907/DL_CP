from google import genai
from google.genai import types
from django.conf import settings
from django.utils import timezone
from datetime import timedelta

from logging_app.models import WeightEntry, DailyCheckIn
from agent.models import AgentDecision
from ml_layer.gpt_module.usage_tracker import record_usage

# Import your tools
from .tools import adjust_macros, flag_for_review, suggest_recipe

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client

def run_weekly_agent(user):
    # Anchor the date window to the user's latest data
    latest_checkin = DailyCheckIn.objects.filter(user=user).order_by('-date').first()
    anchor_date = latest_checkin.date if latest_checkin else timezone.now().date()

    two_weeks_ago_date = anchor_date - timedelta(days=14)
    one_week_ago_date = anchor_date - timedelta(days=7)

    recent_checkins = list(DailyCheckIn.objects.filter(user=user, date__gte=one_week_ago_date))
    recent_weights = list(WeightEntry.objects.filter(user=user, recorded_at__date__gte=two_weeks_ago_date).order_by("recorded_at"))

    previous_target = user.current_calorie_target or 2000
    
    weight_trend = None
    if len(recent_weights) >= 2:
        weight_trend = recent_weights[-1].weight_kg - recent_weights[0].weight_kg

    avg_adherence = None
    if recent_checkins:
        scores = [c.adherence_score for c in recent_checkins if c.adherence_score is not None]
        avg_adherence = sum(scores) / len(scores) if scores else None

    observed_data = {
        "goal": user.goal,
        "current_weight_kg": user.current_weight_kg,
        "weight_trend_kg": weight_trend,
        "avg_adherence": avg_adherence,
        "num_weight_entries": len(recent_weights),
        "num_checkins": len(recent_checkins),
    }

    agent_prompt = f"""You are an expert AI weight-management coach.
    Decide the best action by calling exactly one primary tool (adjust_macros OR flag_for_review).

    User Data:
    - Goal: {user.goal}
    - Weight change over last 2 weeks (kg, positive = gained): {weight_trend}
    - Avg adherence score (0-5): {avg_adherence}
    - Current calorie target: {previous_target}

    Rules:
    1. If avg adherence is not None and <= 2.0, call flag_for_review.
    2. If adherence is above 2.0 and weight change is moving AWAY from the goal
    (e.g. goal is loss but weight went up, or goal is gain but weight went down),
    call adjust_macros with 'decrease' or 'increase' (5-10%).
    3. If progress matches the goal, call adjust_macros with 'maintain' and 0.
    4. Do NOT call flag_for_review when adherence is above 2.0 unless the data is missing or absurd.
    5. Optionally also call suggest_recipe when the user is on track.
    Always fill in the 'reason' argument in plain text with no markdown.
    """

    client = _get_client()
    tools_called_log = []
    
    new_target = previous_target
    decision_type = "maintain_calories" 
    reasoning_text = ""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=agent_prompt,
            config=types.GenerateContentConfig(
                tools=[adjust_macros, flag_for_review, suggest_recipe],
                # KEY FIX: stop the SDK from auto-running tools
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY
                    )
                ),
                temperature=0.0,
            ),
        )
        record_usage("agent_tool_calling", response=response)

        calls = response.function_calls or []
        if not calls:
            decision_type = "flag_for_review"
            reasoning_text = "Model returned no tool call; flagged as a safe fallback."
            tools_called_log.append({"error": "No tools called by model."})
        else:
            flagged = False
            reasons = []
            for fc in calls:
                args = dict(fc.args) if fc.args else {}
                tools_called_log.append({"tool": fc.name, "args": args})

                if fc.name == "adjust_macros":
                    result = adjust_macros(
                        current_calories=previous_target,
                        adjustment_type=str(args.get("adjustment_type", "maintain")),
                        percentage_change=float(args.get("percentage_change", 0.0)),
                        reason=str(args.get("reason", "")),
                    )
                    new_target = result["new_calories"]
                    decision_type = result["decision_type"]
                    reasons.append(result["reason"])

                elif fc.name == "flag_for_review":
                    result = flag_for_review(reason=str(args.get("reason", "Needs review.")))
                    flagged = True
                    reasons.append(result["reason"])

                elif fc.name == "suggest_recipe":
                    result = suggest_recipe(
                        goal=str(args.get("goal", user.goal)),
                        calorie_limit=int(args.get("calorie_limit", previous_target)),
                    )
                    reasons.append(result["suggestion"])

            if flagged:
                decision_type = "flag_for_review"
                new_target = previous_target  # never change calories on a flag
            reasoning_text = " ".join(r for r in reasons if r) or "Tool calls executed."

    except Exception as e:
        decision_type = "flag_for_review"
        new_target = previous_target
        reasoning_text = f"Error: {e}"
        tools_called_log.append({"error": str(e)})

    decision = AgentDecision.objects.create(
        user=user,
        decision_type=decision_type,
        previous_calorie_target=previous_target,
        new_calorie_target=new_target,
        observed_data=observed_data,
        reasoning_text=reasoning_text,
        tools_called=tools_called_log
    )

    if decision_type != "flag_for_review":
        user.current_calorie_target = new_target
        user.save()

    return decision