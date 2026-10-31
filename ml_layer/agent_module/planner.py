import time
from datetime import timedelta

from django.conf import settings
from django.utils import timezone
from google import genai
from google.genai import types

from logging_app.models import WeightEntry, DailyCheckIn
from agent.models import AgentDecision
from ml_layer.gpt_module.usage_tracker import record_usage

from .tools import adjust_macros, flag_for_review, suggest_recipe

_client = None

# Every tool Gemini is allowed to choose from. Used for BOTH the API call and
# the "tools considered" step of the reasoning timeline.
AVAILABLE_TOOLS = [adjust_macros, flag_for_review, suggest_recipe]
TOOL_BY_NAME = {fn.__name__: fn for fn in AVAILABLE_TOOLS}

# ---------------------------------------------------------------------------
# Coaching styles (Task 2b). Each style changes TWO things:
#   1. tone  - how the 'reason' text is written
#   2. bias  - how quickly the agent adjusts calories vs. holding steady
# ---------------------------------------------------------------------------
DEFAULT_STYLE = "motivational"

COACHING_STYLES = {
    "strict": {
        "label": "Strict",
        "tone": ('Direct, firm and accountability-focused. Short sentences, no fluff, no sugar-coating. '
                 'Example: "Weight is up 1.0 kg against your goal. Calories cut by 10%. Stick to the plan."'),
        "flag_below": 2.5,      # adherence at/below this -> flag_for_review
        "off_track_kg": 0.3,    # drift (kg) that already counts as "off goal"
        "pct_range": "8-10",    # size of change the model should choose
        "max_pct": 10.0,        # hard cap enforced in Python
        "bias": ("Be proactive: correct small drifts early instead of waiting. "
                 "Prefer adjusting calories over holding steady when the data is off-goal."),
    },
    "motivational": {
        "label": "Motivational",
        "tone": ('Upbeat, encouraging and energetic. Celebrate wins, frame setbacks as fixable. '
                 'Example: "You\'re putting in real effort! A small 6% tweak will get you back on track."'),
        "flag_below": 2.0,
        "off_track_kg": 0.5,
        "pct_range": "5-7",
        "max_pct": 7.0,
        "bias": "Balance encouragement with action: adjust when the drift is clear, otherwise hold steady.",
    },
    "gentle": {
        "label": "Gentle",
        "tone": ('Warm, patient and reassuring. Never blame or pressure. '
                 'Example: "You\'re doing okay, and small steps count. Let\'s keep your plan the same and build consistency."'),
        "flag_below": 1.5,
        "off_track_kg": 1.0,
        "pct_range": "3-5",
        "max_pct": 5.0,
        "bias": ("Lean toward maintaining. Encourage consistency before changing any numbers: "
                 "if average adherence is below 4.0, do NOT increase or decrease calories - "
                 "call adjust_macros with 'maintain' and 0 and encourage the user to stay consistent first."),
    },
}


def get_style(user):
    """Return (key, profile) for the user. Falls back safely if the value is missing or invalid."""
    key = getattr(user, "coaching_style", None)
    if key not in COACHING_STYLES:
        key = DEFAULT_STYLE
    return key, COACHING_STYLES[key]


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _step(trace, kind, title, detail="", data=None):
    """Append one step to the reasoning timeline (stored as JSON on the decision)."""
    trace.append({"kind": kind, "title": title, "detail": detail, "data": data or {}})


def _first_doc_line(fn):
    lines = (fn.__doc__ or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def _fmt(value, suffix=""):
    return "no data" if value is None else f"{value:.1f}{suffix}"


def run_weekly_agent(user):
    # ---------- 1. Gather observations (unchanged from Task 1) ----------
    latest_checkin = DailyCheckIn.objects.filter(user=user).order_by('-date').first()
    anchor_date = latest_checkin.date if latest_checkin else timezone.now().date()

    two_weeks_ago_date = anchor_date - timedelta(days=14)
    one_week_ago_date = anchor_date - timedelta(days=7)

    recent_checkins = list(DailyCheckIn.objects.filter(user=user, date__gte=one_week_ago_date))
    recent_weights = list(
        WeightEntry.objects.filter(user=user, recorded_at__date__gte=two_weeks_ago_date).order_by("recorded_at")
    )

    previous_target = user.current_calorie_target or 2000

    weight_trend = None
    if len(recent_weights) >= 2:
        weight_trend = recent_weights[-1].weight_kg - recent_weights[0].weight_kg

    avg_adherence = None
    if recent_checkins:
        scores = [c.adherence_score for c in recent_checkins if c.adherence_score is not None]
        avg_adherence = sum(scores) / len(scores) if scores else None

    style_key, style = get_style(user)

    observed_data = {
        "coaching_style": style_key,
        "goal": user.goal,
        "current_weight_kg": user.current_weight_kg,
        "weight_trend_kg": weight_trend,
        "avg_adherence": avg_adherence,
        "num_weight_entries": len(recent_weights),
        "num_checkins": len(recent_checkins),
    }

    agent_prompt = f"""You are an expert AI weight-management coach using a {style['label'].upper()} coaching style.
Decide the best action by calling exactly one primary tool (adjust_macros OR flag_for_review).

COACHING STYLE: {style['label']}
- Tone for every 'reason' you write: {style['tone']}
- Decision bias: {style['bias']}

User Data:
- Goal: {user.goal}
- Weight change over last 2 weeks (kg, positive = gained): {weight_trend}
- Avg adherence score (0-5): {avg_adherence}
- Current calorie target: {previous_target}

Rules:
1. If avg adherence is not None and <= {style['flag_below']}, call flag_for_review.
2. If adherence is above {style['flag_below']} and weight change is moving AWAY from the goal by at least
   {style['off_track_kg']} kg (goal is loss but weight went up, or goal is gain but weight went down,
   or goal is maintain and weight moved by that much either way),
   call adjust_macros with 'decrease' or 'increase' by {style['pct_range']} percent.
3. Otherwise, if progress matches the goal or the drift is smaller than {style['off_track_kg']} kg,
   call adjust_macros with 'maintain' and 0.
4. Do NOT call flag_for_review when adherence is above {style['flag_below']} unless the data is missing or absurd.
5. Optionally also call suggest_recipe when the user is on track.
Always fill in the 'reason' argument in plain text with no markdown, written in the {style['label']} tone above.
"""

    # ---------- 2. Start the reasoning timeline ----------
    trace = []
    _step(
        trace, "observe", "Observed your data",
        f"Goal: {user.goal} · weight change: {_fmt(weight_trend, ' kg')} · "
        f"adherence: {_fmt(avg_adherence, ' / 5')} · target: {previous_target} kcal",
    )

    _step(
        trace, "style", f"Coaching style: {style['label']}",
        f"Flags adherence ≤ {style['flag_below']} · treats a drift of {style['off_track_kg']} kg as off-goal · "
        f"adjusts by {style['pct_range']}%. {style['bias']}",
    )

    # ---------- 3. Ask Gemini to choose tools ----------
    client = _get_client()
    tools_called_log = []

    new_target = previous_target
    decision_type = "maintain_calories"
    reasoning_text = ""

    calls = []
    api_error = None
    latency = None

    try:
        started = time.perf_counter()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=agent_prompt,
            config=types.GenerateContentConfig(
                tools=AVAILABLE_TOOLS,
                # Stop the SDK from auto-running tools; WE execute them so we can log them.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY
                    )
                ),
                temperature=0.0,
            ),
        )
        latency = round(time.perf_counter() - started, 2)
        calls = response.function_calls or []
        try:
            record_usage("agent_tool_calling", response=response)
        except Exception:
            pass  # usage logging must never break a decision
    except Exception as e:
        api_error = e

    # "Considered" step: all tools Gemini was shown, and which ones it picked.
    called_names = [fc.name for fc in calls]
    _step(
        trace, "offer", "Tools offered to Gemini",
        "Gemini chose from these tools. Highlighted = called.",
        {"tools": [
            {"name": fn.__name__, "description": _first_doc_line(fn), "called": fn.__name__ in called_names}
            for fn in AVAILABLE_TOOLS
        ]},
    )

    # ---------- 4. Execute each tool call, in the order Gemini made them ----------
    if api_error is not None:
        decision_type = "flag_for_review"
        reasoning_text = f"Error: {api_error}"
        tools_called_log.append({"error": str(api_error)})
        _step(trace, "error", "Gemini call failed", str(api_error))
        _step(trace, "fallback", "Safe fallback: flag for human review",
              "Calories left unchanged because the agent could not decide.")

    elif not calls:
        decision_type = "flag_for_review"
        reasoning_text = "Model returned no tool call; flagged as a safe fallback."
        tools_called_log.append({"error": "No tools called by model."})
        _step(trace, "model", "Gemini returned no tool call", f"Responded in {latency}s without choosing a tool.")
        _step(trace, "fallback", "Safe fallback: flag for human review",
              "Calories left unchanged because the agent could not decide.")

    else:
        _step(trace, "model", f"Gemini chose {len(calls)} tool call{'s' if len(calls) != 1 else ''}",
              f"Decided in {latency}s. Executed in this order:")
        flagged = False
        reasons = []

        for position, fc in enumerate(calls, start=1):
            args = dict(fc.args) if fc.args else {}
            tools_called_log.append({"tool": fc.name, "args": args})
            known = TOOL_BY_NAME.get(fc.name)
            _step(trace, "call", f"#{position}  {fc.name}",
                  _first_doc_line(known) if known else "Not a registered tool.", {"args": args})

            try:
                if fc.name == "adjust_macros":
                    requested_pct = float(args.get("percentage_change", 0.0))
                    applied_pct = min(requested_pct, style["max_pct"])
                    if applied_pct < requested_pct:
                        # Guardrail: a style never lets the agent change calories by more than its cap.
                        _step(trace, "style", f"{style['label']} style capped the change",
                              f"Gemini asked for {requested_pct:g}%; {style['label']} allows at most "
                              f"{style['max_pct']:g}%, so {applied_pct:g}% was applied.")
                    result = adjust_macros(
                        current_calories=previous_target,
                        adjustment_type=str(args.get("adjustment_type", "maintain")),
                        percentage_change=applied_pct,
                        reason=str(args.get("reason", "")),
                    )
                    new_target = result["new_calories"]
                    decision_type = result["decision_type"]
                    reasons.append(result["reason"])
                    _step(trace, "result", f"Calorie target {previous_target} → {new_target} kcal",
                          result["reason"])

                elif fc.name == "flag_for_review":
                    result = flag_for_review(reason=str(args.get("reason", "Needs review.")))
                    flagged = True
                    reasons.append(result["reason"])
                    _step(trace, "result", "Flagged for human review", result["reason"])

                elif fc.name == "suggest_recipe":
                    result = suggest_recipe(
                        goal=str(args.get("goal", user.goal)),
                        calorie_limit=int(args.get("calorie_limit", previous_target)),
                    )
                    reasons.append(result["suggestion"])
                    _step(trace, "result", "Recipe suggestion", result["suggestion"])

                else:
                    flagged = True
                    _step(trace, "error", f"Unknown tool '{fc.name}'", "Ignored and flagged for safety.")

            except Exception as tool_err:
                flagged = True
                tools_called_log.append({"error": f"{fc.name} failed: {tool_err}"})
                _step(trace, "error", f"{fc.name} failed", str(tool_err))

        if flagged:
            decision_type = "flag_for_review"
            new_target = previous_target  # never change calories on a flag
        reasoning_text = " ".join(r for r in reasons if r) or "Tool calls executed."

    # ---------- 5. Final outcome step ----------
    _step(trace, "decision", dict(AgentDecision.DECISION_TYPE_CHOICES).get(decision_type, decision_type),
          f"Calorie target: {previous_target} → {new_target} kcal")

    decision = AgentDecision.objects.create(
        user=user,
        decision_type=decision_type,
        previous_calorie_target=previous_target,
        new_calorie_target=new_target,
        observed_data=observed_data,
        reasoning_text=reasoning_text,
        tools_called=tools_called_log,
        trace=trace,
    )

    if decision_type != "flag_for_review":
        user.current_calorie_target = new_target
        user.save()

    return decision
