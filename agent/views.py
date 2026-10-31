from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import AgentDecision
from ml_layer.agent_module.planner import run_weekly_agent


def _timeline_for(decision):
    """
    Return the steps to draw on the timeline.
    New decisions have a full `trace`. Older decisions (made before the
    timeline existed) only have `tools_called`, so we rebuild a simple trace
    from that so the page never looks empty.
    """
    if decision.trace:
        return decision.trace

    steps = []
    for entry in decision.tools_called or []:
        if "error" in entry:
            steps.append({"kind": "error", "title": "Error", "detail": entry["error"], "data": {}})
        else:
            steps.append({
                "kind": "call",
                "title": entry.get("tool", "tool"),
                "detail": "",
                "data": {"args": entry.get("args", {})},
            })
    return steps


@login_required
def trigger_agent(request):
    """
    Manually trigger the agent's weekly review (later this can be
    automated via a scheduled task, but manual trigger is fine for the demo).
    """
    if request.method == 'POST':
        decision = run_weekly_agent(request.user)
        messages.success(request, 'Agent has reviewed your progress.')
        return redirect('agent:decision_detail', decision_id=decision.id)

    return render(request, 'agent/trigger_agent.html')


@login_required
def decision_history(request):
    decisions = AgentDecision.objects.filter(user=request.user)
    return render(request, 'agent/decision_history.html', {'decisions': decisions})


@login_required
def decision_detail(request, decision_id):
    # get_object_or_404: a wrong/other-user id shows a 404 page instead of crashing with a 500
    decision = get_object_or_404(AgentDecision, id=decision_id, user=request.user)
    return render(request, 'agent/decision_detail.html', {
        'decision': decision,
        'timeline': _timeline_for(decision),
    })
