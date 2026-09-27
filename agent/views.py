from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import AgentDecision
from ml_layer.agent_module.planner import run_weekly_agent  # built in Step 5


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
    decision = AgentDecision.objects.get(id=decision_id, user=request.user)
    return render(request, 'agent/decision_detail.html', {'decision': decision})