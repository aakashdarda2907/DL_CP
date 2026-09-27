from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Sum, Count
from logging_app.models import MealLog, WeightEntry, DailyCheckIn
from agent.models import AgentDecision
from coach.models import ApiUsageLog


@login_required
def home(request):
    user = request.user

    recent_meals = MealLog.objects.filter(user=user)[:5]
    weight_history = WeightEntry.objects.filter(user=user)[:30]
    recent_checkins = DailyCheckIn.objects.filter(user=user)[:7]
    latest_decision = AgentDecision.objects.filter(user=user).first()

    usage_stats = ApiUsageLog.objects.aggregate(
        total_tokens=Sum('total_tokens'),
        total_calls=Count('id'),
    )
    cached_or_hardcoded_count = (
        ApiUsageLog.objects.filter(was_cached=True).count()
        + ApiUsageLog.objects.filter(was_hardcoded=True).count()
    )

    context = {
        'recent_meals': recent_meals,
        'weight_history': weight_history,
        'recent_checkins': recent_checkins,
        'latest_decision': latest_decision,
        'calorie_target': user.current_calorie_target,
        'total_tokens_used': usage_stats['total_tokens'] or 0,
        'total_api_calls': usage_stats['total_calls'] or 0,
        'calls_saved_by_caching': cached_or_hardcoded_count,
    }

    return render(request, 'dashboard/home.html', context)
