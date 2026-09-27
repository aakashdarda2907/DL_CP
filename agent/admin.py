from django.contrib import admin
from .models import AgentDecision


@admin.register(AgentDecision)
class AgentDecisionAdmin(admin.ModelAdmin):
    list_display = ['user', 'decision_type', 'previous_calorie_target', 'new_calorie_target', 'created_at']
    list_filter = ['decision_type']
    search_fields = ['user__username']