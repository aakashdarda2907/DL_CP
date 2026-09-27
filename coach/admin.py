from django.contrib import admin
from .models import ApiUsageLog, ChatMessage


@admin.register(ApiUsageLog)
class ApiUsageLogAdmin(admin.ModelAdmin):
    list_display = ["source", "total_tokens", "was_cached", "was_hardcoded", "created_at"]
    list_filter = ["source", "was_cached", "was_hardcoded"]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["user", "role", "message", "created_at"]
    list_filter = ["role"]
    search_fields = ["user__username", "message"]
