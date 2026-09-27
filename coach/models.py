from django.db import models


class ApiUsageLog(models.Model):
    SOURCE_CHOICES = [
        ("chat", "Coach Chat"),
        ("agent_explanation", "Agent Explanation"),
    ]

    source = models.CharField(max_length=30, choices=SOURCE_CHOICES)
    prompt_tokens = models.PositiveIntegerField(default=0)
    completion_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    was_cached = models.BooleanField(default=False)
    was_hardcoded = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.source} - {self.total_tokens} tokens - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ["-created_at"]


class ChatMessage(models.Model):
    """
    Stores every chat turn so conversations persist across page reloads,
    and so we can feed recent history back into Gemini as context - giving
    the coach real conversational memory instead of treating every message
    as an isolated question.
    """
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]

    user = models.ForeignKey("accounts.CustomUser", on_delete=models.CASCADE, related_name="chat_messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.role} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ["created_at"]
