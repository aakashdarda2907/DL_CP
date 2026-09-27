from django.db import models
from django.conf import settings


class AgentDecision(models.Model):
    """
    Logs every decision the agent makes — this is what powers the
    'why did my calories change?' transparency feature, and doubles
    as your evaluation trail for the project report.
    """

    DECISION_TYPE_CHOICES = [
        ('increase_calories', 'Increase Calories'),
        ('decrease_calories', 'Decrease Calories'),
        ('maintain_calories', 'Maintain Current Plan'),
        ('flag_for_review', 'Flag for Human Review'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='agent_decisions')

    decision_type = models.CharField(max_length=25, choices=DECISION_TYPE_CHOICES)

    previous_calorie_target = models.PositiveIntegerField(null=True, blank=True)
    new_calorie_target = models.PositiveIntegerField(null=True, blank=True)

    # The raw signals the agent observed — stored for transparency/debugging
    observed_data = models.JSONField(default=dict, blank=True)

    # Human-readable explanation (LLM-generated) shown to the user
    reasoning_text = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.decision_type} - {self.created_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-created_at']