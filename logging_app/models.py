from django.db import models
from django.conf import settings


class MealLog(models.Model):
    """
    A single meal entry — can come from a photo upload or free text.
    Both paths converge here so the dashboard/agent don't care which
    input method the user used.
    """

    INPUT_TYPE_CHOICES = [
        ('photo', 'Photo'),
        ('text', 'Text'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='meal_logs')
    input_type = models.CharField(max_length=10, choices=INPUT_TYPE_CHOICES)

    image = models.ImageField(upload_to='meal_images/', null=True, blank=True)
    raw_text = models.TextField(null=True, blank=True, help_text="Original text input, if input_type is 'text'")

    # Output of ViT/BERT prediction — stored as JSON so we don't need
    # a rigid schema while the models are still being tuned
    predicted_items = models.JSONField(default=dict, blank=True)

    estimated_calories = models.FloatField(null=True, blank=True)
    estimated_protein_g = models.FloatField(null=True, blank=True)
    estimated_carbs_g = models.FloatField(null=True, blank=True)
    estimated_fats_g = models.FloatField(null=True, blank=True)

    # If the user corrects a wrong prediction — this data is valuable
    # for future fine-tuning, so we keep both versions
    is_corrected = models.BooleanField(default=False)
    correction_notes = models.TextField(null=True, blank=True)

    logged_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.input_type} - {self.logged_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['-logged_at']


class WeightEntry(models.Model):
    """
    Tracks weight over time — this is the ground truth signal
    the agent uses to judge whether the current plan is working.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='weight_entries')
    weight_kg = models.FloatField()
    recorded_at = models.DateTimeField(auto_now_add=True)
    note = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.weight_kg}kg - {self.recorded_at.strftime('%Y-%m-%d')}"

    class Meta:
        ordering = ['-recorded_at']


class DailyCheckIn(models.Model):
    """
    Captures the 'soft' signals — mood, energy, adherence — that pure
    calorie/weight numbers miss. Feeds the agent's reasoning step.
    """
    MOOD_CHOICES = [
        ('great', 'Great'),
        ('okay', 'Okay'),
        ('low', 'Low'),
        ('stressed', 'Stressed'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='check_ins')
    date = models.DateField(auto_now_add=True)
    mood = models.CharField(max_length=10, choices=MOOD_CHOICES, blank=True)
    energy_level = models.PositiveIntegerField(null=True, blank=True, help_text="Scale of 1-5")
    adherence_score = models.PositiveIntegerField(null=True, blank=True, help_text="Self-rated 1-5, how well they stuck to plan")
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.date}"

    class Meta:
        ordering = ['-date']