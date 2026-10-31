from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """
    Extends Django's built-in User model with health/fitness profile fields.
    We extend rather than create a separate Profile model to keep queries
    simple — one join-free lookup gets you the user AND their fitness data.
    """

    GOAL_CHOICES = [
        ('lose', 'Lose Weight'),
        ('gain', 'Gain Weight'),
        ('maintain', 'Maintain Weight'),
    ]

    ACTIVITY_CHOICES = [
        ('sedentary', 'Sedentary (little to no exercise)'),
        ('light', 'Lightly Active (1-3 days/week)'),
        ('moderate', 'Moderately Active (3-5 days/week)'),
        ('active', 'Very Active (6-7 days/week)'),
    ]

    COACHING_STYLE_CHOICES = [
        ('strict', 'Strict'),
        ('motivational', 'Motivational'),
        ('gentle', 'Gentle'),
    ]

    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ]

    height_cm = models.FloatField(null=True, blank=True, help_text="Height in centimeters")
    current_weight_kg = models.FloatField(null=True, blank=True)
    target_weight_kg = models.FloatField(null=True, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, blank=True)
    goal = models.CharField(max_length=10, choices=GOAL_CHOICES, default='maintain')
    activity_level = models.CharField(max_length=15, choices=ACTIVITY_CHOICES, default='sedentary')

    # How the AI coach talks to the user AND how eagerly it adjusts calories
    coaching_style = models.CharField(max_length=15, choices=COACHING_STYLE_CHOICES, default='motivational')

    # Computed/cached target — updated by the agent, not the user directly
    current_calorie_target = models.PositiveIntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username