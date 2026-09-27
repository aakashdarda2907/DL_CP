from django.contrib import admin
from .models import MealLog, WeightEntry, DailyCheckIn


@admin.register(MealLog)
class MealLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'input_type', 'estimated_calories', 'is_corrected', 'logged_at']
    list_filter = ['input_type', 'is_corrected']
    search_fields = ['user__username', 'raw_text']


@admin.register(WeightEntry)
class WeightEntryAdmin(admin.ModelAdmin):
    list_display = ['user', 'weight_kg', 'recorded_at']
    search_fields = ['user__username']


@admin.register(DailyCheckIn)
class DailyCheckInAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'mood', 'energy_level', 'adherence_score']
    list_filter = ['mood']