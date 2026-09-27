"""
Seeds a realistic test user with two weeks of history, so every
dashboard feature (weight trend, adherence, agent decisions) has
real data to work with instead of an empty state.

Run with: python manage.py seed_dummy_data
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from accounts.models import CustomUser
from logging_app.models import WeightEntry, DailyCheckIn, MealLog


class Command(BaseCommand):
    help = "Seeds a test user (test12) with two weeks of realistic history."

    def handle(self, *args, **options):
        username = "test12"
        password = "testpass123"

        user, created = CustomUser.objects.get_or_create(
            username=username,
            defaults={"email": "test12@example.com"},
        )
        if created:
            user.set_password(password)

        user.height_cm = 178
        user.age = 24
        user.gender = "male"
        user.goal = "gain"
        user.activity_level = "moderate"
        user.current_weight_kg = 64.0
        user.target_weight_kg = 75.0
        user.save()

        self.stdout.write(f"User '{username}' ready (password: {password})")

        # Clear any previous seeded data for a clean re-run
        WeightEntry.objects.filter(user=user).delete()
        DailyCheckIn.objects.filter(user=user).delete()
        MealLog.objects.filter(user=user).delete()

        now = timezone.now()

        # Two weeks of weight entries, trending up slightly (consistent with a "gain" goal)
        weight_progression = [63.0, 63.2, 63.3, 63.5, 63.6, 63.8, 64.0]
        days_ago_points = [13, 11, 9, 7, 5, 3, 1]

        for weight, days_ago in zip(weight_progression, days_ago_points):
            entry = WeightEntry.objects.create(user=user, weight_kg=weight)
            WeightEntry.objects.filter(id=entry.id).update(recorded_at=now - timedelta(days=days_ago))

        self.stdout.write(f"Seeded {len(weight_progression)} weight entries (63.0kg -> 64.0kg over 13 days).")

        # One week of check-ins with decent adherence (so the agent won't flag for review)
        moods = ["okay", "great", "okay", "great", "low", "okay", "great"]
        adherence_scores = [4, 5, 4, 5, 3, 4, 5]
        energy_levels = [3, 4, 3, 4, 2, 3, 4]

        for i in range(7):
            days_ago = 6 - i
            checkin = DailyCheckIn.objects.create(
                user=user,
                mood=moods[i],
                energy_level=energy_levels[i],
                adherence_score=adherence_scores[i],
                notes="Seeded test data.",
            )
            DailyCheckIn.objects.filter(id=checkin.id).update(date=(now - timedelta(days=days_ago)).date())

        self.stdout.write("Seeded 7 daily check-ins (avg adherence ~4.3/5).")

        # A couple of realistic meal logs (text-based, matching known nutrition entries)
        meal_samples = [
            {
                "raw_text": "2 rotis and dal",
                "predicted_items": {"roti": {"confidence": 0.7, "servings": 2}, "dal": {"confidence": 1.0, "servings": 1}},
                "calories": 420.0, "protein_g": 16.0, "carbs_g": 71.0, "fats_g": 8.0, "days_ago": 2,
            },
            {
                "raw_text": "chicken and rice",
                "predicted_items": {"chicken": {"confidence": 0.9, "servings": 1}, "rice": {"confidence": 0.85, "servings": 1}},
                "calories": 440.0, "protein_g": 31.0, "carbs_g": 45.0, "fats_g": 14.5, "days_ago": 1,
            },
        ]

        for sample in meal_samples:
            meal = MealLog.objects.create(
                user=user,
                input_type="text",
                raw_text=sample["raw_text"],
                predicted_items=sample["predicted_items"],
                estimated_calories=sample["calories"],
                estimated_protein_g=sample["protein_g"],
                estimated_carbs_g=sample["carbs_g"],
                estimated_fats_g=sample["fats_g"],
            )
            MealLog.objects.filter(id=meal.id).update(logged_at=now - timedelta(days=sample["days_ago"]))

        self.stdout.write("Seeded 2 sample meal logs.")
        self.stdout.write(self.style.SUCCESS(f"\nDone. Log in as '{username}' / '{password}' to test."))
