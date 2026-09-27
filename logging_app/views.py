from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import MealLog, WeightEntry, DailyCheckIn
from ml_layer.vit_module.food_classifier import classify_food_image  # built in Step 5
from ml_layer.bert_module.text_parser import parse_meal_text          # built in Step 5


@login_required
def log_meal_photo(request):
    if request.method == 'POST':
        image = request.FILES.get('meal_image')

        if not image:
            messages.error(request, 'Please upload an image.')
            return redirect('logging_app:log_meal_photo')

        # Save the log first so we have a record even if ML fails
        meal_log = MealLog.objects.create(
            user=request.user,
            input_type='photo',
            image=image,
        )

        # Run ViT classification on the saved image
        result = classify_food_image(meal_log.image.path)

        meal_log.predicted_items = result.get('items', {})
        meal_log.estimated_calories = result.get('calories')
        meal_log.estimated_protein_g = result.get('protein_g')
        meal_log.estimated_carbs_g = result.get('carbs_g')
        meal_log.estimated_fats_g = result.get('fats_g')
        meal_log.save()

        messages.success(request, 'Meal logged from photo.')
        return redirect('logging_app:meal_result', meal_id=meal_log.id)

    return render(request, 'logging_app/log_meal_photo.html')


@login_required
def log_meal_text(request):
    if request.method == 'POST':
        raw_text = request.POST.get('meal_text', '').strip()

        if not raw_text:
            messages.error(request, 'Please describe your meal.')
            return redirect('logging_app:log_meal_text')

        meal_log = MealLog.objects.create(
            user=request.user,
            input_type='text',
            raw_text=raw_text,
        )

        # Run BERT-based parsing on the text
        result = parse_meal_text(raw_text)

        meal_log.predicted_items = result.get('items', {})
        meal_log.estimated_calories = result.get('calories')
        meal_log.estimated_protein_g = result.get('protein_g')
        meal_log.estimated_carbs_g = result.get('carbs_g')
        meal_log.estimated_fats_g = result.get('fats_g')
        meal_log.save()

        messages.success(request, 'Meal logged from text.')
        return redirect('logging_app:meal_result', meal_id=meal_log.id)

    return render(request, 'logging_app/log_meal_text.html')


@login_required
def meal_result(request, meal_id):
    """Shows prediction results and lets user correct them."""
    meal_log = get_object_or_404(MealLog, id=meal_id, user=request.user)

    if request.method == 'POST':
        # User is submitting a correction
        meal_log.estimated_calories = request.POST.get('estimated_calories') or meal_log.estimated_calories
        meal_log.estimated_protein_g = request.POST.get('estimated_protein_g') or meal_log.estimated_protein_g
        meal_log.estimated_carbs_g = request.POST.get('estimated_carbs_g') or meal_log.estimated_carbs_g
        meal_log.estimated_fats_g = request.POST.get('estimated_fats_g') or meal_log.estimated_fats_g
        meal_log.correction_notes = request.POST.get('correction_notes', '')
        meal_log.is_corrected = True
        meal_log.save()
        messages.success(request, 'Correction saved. Thanks — this helps improve future predictions.')
        return redirect('dashboard:home')

    return render(request, 'logging_app/meal_result.html', {'meal': meal_log})


@login_required
def meal_history(request):
    meals = MealLog.objects.filter(user=request.user)
    return render(request, 'logging_app/meal_history.html', {'meals': meals})


@login_required
def log_weight(request):
    if request.method == 'POST':
        weight_kg = request.POST.get('weight_kg')
        note = request.POST.get('note', '')

        if not weight_kg:
            messages.error(request, 'Please enter a weight.')
            return redirect('logging_app:log_weight')

        WeightEntry.objects.create(user=request.user, weight_kg=float(weight_kg), note=note)

        # Keep the user's current_weight_kg in sync
        request.user.current_weight_kg = float(weight_kg)
        request.user.save()

        messages.success(request, 'Weight logged.')
        return redirect('dashboard:home')

    return render(request, 'logging_app/log_weight.html')


@login_required
def daily_checkin(request):
    if request.method == 'POST':
        DailyCheckIn.objects.create(
            user=request.user,
            mood=request.POST.get('mood', ''),
            energy_level=request.POST.get('energy_level') or None,
            adherence_score=request.POST.get('adherence_score') or None,
            notes=request.POST.get('notes', ''),
        )
        messages.success(request, 'Check-in recorded.')
        return redirect('dashboard:home')

    return render(request, 'logging_app/daily_checkin.html')