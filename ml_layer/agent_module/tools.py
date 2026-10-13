def recalculate_tdee(weight_kg: float, height_cm: float, age: int, gender: str, activity_level: str) -> dict:
    """
    Calculates the user's Total Daily Energy Expenditure (TDEE).
    Call this when a user needs a total reset of their baseline calories.
    """
    if gender == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    tdee = bmr * multipliers.get(activity_level, 1.2)
    return {"status": "success", "new_tdee": round(tdee)}

def adjust_macros(current_calories: int, adjustment_type: str, percentage_change: float, reason: str) -> dict:
    """
    Adjusts the user's daily calorie target based on recent weight trends.

    Args:
        current_calories: The current daily calorie limit.
        adjustment_type: Must be strictly 'increase', 'decrease', or 'maintain'.
        percentage_change: Percent to change (e.g. 10.0 for 10%, 0.0 for maintain). Max 15.
        reason: One or two plain sentences explaining why, addressed to the user.
    """
    percentage_change = max(0.0, min(float(percentage_change), 15.0))  # safety clamp
    if adjustment_type == 'decrease':
        new_cal = round(current_calories * (1 - percentage_change / 100))
        decision = "decrease_calories"
    elif adjustment_type == 'increase':
        new_cal = round(current_calories * (1 + percentage_change / 100))
        decision = "increase_calories"
    else:
        new_cal = current_calories
        decision = "maintain_calories"
    return {"status": "success", "new_calories": new_cal, "decision_type": decision, "reason": reason}

def flag_for_review(reason: str) -> dict:
    """
    Flags the user's profile for manual review by a human coach.
    Call this if adherence is <= 2.0, weight trends are alarming, or data is wildly inconsistent.
    """
    return {"status": "flagged", "decision_type": "flag_for_review", "reason": reason}

def suggest_recipe(goal: str, calorie_limit: int) -> dict:
    """
    Suggests a recipe archetype for the user. 
    Call this as an optional bonus tool if the user is maintaining and doing well.
    """
    return {"status": "success", "suggestion": f"A {goal}-friendly meal under {calorie_limit} kcal"}