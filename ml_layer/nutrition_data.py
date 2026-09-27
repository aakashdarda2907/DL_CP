"""
Static nutrition lookup table - a small, curated set of common foods.
This intentionally trades completeness for simplicity (80/20 principle):
instead of calling a live nutrition API, we look up approximate values
from this dictionary. Good enough for a demo; easy to extend later.

Values are approximate, per typical single serving.
"""

FOOD_NUTRITION = {
    "rice":        {"calories": 200, "protein_g": 4.0,  "carbs_g": 45.0, "fats_g": 0.5},
    "roti":        {"calories": 120, "protein_g": 3.5,  "carbs_g": 22.0, "fats_g": 2.0},
    "chapati":     {"calories": 120, "protein_g": 3.5,  "carbs_g": 22.0, "fats_g": 2.0},
    "dal":         {"calories": 180, "protein_g": 9.0,  "carbs_g": 27.0, "fats_g": 4.0},
    "paneer":      {"calories": 265, "protein_g": 18.0, "carbs_g": 6.0,  "fats_g": 20.0},
    "chicken":     {"calories": 240, "protein_g": 27.0, "carbs_g": 0.0,  "fats_g": 14.0},
    "egg":         {"calories": 78,  "protein_g": 6.0,  "carbs_g": 0.6,  "fats_g": 5.0},
    "salad":       {"calories": 50,  "protein_g": 2.0,  "carbs_g": 10.0, "fats_g": 0.3},
    "curd":        {"calories": 98,  "protein_g": 11.0, "carbs_g": 4.0,  "fats_g": 4.3},
    "yogurt":      {"calories": 98,  "protein_g": 11.0, "carbs_g": 4.0,  "fats_g": 4.3},
    "banana":      {"calories": 105, "protein_g": 1.3,  "carbs_g": 27.0, "fats_g": 0.4},
    "apple":       {"calories": 95,  "protein_g": 0.5,  "carbs_g": 25.0, "fats_g": 0.3},
    "bread":       {"calories": 80,  "protein_g": 3.0,  "carbs_g": 15.0, "fats_g": 1.0},
    "pizza":       {"calories": 285, "protein_g": 12.0, "carbs_g": 36.0, "fats_g": 10.0},
    "burger":      {"calories": 350, "protein_g": 17.0, "carbs_g": 33.0, "fats_g": 17.0},
    "pasta":       {"calories": 220, "protein_g": 8.0,  "carbs_g": 43.0, "fats_g": 1.3},
    "sandwich":    {"calories": 250, "protein_g": 10.0, "carbs_g": 30.0, "fats_g": 9.0},
    "noodles":     {"calories": 220, "protein_g": 6.0,  "carbs_g": 40.0, "fats_g": 4.0},
    "fish":        {"calories": 206, "protein_g": 22.0, "carbs_g": 0.0,  "fats_g": 12.0},
    "potato":      {"calories": 160, "protein_g": 4.0,  "carbs_g": 37.0, "fats_g": 0.2},
    "milk":        {"calories": 122, "protein_g": 8.0,  "carbs_g": 12.0, "fats_g": 5.0},
    "oats":        {"calories": 150, "protein_g": 5.0,  "carbs_g": 27.0, "fats_g": 3.0},
    "idli":        {"calories": 39,  "protein_g": 2.0,  "carbs_g": 8.0,  "fats_g": 0.2},
    "dosa":        {"calories": 168, "protein_g": 3.9,  "carbs_g": 29.0, "fats_g": 3.7},
    "biryani":     {"calories": 350, "protein_g": 12.0, "carbs_g": 45.0, "fats_g": 12.0},
    "samosa":      {"calories": 262, "protein_g": 3.5,  "carbs_g": 24.0, "fats_g": 17.0},
    "juice":       {"calories": 110, "protein_g": 0.5,  "carbs_g": 26.0, "fats_g": 0.2},
    "soup":        {"calories": 90,  "protein_g": 4.0,  "carbs_g": 12.0, "fats_g": 3.0},
    "chocolate":   {"calories": 150, "protein_g": 2.0,  "carbs_g": 17.0, "fats_g": 8.0},
    "ice cream":   {"calories": 210, "protein_g": 3.5,  "carbs_g": 24.0, "fats_g": 11.0},
}

DEFAULT_UNKNOWN_ITEM = {"calories": 150, "protein_g": 5.0, "carbs_g": 20.0, "fats_g": 5.0}


# Added for fine-tuned classifier classes (Food-101 subset)
FOOD_NUTRITION.update({
    "fried_rice":    {"calories": 238, "protein_g": 5.0,  "carbs_g": 45.0, "fats_g": 4.0},
    "chicken_curry": {"calories": 240, "protein_g": 20.0, "carbs_g": 8.0,  "fats_g": 15.0},
    "omelette":      {"calories": 154, "protein_g": 11.0, "carbs_g": 1.5,  "fats_g": 12.0},
    "spring_rolls":  {"calories": 150, "protein_g": 4.0,  "carbs_g": 18.0, "fats_g": 7.0},
    "ramen":         {"calories": 436, "protein_g": 18.0, "carbs_g": 65.0, "fats_g": 14.0},
    "french_fries":  {"calories": 312, "protein_g": 3.4,  "carbs_g": 41.0, "fats_g": 15.0},
    "hamburger":     {"calories": 350, "protein_g": 17.0, "carbs_g": 33.0, "fats_g": 17.0},
    "ice_cream":     {"calories": 210, "protein_g": 3.5,  "carbs_g": 24.0, "fats_g": 11.0},
})
