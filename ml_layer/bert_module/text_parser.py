"""
Parses free-text meal descriptions ("2 rotis and dal") into structured
food items using semantic similarity from a BERT-family sentence
embedding model - no training required (zero-shot matching against our
known food list).
"""

import re
from ml_layer.utils import get_embedder
from ml_layer.nutrition_data import FOOD_NUTRITION, DEFAULT_UNKNOWN_ITEM
from sentence_transformers import util as st_util

SIMILARITY_THRESHOLD = 0.55


def parse_meal_text(text):
    """
    Simplification (80/20): instead of full NER with quantity-to-food
    linking, we (1) split text into candidate phrases, (2) embed each
    phrase and compare to embeddings of known food names, (3) pick
    phrases above a similarity threshold, (4) look for a leading number
    as the serving count (defaults to 1 if none found).
    """
    embedder = get_embedder()

    known_foods = list(FOOD_NUTRITION.keys())
    food_embeddings = embedder.encode(known_foods, convert_to_tensor=True)

    raw_phrases = re.split(r"\band\b|,|\bwith\b", text.lower())
    raw_phrases = [p.strip() for p in raw_phrases if p.strip()]

    matched_items = {}
    total = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}

    for phrase in raw_phrases:
        qty_match = re.search(r"(\d+)", phrase)
        quantity = int(qty_match.group(1)) if qty_match else 1

        phrase_embedding = embedder.encode(phrase, convert_to_tensor=True)
        similarities = st_util.cos_sim(phrase_embedding, food_embeddings)[0]

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])

        if best_score >= SIMILARITY_THRESHOLD:
            food_name = known_foods[best_idx]
            nutrition = FOOD_NUTRITION[food_name]
        else:
            food_name = phrase
            nutrition = DEFAULT_UNKNOWN_ITEM

        matched_items[food_name] = {"confidence": round(best_score, 3), "servings": quantity}

        total["calories"] += nutrition["calories"] * quantity
        total["protein_g"] += nutrition["protein_g"] * quantity
        total["carbs_g"] += nutrition["carbs_g"] * quantity
        total["fats_g"] += nutrition["fats_g"] * quantity

    return {
        "items": matched_items,
        "calories": round(total["calories"], 1),
        "protein_g": round(total["protein_g"], 1),
        "carbs_g": round(total["carbs_g"], 1),
        "fats_g": round(total["fats_g"], 1),
    }
