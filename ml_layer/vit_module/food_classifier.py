"""
Food image classification. Primary path: our FINE-TUNED linear head on
frozen CLIP features. Since the head is trained as a single-label softmax
classifier, we approximate multi-item detection via top-k probability
thresholding - if a second class has meaningfully high probability, we
report it too, rather than forcing a single label per photo. This is a
lightweight approximation, not full object detection - noted as a scoping
decision, not treated as detection-grade multi-object recognition.
Fallback path: zero-shot CLIP matching for foods outside the 10 trained
classes.
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from ml_layer.nutrition_data import FOOD_NUTRITION, DEFAULT_UNKNOWN_ITEM

_clip_model = None
_clip_processor = None
_classifier_head = None
_trained_labels = None

MODULE_DIR = os.path.dirname(__file__)
CONFIDENCE_THRESHOLD = 0.4       # below this for the top item, fall back to zero-shot entirely
SECOND_ITEM_THRESHOLD = 0.15     # a second class above this probability is also reported
MAX_ITEMS = 3                    # cap to avoid noisy low-probability tails


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_model, _clip_processor


def _extract_image_features(clip_model, pixel_values):
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        image_features = clip_model.visual_projection(pooled_output)
    return image_features


def _load_fine_tuned_head():
    global _classifier_head, _trained_labels
    if _classifier_head is None:
        labels_path = os.path.join(MODULE_DIR, "labels.json")
        weights_path = os.path.join(MODULE_DIR, "classifier_head.pt")

        if not os.path.exists(labels_path) or not os.path.exists(weights_path):
            return None, None

        with open(labels_path) as f:
            _trained_labels = json.load(f)

        clip_model, _ = _load_clip()
        feature_dim = clip_model.config.projection_dim

        _classifier_head = nn.Linear(feature_dim, len(_trained_labels))
        _classifier_head.load_state_dict(torch.load(weights_path))
        _classifier_head.eval()

    return _classifier_head, _trained_labels


def _zero_shot_classify(image, model, processor):
    candidate_labels = list(FOOD_NUTRITION.keys())
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    best_idx = int(probs.argmax())
    return candidate_labels[best_idx], float(probs[best_idx])


def classify_food_image(image_path):
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")

    classifier_head, trained_labels = _load_fine_tuned_head()
    detected_items = {}

    if classifier_head is not None:
        inputs = processor(images=image, return_tensors="pt")
        features = _extract_image_features(model, inputs["pixel_values"])

        with torch.no_grad():
            logits = classifier_head(features)
            probs = torch.softmax(logits, dim=1)[0]

        sorted_indices = torch.argsort(probs, descending=True)
        top_idx = int(sorted_indices[0])
        top_confidence = float(probs[top_idx])

        if top_confidence < CONFIDENCE_THRESHOLD:
            # Not confident enough even for the top guess - fall back to zero-shot
            best_label, confidence = _zero_shot_classify(image, model, processor)
            detected_items[best_label] = {"confidence": round(confidence, 3), "servings": 1}
        else:
            # Always include the top prediction
            detected_items[trained_labels[top_idx]] = {"confidence": round(top_confidence, 3), "servings": 1}

            # Check subsequent classes for genuine secondary items (multi-food photo)
            for rank in range(1, min(MAX_ITEMS, len(sorted_indices))):
                idx = int(sorted_indices[rank])
                confidence = float(probs[idx])
                if confidence >= SECOND_ITEM_THRESHOLD:
                    detected_items[trained_labels[idx]] = {"confidence": round(confidence, 3), "servings": 1}
                else:
                    break  # probabilities are sorted, so no point checking further
    else:
        best_label, confidence = _zero_shot_classify(image, model, processor)
        detected_items[best_label] = {"confidence": round(confidence, 3), "servings": 1}

    # Sum nutrition across all detected items
    totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fats_g": 0.0}
    for item_name, item_data in detected_items.items():
        nutrition = FOOD_NUTRITION.get(item_name, DEFAULT_UNKNOWN_ITEM)
        servings = item_data["servings"]
        totals["calories"] += nutrition["calories"] * servings
        totals["protein_g"] += nutrition["protein_g"] * servings
        totals["carbs_g"] += nutrition["carbs_g"] * servings
        totals["fats_g"] += nutrition["fats_g"] * servings

    return {
        "items": detected_items,
        "calories": round(totals["calories"], 1),
        "protein_g": round(totals["protein_g"], 1),
        "carbs_g": round(totals["carbs_g"], 1),
        "fats_g": round(totals["fats_g"], 1),
    }
