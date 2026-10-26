"""
Tiny static knowledge base for the RAG-grounded coach.
Simplification (80/20): a hardcoded list of ~15 nutrition/fitness facts
instead of a full document corpus + vector database. Embedded once using
the shared BERT-family embedder.
"""

from ml_layer.utils import get_embedder

KNOWLEDGE_BASE = [
    "A safe rate of weight loss is about 0.5 to 1 kg per week, achieved through a moderate calorie deficit.",
    "A calorie deficit of roughly 500 kcal per day typically leads to about 0.5 kg of weight loss per week.",
    "Protein intake of around 1.6 to 2.2 grams per kg of body weight supports muscle retention during weight loss.",
    "Weight loss plateaus are common and often due to metabolic adaptation, water retention, or underestimated calorie intake.",
    "Strength training combined with a calorie deficit helps preserve muscle mass while losing fat.",
    "Weight gain for muscle building typically requires a calorie surplus of 250 to 500 kcal per day.",
    "Sleep deprivation can increase hunger hormones and make weight management harder.",
    "Drinking water before meals can help with portion control and satiety.",
    "Extremely low-calorie diets can slow metabolism and are generally not sustainable long-term.",
    "Fiber-rich foods like vegetables and whole grains promote satiety and support digestive health.",
    "Consistency in tracking food intake, even imperfectly, is more effective than occasional perfect tracking.",
    "Stress and poor sleep can affect weight management independent of diet and exercise.",
    "Muscle weighs more densely than fat, so body composition changes may not always show on the scale.",
    "Sudden large weight changes on the scale are often water weight, not fat gain or loss.",
    "Sustainable weight management focuses on long-term habits rather than short-term restrictive diets.",
]

_kb_embeddings = None

# Similarity threshold chosen empirically from the RAG evaluation dataset.
# Evaluation showed a clean gap between the minimum in-scope expected fact
# score (0.3935) and the maximum out-of-scope top-1 score (0.1146).
# A threshold of 0.35 sits safely above all out-of-scope scores while
# preserving recall for every in-scope query in the evaluation set.
# See: ml_layer/gpt_module/baseline_results.json and evaluate_retrieval.py.
DEFAULT_SIMILARITY_THRESHOLD = 0.35


def get_relevant_context(query, top_k=3, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """Return up to top_k knowledge-base facts most relevant to the query.

    Args:
        query: User's natural-language question.
        top_k: Maximum number of facts to return.
        similarity_threshold: Minimum cosine similarity a fact must reach to
            be included. Facts below this score are silently dropped, even if
            they rank in the top-k. Pass ``None`` to disable filtering and
            restore the original behaviour (returns exactly top_k facts).

    Returns:
        List of fact strings, ordered by descending similarity.  May be
        shorter than top_k (or empty) when few facts exceed the threshold.
    """
    global _kb_embeddings
    from sentence_transformers import util as st_util

    embedder = get_embedder()

    if _kb_embeddings is None:
        _kb_embeddings = embedder.encode(KNOWLEDGE_BASE, convert_to_tensor=True)

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = st_util.cos_sim(query_embedding, _kb_embeddings)[0]

    top_indices = similarities.argsort(descending=True)[:top_k]

    if similarity_threshold is None:
        # Backward-compatible: no filtering, return exactly top_k facts.
        return [KNOWLEDGE_BASE[i] for i in top_indices]

    return [
        KNOWLEDGE_BASE[i]
        for i in top_indices
        if similarities[i].item() >= similarity_threshold
    ]
