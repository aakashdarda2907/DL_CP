"""
Shared helper: loads the sentence-embedding model ONCE and reuses it
across the BERT text-parsing module and the Gemini RAG module.
Loading it twice would waste memory on a laptop, so this acts as a
simple singleton.
"""

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
