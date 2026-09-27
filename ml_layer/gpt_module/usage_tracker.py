"""
Shared helpers for the Gemini API layer:
- record_usage(): logs every real API call's token usage to the DB
- get_cached_or_none() / set_cache(): simple response caching using
  Django's built-in cache framework, to avoid repeat calls for
  identical inputs (no Redis needed - uses Django's default LocMemCache)
"""

import hashlib
from django.core.cache import cache

CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours - chat/explanation content doesn't need to be fresher than this


def make_cache_key(prefix, *parts):
    raw = prefix + "::" + "::".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"gemini_cache:{prefix}:{digest}"


def get_cached_or_none(key):
    return cache.get(key)


def set_cache(key, value):
    cache.set(key, value, timeout=CACHE_TTL_SECONDS)


def record_usage(source, response, was_cached=False, was_hardcoded=False):
    from coach.models import ApiUsageLog

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if response is not None and hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

    ApiUsageLog.objects.create(
        source=source,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        was_cached=was_cached,
        was_hardcoded=was_hardcoded,
    )
