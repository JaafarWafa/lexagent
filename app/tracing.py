"""Optional Langfuse tracing. On only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

When on, every LLM call (prompt, output, tokens, latency, cost) is recorded as a generation,
nested under one trace per API request (`observe`). When off, nothing is imported and the
helpers below are no-ops.
"""
import logging
from functools import lru_cache

from app.config import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def enabled() -> bool:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    try:
        from langfuse import Langfuse

        # Registers the client that `observe` and the OpenAI wrapper use.
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return True
    except Exception as e:
        log.warning("Langfuse tracing disabled: %s", e)
        return False


def openai_class():
    """The OpenAI client class: Langfuse's drop-in wrapper when tracing is on."""
    if enabled():
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI
    return OpenAI


def observe(name: str):
    """Decorator: one Langfuse trace/span named `name` around the function."""
    if enabled():
        from langfuse import observe as lf_observe

        return lf_observe(name=name)
    return lambda fn: fn


def flush() -> None:
    if enabled():
        from langfuse import get_client

        get_client().flush()
