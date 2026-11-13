"""Thin wrapper over any OpenAI-compatible chat endpoint, with JSON output."""
import json
import re
import time

import openai
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.tracing import openai_class

_client = openai_class()(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=2)

SHORT_LIMIT_RETRIES = 6
MAX_WAIT_S = 65.0


def _retry_after(err: openai.RateLimitError) -> float | None:
    """Seconds to wait, from the Retry-After header or the 'try again in 4.26s' message."""
    header = err.response.headers.get("retry-after") if err.response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(err))
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else None


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """One chat call. Per-minute rate limits (free tiers) are waited out; a daily quota or a
    wait longer than MAX_WAIT_S is raised at once, since waiting would only hang the caller."""
    for attempt in range(SHORT_LIMIT_RETRIES + 1):
        try:
            resp = _client.chat.completions.create(
                model=settings.llm_model,
                temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content or ""
        except openai.RateLimitError as e:
            wait = _retry_after(e)
            daily = "per day" in str(e).lower()
            if daily or attempt == SHORT_LIMIT_RETRIES or (wait is not None and wait > MAX_WAIT_S):
                raise
            time.sleep((wait or 5.0 * (attempt + 1)) + 0.5)
    raise AssertionError("unreachable")


def chat_json(system: str, user: str, schema: type[BaseModel], retries: int = 2) -> BaseModel:
    """Ask for JSON matching `schema`; retry with the validation error if it fails."""
    instructions = (
        f"{system}\n\nReturn ONLY a JSON object matching this schema:\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )
    last_error = ""
    for _ in range(retries + 1):
        prompt = user if not last_error else f"{user}\n\nYour previous output was invalid: {last_error}"
        raw = chat(instructions, prompt)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last_error = str(e)[:500]
    raise ValueError(f"LLM did not return valid {schema.__name__}: {last_error}")
