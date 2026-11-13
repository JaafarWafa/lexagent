"""Rate-limit handling in the LLM wrapper, with a fake client (no network)."""
import httpx
import openai
import pytest

import app.llm as llm


def rate_limit(msg: str, retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://llm"))
    return openai.RateLimitError(msg, response=resp, body=None)


class FakeCompletions:
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def create(self, **kw):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        msg = type("M", (), {"content": "ok"})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


@pytest.fixture
def fake(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def install(*errors):
        comp = FakeCompletions(errors)
        monkeypatch.setattr(llm, "_client", type("C", (), {"chat": type("X", (), {"completions": comp})})())
        return comp, sleeps

    return install


def test_per_minute_limit_is_waited_out(fake):
    comp, sleeps = fake(rate_limit("tokens per minute (TPM): Please try again in 4.26s."))
    assert llm.chat("s", "u") == "ok" and comp.calls == 2
    assert sleeps == [pytest.approx(4.76)]


def test_retry_after_header_wins(fake):
    _, sleeps = fake(rate_limit("TPM", retry_after="2"))
    llm.chat("s", "u")
    assert sleeps == [2.5]


def test_daily_quota_fails_fast(fake):
    comp, sleeps = fake(rate_limit("tokens per day (TPD): Please try again in 7m20.2s."))
    with pytest.raises(openai.RateLimitError):
        llm.chat("s", "u")
    assert comp.calls == 1 and sleeps == []


def test_long_wait_fails_fast(fake):
    _, sleeps = fake(rate_limit("requests per minute: Please try again in 2m10s."))
    with pytest.raises(openai.RateLimitError):
        llm.chat("s", "u")
    assert sleeps == []


def test_gives_up_after_retries(fake):
    errors = [rate_limit("TPM: try again in 1s.") for _ in range(llm.SHORT_LIMIT_RETRIES + 1)]
    comp, _ = fake(*errors)
    with pytest.raises(openai.RateLimitError):
        llm.chat("s", "u")
    assert comp.calls == llm.SHORT_LIMIT_RETRIES + 1
