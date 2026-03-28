"""Shared API helper — Anthropic client construction and rate-limit retry."""

import time

import anthropic  # type: ignore

from improvement_loop.loop_config import get_config
from improvement_loop.project_config import get_project_config


def get_client() -> anthropic.Anthropic:
    """Return an Anthropic client, using the config API key if set.

    Resolution order: project_config.yaml → loop config JSON → ANTHROPIC_API_KEY env var.
    """
    project_cfg = get_project_config()
    loop_cfg = get_config()
    api_key = project_cfg.anthropic_api_key or loop_cfg.anthropic_api_key
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)


def api_call_with_retry(create_kwargs: dict) -> str:
    """Call client.messages.create with rate-limit retry.  Returns response text.

    Uses streaming to avoid the SDK's 10-minute timeout guard for large
    max_tokens values.
    """
    cfg = get_config()
    client = get_client()
    for attempt in range(1, cfg.max_api_retries + 1):
        try:
            collected: list[str] = []
            with client.messages.stream(**create_kwargs) as stream:
                for text in stream.text_stream:
                    collected.append(text)
            return "".join(collected)
        except anthropic.RateLimitError:
            delay = cfg.retry_base_delay * attempt
            print(f"    Rate limited (attempt {attempt}/{cfg.max_api_retries}), "
                  f"waiting {delay:.0f}s...")
            time.sleep(delay)
            if attempt == cfg.max_api_retries:
                raise
        except anthropic.APIError:
            raise
    return ""  # unreachable
