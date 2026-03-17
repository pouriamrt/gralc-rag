"""OpenAI-based answer generation for GraLC-RAG.

Uses the OpenAI Python SDK to generate answers grounded in retrieved context
passages, with retry logic for transient API failures.
"""

from __future__ import annotations

import logging
import re
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the question based ONLY "
    "on the provided context. If the context doesn't contain enough information, "
    "say so. For yes/no questions, start your answer with Yes, No, or Maybe, "
    "then explain."
)

_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds


def _build_user_prompt(question: str, contexts: list[str]) -> str:
    """Format numbered context passages and the question into a user prompt."""
    parts: list[str] = ["Context:"]
    for idx, ctx in enumerate(contexts, 1):
        parts.append(f"[{idx}] {ctx}")
    parts.append("")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


def _extract_decision(answer: str) -> str:
    """Extract yes / no / maybe from the first word of the answer.

    Returns the normalised decision string, or ``"unknown"`` if no clear
    decision word is found.
    """
    first_word = answer.strip().split()[0].lower().rstrip(".,;:!") if answer.strip() else ""
    if re.match(r"^yes\b", first_word):
        return "yes"
    if re.match(r"^no\b", first_word):
        return "no"
    if re.match(r"^maybe\b", first_word):
        return "maybe"
    return "unknown"


def generate_answer(
    question: str,
    contexts: list[str],
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Generate an answer using the OpenAI Chat Completions API.

    Parameters:
        question: The user question.
        contexts: Retrieved context passages to ground the answer.
        api_key: OpenAI API key.
        model: Model identifier (default ``gpt-4o-mini``).

    Returns:
        Dictionary with keys:
            - ``full_answer`` (str): The complete generated answer.
            - ``decision`` (str): Extracted yes/no/maybe/unknown.
            - ``model`` (str): Model used.
            - ``usage_tokens`` (int): Total tokens consumed (prompt + completion).

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    client = OpenAI(api_key=api_key)
    user_prompt = _build_user_prompt(question, contexts)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )

            answer_text = response.choices[0].message.content or ""
            usage = response.usage
            total_tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

            return {
                "full_answer": answer_text.strip(),
                "decision": _extract_decision(answer_text),
                "model": model,
                "usage_tokens": total_tokens,
            }

        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            wait = _INITIAL_BACKOFF * (2 ** attempt)
            logger.warning(
                "OpenAI API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

        except Exception as exc:
            # Non-retryable errors bubble up immediately.
            logger.error("OpenAI API non-retryable error: %s", exc)
            raise

    raise RuntimeError(
        f"OpenAI API call failed after {_MAX_RETRIES} retries. "
        f"Last error: {last_error}"
    )
