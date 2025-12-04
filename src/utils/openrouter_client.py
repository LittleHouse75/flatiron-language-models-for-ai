import os
import time
import requests

# OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}

REQUEST_TIMEOUT = (10, 60)  # (connect_timeout, read_timeout) in seconds
MAX_RETRIES = 3              # how many times to retry on timeout


def _reasoning_config_for_model(model: str) -> dict:
    """
    Return reasoning config for a given model name.

    - Default: no reasoning ("effort": "none")
    - Any GPT-5 family model: minimal reasoning
    """
    effort = "none"

    # Strip provider prefix like "openai/gpt-5-mini"
    base = model.split("/")[-1]

    reasoning_models_minimal = (
        "gpt-5",       # catches gpt-5, gpt-5-large, etc.
        "gpt-5-nano",
        "gpt-5-mini",
    )

    if any(name in base for name in reasoning_models_minimal):
        effort = "minimal"

    return {
        "effort": effort,
        "exclude": True,
    }


def call_openrouter_llm(
    model: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.2,
):
    if OPENROUTER_KEY is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found in environment. Set it before running."
        )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You summarize chat conversations accurately and concisely."},
            {"role": "user", "content": prompt},
        ],
        "reasoning": _reasoning_config_for_model(model),
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = requests.post(
                BASE_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,  # <- key line
            )
            t1 = time.time()
            time.sleep(0.05)  # ~20 requests/sec ceiling

            try:
                data = resp.json()
            except Exception:
                data = None

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

            text = data["choices"][0]["message"].get("content", "")
            if not isinstance(text, str) or text.strip() == "":
                text = "[EMPTY_RESPONSE]"

            latency = t1 - t0
            return text, latency

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            # exponential-ish backoff: 0.5s, 1s, 2s...
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (2 ** (attempt - 1)))
            else:
                break

    # If we get here, all retries failed
    raise RuntimeError(f"OpenRouter request failed after {MAX_RETRIES} attempts: {last_err}")