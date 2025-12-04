import os
import time
import requests

# Load API key from environment variable
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_headers():
    """Get headers, checking that API key exists."""
    if OPENROUTER_KEY is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set.\n"
            "Set it with: export OPENROUTER_API_KEY='your-key-here'\n"
            "Or in Python: os.environ['OPENROUTER_API_KEY'] = 'your-key-here'"
        )
    return {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }


REQUEST_TIMEOUT = (10, 60)  # (connect_timeout, read_timeout) in seconds
MAX_RETRIES = 3


def _reasoning_config_for_model(model: str) -> dict:
    """
    Return reasoning config for a given model name.
    """
    effort = "none"
    base = model.split("/")[-1]

    reasoning_models_minimal = (
        "gpt-5",
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
    """
    Call an LLM via OpenRouter API.
    
    Returns:
        tuple: (response_text, latency_seconds)
    """
    headers = _get_headers()  # This will raise if key not set
    
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
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            t1 = time.time()
            time.sleep(0.05)  # Rate limiting

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
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (2 ** (attempt - 1)))
            else:
                break

    raise RuntimeError(f"OpenRouter request failed after {MAX_RETRIES} attempts: {last_err}")
