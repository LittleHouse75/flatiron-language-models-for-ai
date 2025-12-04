import os
import time
import requests

# Load API key from environment variable
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Use a unique prefix that's extremely unlikely to appear in real summaries
# The UUID-like string makes accidental collisions virtually impossible
ERROR_PREFIX = "[__OPENROUTER_ERROR_7f3d2a1b__:"


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
    
    Models that support reasoning get "minimal" effort to reduce latency/cost.
    All other models get "none".
    """
    # Extract just the model name (after the provider prefix)
    base = model.split("/")[-1].lower()  # lowercase for consistent matching

    # Use exact matches or clear prefixes, ordered from most specific to least
    # This avoids substring ambiguity
    reasoning_models = {
        # Exact matches for specific model versions
        "gpt-5-nano",
        "gpt-5-mini", 
        "gpt-5",
        # Add other reasoning-capable models here as needed
        # "o1-preview",
        # "o1-mini",
    }
    
    # Check for exact match first
    if base in reasoning_models:
        return {"effort": "minimal", "exclude": True}
    
    # For models that start with a reasoning prefix but might have version suffixes
    # e.g., "gpt-5-nano-2024-01" would still match
    reasoning_prefixes = ("gpt-5-nano", "gpt-5-mini", "gpt-5")
    for prefix in reasoning_prefixes:
        if base.startswith(prefix):
            return {"effort": "minimal", "exclude": True}
    
    # Default: no reasoning
    return {"effort": "none", "exclude": True}


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
    }

    reasoning_config = _reasoning_config_for_model(model)
    if reasoning_config:
        payload["reasoning"] = reasoning_config

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

            # Try to parse JSON response
            try:
                data = resp.json()
            except Exception as json_err:
                raise RuntimeError(f"Failed to parse JSON response: {json_err}")

            # Check HTTP status
            if resp.status_code != 200:
                error_msg = data.get("error", {}).get("message", resp.text) if data else resp.text
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {error_msg}")

            # Safely extract the response text with proper error handling
            try:
                # Check if 'choices' exists and has at least one item
                if not data:
                    raise ValueError("Empty response data")
                
                choices = data.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise ValueError(f"No choices in response: {data}")
                
                message = choices[0].get("message")
                if not message or not isinstance(message, dict):
                    raise ValueError(f"No message in first choice: {choices[0]}")
                
                text = message.get("content", "")
                
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                
                if text.strip() == "":
                    text = f"{ERROR_PREFIX} EMPTY_RESPONSE]"
                    
            except (KeyError, IndexError, TypeError, ValueError) as extract_err:
                text = f"{ERROR_PREFIX} MALFORMED_RESPONSE: {extract_err}]"

            latency = t1 - t0
            return text, latency

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (2 ** (attempt - 1)))  # Exponential backoff
            else:
                break
        except RuntimeError:
            # Re-raise RuntimeErrors (our own errors) without retry
            raise

    # If we get here, all retries failed
    return f"{ERROR_PREFIX} Request failed after {MAX_RETRIES} attempts: {last_err}]", float('nan')

