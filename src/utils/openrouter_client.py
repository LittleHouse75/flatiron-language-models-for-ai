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


def call_openrouter_llm(model: str, prompt: str, max_tokens: int = 128, temperature: float = 0.2):
    """
    Generic OpenRouter chat completion caller.
    Supports OpenAI, Anthropic, Google, Mistral, etc.
    """
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
    }

    t0 = time.time()
    resp = requests.post(BASE_URL, headers=HEADERS, json=payload)
    t1 = time.time()

    print(model_id)
    print(resp.status_code)
    print(resp.text)   # or response.json()

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    latency = t1 - t0

    return text, latency