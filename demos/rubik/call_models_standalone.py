"""
Standalone: how to call each model's API (GPT-5.5 / Opus 4.8 / Fable 5 /
Gemini / fugu / fugu-ultra). No eval harness -- just the raw API call so
you can verify access and see the returned text.

Env vars needed (export before running):
  OPENAI_API_KEY      # for gpt-5.5  (OpenAI Responses API)
  ANTHROPIC_API_KEY   # for opus 4.8 / fable 5 (Anthropic Messages API)
  GEMINI_API_KEY      # for gemini 3.1 pro
  FUGU_API_KEY        # for fugu / fugu-ultra (OpenAI-compatible)

Run:
  python3 call_models_standalone.py fable5
  python3 call_models_standalone.py fugu_ultra
  python3 call_models_standalone.py              # default: fable5
"""
import os
import sys

API_TIMEOUT = 8000  # reasoning models are slow; do not cut them off

PROMPT = "Write one paragraph explaining what a Rubik's cube solver does."


def call_gpt55():
    import openai
    client = openai.OpenAI(timeout=API_TIMEOUT, max_retries=0)  # OPENAI_API_KEY
    resp = client.responses.create(
        model="gpt-5.5",
        input=PROMPT,
        reasoning={"effort": "high"},
        max_output_tokens=64000,
    )
    return resp.output_text


def call_opus48():
    import anthropic
    client = anthropic.Anthropic(timeout=API_TIMEOUT, max_retries=0)  # ANTHROPIC_API_KEY
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},   # "max" can starve output to 0 chars
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for _ in stream:
            pass
    resp = stream.get_final_message()
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def call_fable5():
    import anthropic
    client = anthropic.Anthropic(timeout=API_TIMEOUT, max_retries=0)  # ANTHROPIC_API_KEY
    with client.messages.stream(
        model="claude-fable-5",
        max_tokens=128000,                  # 64k starves at xhigh; 128k gives room
        thinking={"type": "adaptive"},
        output_config={"effort": "xhigh"},  # high also works; xhigh needs 128k budget
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for _ in stream:
            pass
    resp = stream.get_final_message()
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def call_gemini():
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=API_TIMEOUT * 1000),  # ms
    )
    resp = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=PROMPT,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH
            ),
            max_output_tokens=65536,
        ),
    )
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None) and not getattr(part, "thought", False):
            return part.text
    return resp.text or ""


def call_fugu():
    import openai
    client = openai.OpenAI(
        api_key=os.environ["FUGU_API_KEY"],
        base_url="https://api.sakana.ai/v1",
        timeout=API_TIMEOUT, max_retries=0,
    )
    resp = client.chat.completions.create(
        model="fugu",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=64000,
    )
    return resp.choices[0].message.content


def call_fugu_ultra():
    import openai
    client = openai.OpenAI(
        api_key=os.environ["FUGU_API_KEY"],
        base_url="https://api.sakana.ai/v1",
        timeout=API_TIMEOUT, max_retries=0,
    )
    resp = client.chat.completions.create(
        model="fugu-ultra",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=64000,
        extra_body={"reasoning_effort": "max"},   # fugu-ultra accepts only high|max
        stream=True,
        stream_options={"include_usage": True},
    )
    text = ""
    for chunk in resp:
        if getattr(chunk, "choices", None):
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                text += piece
    return text


MODELS = {
    "gpt55": call_gpt55,
    "opus48": call_opus48,
    "fable5": call_fable5,
    "gemini": call_gemini,
    "fugu": call_fugu,
    "fugu_ultra": call_fugu_ultra,
}

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "fable5"
    if key not in MODELS:
        sys.exit(f"unknown model '{key}'. choose from: {', '.join(MODELS)}")
    print(f"calling {key} ...", flush=True)
    out = MODELS[key]()
    print(f"\n--- {key} returned {len(out or '')} chars ---\n")
    print(out)
