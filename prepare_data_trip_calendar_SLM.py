#!/usr/bin/env python3
"""
Generate natural-planner predictions and write back to JSON,
but calling a hosted Gemma model via OpenRouter (no local model download).

Prereqs:
  pip install openai python-dotenv absl-py

.env should contain:
  OPENROUTER_API_KEY=hf_... or sk-or-...

Usage:
  python generate_preds_openrouter_gemma.py \
    --data_path=data/trip_planning_new.json \
    --out_path=data/trip_planning_new.with_preds.json \
    --model=google/gemma-2-9b-it
"""

import json
import os
import time
from typing import Dict, Any

from absl import app, flags
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FLAGS = flags.FLAGS

DATA_PATH = flags.DEFINE_string(
    "data_path",
    "data/trip_planning_new.json",
    "Path to the input data file containing examples in json format.",
)
OUT_PATH = flags.DEFINE_string(
    "out_path",
    "data/trip_planning_new.with_preds.json",
    "Path to write the updated json with new model predictions.",
)
MODEL = flags.DEFINE_string(
    "model",
    "google/gemma-2-9b-it",  # example hosted Gemma on OpenRouter
    "Model id on OpenRouter (e.g., google/gemma-2-9b-it).",
)
MAX_OUTPUT_TOKENS = flags.DEFINE_integer(
    "max_output_tokens",
    500,
    "Maximum tokens to generate per example.",
)
TEMPERATURE = flags.DEFINE_float(
    "temperature",
    0.0,
    "Sampling temperature (0 for deterministic-ish evaluation).",
)
SLEEP_BETWEEN_CALLS_S = flags.DEFINE_float(
    "sleep_between_calls_s",
    0.0,
    "Optional fixed sleep between API calls (seconds).",
)

TASK_MARKER = "\n\nTASK:"
SOLUTION_MARKER = "\nSOLUTION:"


def build_system_prompt_from_prompt5shot(prompt_5shot: str) -> str:
    """
    Keep only the instruction + the 5 solved examples.
    Remove the final TASK/SOLUTION stub that corresponds to the current example.
    """
    marker = "\n\nTASK:"
    parts = prompt_5shot.split(marker)

    # If it doesn't look like a multi-task 5-shot prompt, just return it as-is.
    if len(parts) <= 2:
        return prompt_5shot.strip()

    # Keep intro + first 5 tasks; drop the final task stub (current example).
    kept = parts[0] + "".join(marker + p for p in parts[1:-1])
    return kept.strip()


def extract_last_task_as_user_prompt(prompt_5shot: str) -> str:
    """
    From a multi-example 5-shot prompt, extract ONLY the final task text
    (the current example), excluding the trailing 'SOLUTION:' stub.
    """
    if not prompt_5shot:
        return ""

    last_task_idx = prompt_5shot.rfind(TASK_MARKER)
    if last_task_idx == -1:
        # Fallback: no TASK blocks found
        return prompt_5shot.strip()

    tail = prompt_5shot[last_task_idx + len(TASK_MARKER):].lstrip()

    # Cut off the SOLUTION stub for the last task
    sol_idx = tail.rfind(SOLUTION_MARKER)
    if sol_idx != -1:
        tail = tail[:sol_idx].rstrip()

    # Extra safety: sometimes SOLUTION: may appear at very end without newline variations
    if tail.strip().endswith("SOLUTION:"):
        tail = tail.strip()[:-len("SOLUTION:")].rstrip()

    return tail.strip()


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    """
    Calls an OpenAI-compatible Chat Completions API (OpenRouter) with retry.
    """
    last_err: Exception | None = None

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:
            last_err = e
            # Exponential backoff: 1,2,4,8,16,32 (+small jitter)
            sleep_s = (2 ** attempt) + 0.1 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Model call failed after retries: {last_err}") from last_err


def main(_: list[str]) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing OPENROUTER_API_KEY. Put it in your .env or environment variables."
        )

    # OpenRouter is OpenAI-compatible; point SDK to their base URL
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            # OpenRouter recommends these headers (referer can be any URL for local scripts)
            "HTTP-Referer": "http://localhost",
            "X-Title": "natural-planner-eval",
        },
    )

    with open(DATA_PATH.value, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    if not data:
        raise ValueError("Input JSON is empty — no examples found.")

    # Build reusable system prompt from the first example's prompt_5shot
    first_key = next(iter(data.keys()))
    base_prompt_5shot = data[first_key].get("prompt_5shot", "")
    if not base_prompt_5shot:
        raise KeyError(f"First example '{first_key}' is missing 'prompt_5shot'.")

    system_prompt = build_system_prompt_from_prompt5shot(base_prompt_5shot)

    print(f"Using system prompt (from example '{first_key}'): \n{system_prompt}\n")

    # Iterate items, call model, overwrite pred_5shot_pro
    for ex_id, item in data.items():
        p5 = item.get("prompt_5shot", "")
        user_prompt = extract_last_task_as_user_prompt(p5)
        user_prompt += "\nSOLUTION:"

        # Optional fallback if some items don't have prompt_5shot
        if not user_prompt.strip():
            user_prompt = item.get("prompt_0shot", "")

        if not user_prompt.strip():
            print(f"[SKIP] {ex_id}: missing prompt_5shot and prompt_0shot")
            continue

        print(f"\n--- {ex_id} ---")
        print(f"Model: {MODEL.value}")
        print(f"User prompt:\n{user_prompt}\n")

        pred_text = call_llm_with_retry(
            client=client,
            model=MODEL.value,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS.value,
            temperature=TEMPERATURE.value,
        )

        item["pred_5shot_pro"] = pred_text
        item["pred_model"] = MODEL.value

        print(f"[OK] {ex_id}: wrote pred_5shot_pro ({len(pred_text)} chars)")

        if SLEEP_BETWEEN_CALLS_S.value > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_S.value)

    # Write updated file
    with open(OUT_PATH.value, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote updated dataset to: {OUT_PATH.value}")


if __name__ == "__main__":
    app.run(main)
