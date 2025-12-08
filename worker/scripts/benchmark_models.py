#!/usr/bin/env python3
"""
Benchmark different LLMs for 20 Questions answering accuracy.
Tests which model best matches expected answers based on the game rules.

Rules:
- yes: the secret word fits/matches what they asked
- no: the secret word does NOT fit
- maybe: depends on context, uncertain
- so close: they're describing the core concept/meaning of the word
- N/A: ONLY for complete gibberish (almost never use)
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Set OPENROUTER_API_KEY environment variable")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models to test
MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-8b-instruct:free",
]

Answer = Literal["yes", "no", "maybe", "so close", "N/A"]

@dataclass
class TestCase:
    secret: str
    question: str
    expected: Answer
    note: str = ""


# Test cases based on the game rules and user feedback
TEST_CASES = [
    # === GUIDANCE ===
    TestCase("guidance", "Is it about guiding people?", "so close", "core meaning"),
    TestCase("guidance", "Does it involve direction or advice?", "so close", "core meaning"),
    TestCase("guidance", "Is it a noun?", "yes", "factual"),
    TestCase("guidance", "Is it alive?", "no", "factual"),
    TestCase("guidance", "Is it a word related to humans?", "yes", "humans give/receive guidance"),
    TestCase("guidance", "Is it related to seeking the truth?", "maybe", "guidance can help seek truth"),
    TestCase("guidance", "Is it related to tools?", "no", "guidance is not a tool"),
    TestCase("guidance", "Would 100 people know this word?", "yes", "common word"),

    # === HAPPINESS ===
    TestCase("happiness", "Is it about feeling good?", "so close", "core meaning"),
    TestCase("happiness", "Is it joy?", "so close", "near-synonym"),
    TestCase("happiness", "Is it an emotion?", "yes", "factual"),
    TestCase("happiness", "Is it a verb?", "no", "factual - it's a noun"),
    TestCase("happiness", "Can you touch it?", "no", "abstract concept"),

    # === BRIDGE ===
    TestCase("bridge", "Does it connect things?", "so close", "core function"),
    TestCase("bridge", "Is it a structure?", "yes", "factual"),
    TestCase("bridge", "Is it made of wood?", "maybe", "some are, some aren't"),
    TestCase("bridge", "Is it alive?", "no", "factual"),
    TestCase("bridge", "Can you walk on it?", "yes", "typical use"),

    # === WHISPER ===
    TestCase("whisper", "Is it about speaking quietly?", "so close", "core meaning"),
    TestCase("whisper", "Is it a verb?", "yes", "factual"),
    TestCase("whisper", "Can you do it?", "yes", "it's an action"),
    TestCase("whisper", "Is it loud?", "no", "opposite"),

    # === OCEAN ===
    TestCase("ocean", "Is it a large body of water?", "so close", "core definition"),
    TestCase("ocean", "Is it wet?", "yes", "factual"),
    TestCase("ocean", "Is it bigger than a house?", "yes", "factual"),
    TestCase("ocean", "Can you drink it?", "no", "salt water"),

    # === KNOWLEDGE ===
    TestCase("knowledge", "Is it related to learning?", "so close", "core concept"),
    TestCase("knowledge", "Is it something you can have?", "yes", "you can have knowledge"),
    TestCase("knowledge", "Is it physical?", "no", "abstract"),

    # === MOUNTAIN ===
    TestCase("mountain", "Is it tall?", "yes", "defining characteristic"),
    TestCase("mountain", "Is it a landform?", "so close", "core definition"),
    TestCase("mountain", "Can you climb it?", "yes", "typical activity"),
    TestCase("mountain", "Is it man-made?", "no", "natural"),

    # === FRIENDSHIP ===
    TestCase("friendship", "Is it a relationship between people?", "so close", "core meaning"),
    TestCase("friendship", "Is it an emotion?", "maybe", "it involves emotions but is more"),
    TestCase("friendship", "Can you see it?", "no", "abstract"),
    TestCase("friendship", "Is it positive?", "yes", "generally positive"),

    # === N/A cases (should almost never trigger) ===
    TestCase("guidance", "asdfghjkl?", "N/A", "gibberish"),
    TestCase("bridge", "Purple monkey dishwasher?", "N/A", "nonsense"),

    # === Edge cases ===
    TestCase("oracle", "Is it similar to an oracle?", "yes", "contains the word - should win"),
    TestCase("fire", "Does it start with F?", "yes", "letter question"),
    TestCase("water", "Is the first letter in the second half of the alphabet?", "yes", "W is after M"),
    TestCase("apple", "Is it 5 letters long?", "yes", "length question"),
]


def build_prompt(secret: str, question: str) -> str:
    """Build the same prompt used in the worker."""
    return f"""20 Questions game. Secret word: "{secret}"
Question: "{question}"

Word info: starts with "{secret[0].upper()}", {len(secret)} letters

RULES - READ CAREFULLY:
1. Answer whether "{secret}" relates to what they asked
2. NEVER say N/A for valid questions. N/A is ONLY for gibberish like "asdfjkl?"

ANSWER GUIDE:
- "yes" = the secret word fits their question
- "no" = the secret word does NOT fit
- "maybe" = depends on context
- "so close" = they're describing the core concept of the word
- "N/A" = ONLY for complete gibberish (almost never use this)

CRITICAL: These are ALL valid yes/no questions - answer them:
- "Is it related to X?" → Answer yes/no/maybe based on whether "{secret}" relates to X
- "Is the word about X?" → Answer yes/no/maybe
- "Does it involve X?" → Answer yes/no/maybe
- "Is it a type of X?" → Answer yes/no/maybe

EXAMPLES:
"{secret}" asked "Is it related to knowledge?" → If {secret} relates to knowledge, say "yes" or "so close"
"{secret}" asked "Is it a verb?" → Answer based on whether {secret} is a verb
"{secret}" asked "Is it alive?" → Answer based on whether {secret} is alive

Answer: {{"answer": "yes|no|maybe|so close"}}"""


def query_model(model: str, secret: str, question: str) -> tuple[str, float]:
    """Query a model and return (answer, latency_ms)."""
    prompt = build_prompt(secret, question)

    start = time.time()
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 50,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        latency = (time.time() - start) * 1000

        if not response.ok:
            return f"ERROR: {response.status_code}", latency

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON response
        try:
            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(json_str)
            answer = parsed.get("answer", "").lower().strip()

            # Normalize
            if answer == "yes":
                return "yes", latency
            elif answer == "no":
                return "no", latency
            elif answer == "maybe":
                return "maybe", latency
            elif answer in ("so close", "soclose"):
                return "so close", latency
            else:
                return "N/A", latency
        except json.JSONDecodeError:
            return f"PARSE_ERROR: {content[:50]}", latency

    except Exception as e:
        return f"ERROR: {str(e)[:50]}", (time.time() - start) * 1000


def run_benchmark():
    """Run benchmark across all models and test cases."""
    print(f"Running benchmark with {len(TEST_CASES)} test cases across {len(MODELS)} models\n")
    print("=" * 80)

    # Results: model -> list of (test_case, answer, correct, latency)
    results: dict[str, list] = {m: [] for m in MODELS}

    # Create all tasks
    tasks = []
    for model in MODELS:
        for tc in TEST_CASES:
            tasks.append((model, tc))

    print(f"Running {len(tasks)} queries in parallel...\n")

    # Run in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {
            executor.submit(query_model, model, tc.secret, tc.question): (model, tc)
            for model, tc in tasks
        }

        completed = 0
        for future in as_completed(future_to_task):
            model, tc = future_to_task[future]
            answer, latency = future.result()
            correct = answer == tc.expected
            results[model].append((tc, answer, correct, latency))
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed {completed}/{len(tasks)} queries...")

    print("\n" + "=" * 80)
    print("RESULTS BY MODEL")
    print("=" * 80)

    # Calculate and print results per model
    model_scores = []
    for model in MODELS:
        model_results = results[model]
        correct_count = sum(1 for _, _, correct, _ in model_results if correct)
        total = len(model_results)
        accuracy = correct_count / total * 100 if total > 0 else 0
        avg_latency = sum(lat for _, _, _, lat in model_results) / total if total > 0 else 0

        model_scores.append((model, accuracy, avg_latency, correct_count, total))

    # Sort by accuracy descending
    model_scores.sort(key=lambda x: (-x[1], x[2]))

    for model, accuracy, avg_latency, correct, total in model_scores:
        print(f"\n{model}")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"  Avg latency: {avg_latency:.0f}ms")

    # Print detailed results for best and worst models
    best_model = model_scores[0][0]
    worst_model = model_scores[-1][0]

    print("\n" + "=" * 80)
    print(f"DETAILED RESULTS - BEST MODEL: {best_model}")
    print("=" * 80)
    print_detailed_results(results[best_model])

    if best_model != worst_model:
        print("\n" + "=" * 80)
        print(f"DETAILED RESULTS - WORST MODEL: {worst_model}")
        print("=" * 80)
        print_detailed_results(results[worst_model])

    # Print mismatches for all models
    print("\n" + "=" * 80)
    print("ALL MISMATCHES BY MODEL")
    print("=" * 80)
    for model in MODELS:
        mismatches = [(tc, ans) for tc, ans, correct, _ in results[model] if not correct]
        if mismatches:
            print(f"\n{model} ({len(mismatches)} mismatches):")
            for tc, ans in mismatches:
                print(f"  [{tc.secret}] \"{tc.question}\"")
                print(f"    Expected: {tc.expected}, Got: {ans} ({tc.note})")


def print_detailed_results(model_results: list):
    """Print detailed results for a model."""
    for tc, answer, correct, latency in model_results:
        status = "✓" if correct else "✗"
        print(f"  {status} [{tc.secret}] \"{tc.question}\"")
        if not correct:
            print(f"      Expected: {tc.expected}, Got: {answer}")


if __name__ == "__main__":
    run_benchmark()
