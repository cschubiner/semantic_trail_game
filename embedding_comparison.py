#!/usr/bin/env python3
"""
Compare word embeddings and LLM rankings across multiple OpenRouter models.
Useful for building Semantic Trail–style word guessing games.

Now includes:
- Embedding models (cosine similarity)
- LLM models with structured outputs (asking models to rank words)
"""

import os
import json
import time
from pathlib import Path

import requests
import numpy as np


# Available embedding models on OpenRouter
EMBEDDING_MODELS = [
    # OpenAI
    "openai/text-embedding-3-small",   # $0.02/M, 8K context
    "openai/text-embedding-3-large",   # $0.13/M, 8K context
    # Google
    "google/gemini-embedding-001",     # $0.15/M, 20K context, top MTEB benchmark
    # Mistral
    "mistralai/mistral-embed-2312",    # $0.10/M, 8K context, 1024-dim
    # Qwen
    "qwen/qwen3-embedding-0.6b",       # ~$0.01/M, 32K context
    "qwen/qwen3-embedding-8b",         # $0.01/M, 32K context, #1 MTEB multilingual
    # Open-source models
    "thenlper/gte-large",
    "thenlper/gte-base",
    "intfloat/e5-large-v2",
    "baai/bge-base-en-v1.5",
]

# LLM models for ranking via structured outputs
# These models are asked to rank words by semantic similarity
LLM_RANKING_MODELS = [
    ("openai/gpt-4o-mini", 0.15),           # $0.15/M input, supports structured outputs
    ("google/gemini-2.0-flash-001", 0.10),  # $0.10/M input, fast
    ("google/gemini-2.5-flash-lite", 0.02), # Gemini 2.5 Flash Lite
    ("google/gemini-2.5-flash", 0.15),      # Gemini 2.5 Flash
    ("deepseek/deepseek-v3.2", 0.14),       # DeepSeek V3.2
    ("openai/gpt-oss-120b", 0.50),          # GPT OSS 120B
    ("openai/gpt-oss-20b", 0.10),           # GPT OSS 20B
    ("google/gemini-3-pro-preview", 1.25),  # Gemini 3 Pro Preview
]


# =============================================================================
# GROUND TRUTH SCORES - Human-defined expected semantic similarity
# =============================================================================
# These scores represent my (Claude's) judgment of how semantically similar
# each word is to the target word, on a 0-100 scale.

EXPECTED_SCORES = {
    "fire": {
        # Direct synonyms / very close
        "flame": 95,          # Direct synonym/component
        "blaze": 90,          # Synonym for large fire
        "burn": 82,           # What fire does
        # Related concepts
        "heat": 70,           # Property of fire
        "smoke": 62,          # Byproduct of fire
        "ember": 58,          # Remnant/part of fire
        "ash": 50,            # Result of fire
        # Loosely related
        "warm": 38,           # Effect of fire, but adjective
        # Opposite
        "water": 15,          # Extinguishes fire (some relation as opposite)
        # Unrelated
        "pencil": 2,          # Completely unrelated
    },
    "book": {
        # Direct synonyms / very close
        "novel": 92,          # Type of book
        "text": 82,           # Synonym in some contexts
        "volume": 78,         # Synonym
        # Related concepts
        "read": 68,           # What you do with books
        "page": 65,           # Component of book
        "author": 55,         # Who writes books
        "library": 52,        # Where books are kept
        "story": 48,          # What books contain
        # Loosely related
        "magazine": 32,       # Similar format, different medium
        # Unrelated
        "hammer": 2,          # Completely unrelated
    },
    "doctor": {
        # Direct synonyms / very close
        "physician": 95,      # Direct synonym
        "surgeon": 82,        # Specific type of doctor
        "medical": 78,        # Adjective for doctor's field
        # Related concepts
        "medicine": 68,       # What doctors practice/prescribe
        "nurse": 58,          # Healthcare worker, different role
        "hospital": 52,       # Where doctors work
        "patient": 48,        # Who doctors treat
        "stethoscope": 42,    # Tool doctors use
        # Loosely related
        "healthy": 28,        # Goal of doctor's work, but adjective
        # Unrelated
        "skateboard": 2,      # Completely unrelated
    },
    "happy": {
        # Direct synonyms
        "joyful": 95,         # Direct synonym
        "cheerful": 92,       # Very close synonym
        "glad": 88,           # Synonym
        # Related positive emotions
        "pleased": 75,        # Similar but milder
        "content": 68,        # Related but calmer state
        "excited": 60,        # Positive but different energy
        # Associated concepts
        "smile": 52,          # Expression of happiness
        "celebrate": 45,      # Action from happiness
        # Opposite / category
        "sad": 18,            # Direct antonym (some similarity as emotions)
        # Unrelated
        "concrete": 2,        # Completely unrelated
    },
    "ocean": {
        # Direct synonyms / very close
        "sea": 95,            # Nearly identical meaning
        "marine": 85,         # Ocean-related, adjective form
        "waves": 78,          # Defining feature of ocean
        # Related concepts
        "water": 70,          # What ocean is made of, but water is broader
        "tide": 72,           # Ocean-specific phenomenon
        "beach": 50,          # Borders ocean, but is land
        # Loosely related
        "fish": 40,           # Lives in ocean, but is an animal
        "ship": 35,           # Travels on ocean, but is a vehicle
        # Opposites / unrelated
        "desert": 12,         # Opposite environment (dry vs wet)
        "calculator": 2,      # Completely unrelated
    },
    "ocean2": {
        # Very related - ocean activities
        "sail": 90,           # Sailing is done on the ocean
        "cruise": 88,         # Ocean cruises, very strong association
        "embark": 72,         # Embarking on ocean voyages
        # Temperature words - ocean association
        "cold": 55,           # Ocean water is often cold
        "freezing": 48,       # Arctic/cold ocean waters
        "rinse": 40,          # Involves water, but not ocean-specific
        "sounds": 32,         # Ocean sounds, "sound" = body of water
        "warm": 22,           # Tropical oceans, but generic adjective
        "hot": 12,            # Less ocean-related than cold
        "tepid": 10,          # Temperature adjective, very weak ocean relation
        # Unrelated words
        "sale": 6,            # Commerce, unrelated (sounds like "sail" though!)
        "spider": 5,          # Unrelated (sea spiders exist but obscure)
        "mouse": 5,           # Unrelated animal/computer device
        "kindle": 4,          # E-reader or fire, unrelated
        "chocolate": 3,       # Food, unrelated
        "score": 3,           # Unrelated
        "airpods": 2,         # Tech product, completely unrelated
    },
    "king": {
        # Direct synonyms / very close
        "queen": 90,          # Royal counterpart, same domain
        "monarch": 95,        # Direct synonym
        "royal": 82,          # Adjective describing king
        # Related concepts
        "throne": 70,         # Where king sits, strong association
        "crown": 72,          # King's symbol
        "kingdom": 65,        # What king rules
        "prince": 58,         # Royal family but lower rank
        # Loosely related
        "castle": 45,         # Where king lives, but is a building
        "chess": 30,          # Has king piece, but it's a game
        # Opposites / unrelated
        "peasant": 20,        # Opposite end of social hierarchy
        "refrigerator": 2,    # Completely unrelated
    },
    "void": {
        # Direct synonyms / very close meanings
        "null": 95,           # Programming: void and null are nearly interchangeable concepts
        "empty": 90,          # Core meaning of void = empty
        "nothing": 88,        # Void means nothingness
        "vacuum": 85,         # Physical void = vacuum
        "abyss": 82,          # Void often refers to an abyss
        "hollow": 78,         # Hollow = having a void inside
        "blank": 75,          # Blank space, empty void
        "nil": 90,            # Programming synonym for void/null
        "none": 85,           # Absence, like void
        # Related but slightly different
        "chasm": 70,          # A void-like opening, but more specific
        "darkness": 65,       # Voids are often dark, but darkness isn't void
        "vacant": 72,         # Empty, unoccupied
        "barren": 60,         # Empty of life, but not void itself
        "expanse": 50,        # Can be void-like but also means large area
        "space": 65,          # Void of space, but space has other meanings
        # Legal/action sense (to void = to cancel)
        "cancel": 55,         # To void a contract = cancel it
        "invalid": 60,        # Void = invalid
        "nullify": 70,        # To make void
        "annul": 65,          # To void/cancel
        "revoke": 50,         # Related to voiding but less direct
        # Programming adjacent
        "undefined": 75,      # Similar to void in programming
        # Somewhat related
        "gap": 45,            # A void is a gap, but gap is more general
        "hole": 50,           # Physical void, but hole is more concrete
        "pit": 40,            # A hole, less abstract than void
        "depth": 35,          # Voids have depth, but weak connection
        # Unrelated words
        "banana": 2,
        "keyboard": 5,
        "happy": 3,
        "running": 2,
        "bicycle": 1,
        "sandwich": 2,
        "purple": 3,
        "whisper": 5,
        "mountain": 8,
        "telephone": 3,
    },
    "skirt": {
        # Direct synonyms / very close (clothing sense)
        "miniskirt": 98,      # A type of skirt - closest possible
        "dress": 75,          # Similar garment, often worn together
        "garment": 70,        # Skirt is a garment
        "hem": 72,            # Skirts have hems, strongly associated
        "pleated": 65,        # Describes skirts often
        "fabric": 55,         # Skirts are made of fabric
        # Related clothing
        "blouse": 60,         # Often paired with skirts
        "pants": 55,          # Alternative lower-body garment
        "shirt": 45,          # Clothing but different body area
        "jacket": 40,         # Clothing but less related
        "clothing": 65,       # Skirt is clothing
        "wardrobe": 50,       # Contains skirts
        "outfit": 55,         # Skirts are part of outfits
        "fashion": 50,        # Skirts are fashion items
        # Verb sense (to skirt = to go around/avoid)
        "avoid": 40,          # To skirt an issue = avoid it
        "circumvent": 45,     # To skirt around = circumvent
        "bypass": 42,         # Similar to skirting
        "evade": 38,          # To skirt = evade
        "sidestep": 40,       # To skirt = sidestep
        # Edge/border sense (skirt of a forest)
        "edge": 35,           # Skirt can mean edge
        "border": 35,         # Skirt of something = border
        "fringe": 40,         # Edge, also fashion term
        "margin": 30,         # Edge meaning
        "perimeter": 32,      # Outer edge
        "periphery": 33,      # Outer area
        # Unrelated words
        "elephant": 2,
        "thunder": 3,
        "calculate": 1,
        "volcano": 2,
        "algorithm": 1,
        "breakfast": 3,
        "gravity": 2,
        "saxophone": 3,
        "microscope": 2,
        "democracy": 1,
    },
}

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_DIR = Path(__file__).parent / ".embedding_cache"
LLM_CACHE_DIR = Path(__file__).parent / ".llm_ranking_cache"


def get_cache_path(word: str, model: str) -> Path:
    """Get the cache file path for a word/model combination."""
    # Sanitize model name for filesystem
    model_safe = model.replace("/", "_")
    return CACHE_DIR / model_safe / f"{word}.json"


def load_from_cache(word: str, model: str) -> list[float] | None:
    """Load embedding from cache if it exists."""
    cache_path = get_cache_path(word, model)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    return None


def save_to_cache(word: str, model: str, embedding: list[float]):
    """Save embedding to cache."""
    cache_path = get_cache_path(word, model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(embedding, f)


# =============================================================================
# LLM Ranking Functions - Using structured outputs to ask LLMs to rank words
# =============================================================================

def get_llm_cache_path(target_word: str, words: tuple[str, ...], model: str) -> Path:
    """Get cache file path for an LLM ranking result."""
    model_safe = model.replace("/", "_")
    # Create a hash of the words list for the filename
    words_hash = hash(words) & 0xFFFFFFFF  # Positive 32-bit hash
    return LLM_CACHE_DIR / model_safe / f"{target_word}_{words_hash:08x}.json"


def load_llm_ranking_from_cache(
    target_word: str,
    words: list[str],
    model: str,
) -> list[str] | None:
    """Load LLM ranking from cache if it exists."""
    cache_path = get_llm_cache_path(target_word, tuple(sorted(words)), model)
    if cache_path.exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
            # Verify the words match
            if set(data.get("words", [])) == set(words):
                return data.get("ranking", [])
    return None


def save_llm_ranking_to_cache(
    target_word: str,
    words: list[str],
    model: str,
    ranking: list[str],
):
    """Save LLM ranking to cache."""
    cache_path = get_llm_cache_path(target_word, tuple(sorted(words)), model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "target_word": target_word,
            "words": words,
            "ranking": ranking,
            "model": model,
        }, f, indent=2)


def get_llm_word_ranking(
    words: list[str],
    target_word: str,
    model: str,
    api_key: str,
    use_structured_output: bool = True,
) -> list[str]:
    """
    Ask an LLM to rank words by semantic similarity to a target word.

    Args:
        words: List of words to rank
        target_word: The target word to compare against
        model: OpenRouter model ID (e.g., "openai/gpt-4o-mini")
        api_key: OpenRouter API key
        use_structured_output: Whether to use JSON schema structured output

    Returns:
        List of words ordered from most to least similar
    """
    # Check cache first
    cached = load_llm_ranking_from_cache(target_word, words, model)
    if cached is not None:
        return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Shuffle words to avoid position bias
    import random
    shuffled_words = words.copy()
    random.shuffle(shuffled_words)
    words_list = ", ".join(shuffled_words)

    prompt = f"""Rank the following words by their semantic similarity to the target word "{target_word}".

Words to rank: {words_list}

Order them from MOST similar to LEAST similar to "{target_word}".
Consider meaning, context, associations, and conceptual relationships.
Return ONLY the ranked list of words, nothing else."""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a semantic similarity expert. Rank words by how semantically related they are to a target word. Be precise and consistent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,  # Deterministic output
    }

    if use_structured_output:
        # Use JSON schema for structured output
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "word_ranking",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_words": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Words ordered from most to least similar to the target"
                        }
                    },
                    "required": ["ranked_words"],
                    "additionalProperties": False
                }
            }
        }
        # Update prompt to request JSON
        payload["messages"][1]["content"] = f"""Rank the following words by their semantic similarity to the target word "{target_word}".

Words to rank: {words_list}

Order them from MOST similar to LEAST similar to "{target_word}".
Consider meaning, context, associations, and conceptual relationships.

Return the ranking as JSON with a "ranked_words" array."""

    try:
        response = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        if use_structured_output:
            # Parse JSON response
            result = json.loads(content)
            ranking = result.get("ranked_words", [])
        else:
            # Parse plain text response - extract words
            ranking = []
            for line in content.strip().split("\n"):
                # Try to extract word from numbered list or comma-separated
                line = line.strip()
                if not line:
                    continue
                # Remove numbering like "1. " or "1) "
                import re
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                # Extract word (first word-like token)
                word_match = re.match(r"([a-zA-Z]+)", line)
                if word_match:
                    word = word_match.group(1).lower()
                    if word in words and word not in ranking:
                        ranking.append(word)

            # If comma-separated, try that
            if len(ranking) < len(words) // 2:
                ranking = []
                for word in re.findall(r"[a-zA-Z]+", content.lower()):
                    if word in words and word not in ranking:
                        ranking.append(word)

        # Ensure all words are included (add missing ones at the end)
        for word in words:
            if word not in ranking:
                ranking.append(word)

        # Cache the result
        save_llm_ranking_to_cache(target_word, words, model, ranking)

        return ranking

    except Exception as e:
        print(f"    LLM ranking error for {model}: {e}")
        # Return words in original order on error
        return words


def llm_ranking_to_similarities(
    ranking: list[str],
    words: list[str],
) -> dict[str, float]:
    """
    Convert an LLM ranking to pseudo-similarity scores.

    The highest ranked word gets score 1.0, lowest gets score close to 0.
    This allows comparison with embedding-based similarities.

    Args:
        ranking: Words ordered from most to least similar
        words: Original list of words

    Returns:
        Dictionary mapping each word to a similarity score (0-1)
    """
    n = len(ranking)
    if n == 0:
        return {word: 0.5 for word in words}

    similarities = {}
    for i, word in enumerate(ranking):
        # Linear scale from 1.0 (rank 0) to ~0.1 (last rank)
        similarities[word] = 1.0 - (i / n) * 0.9

    # Add any missing words with low score
    for word in words:
        if word not in similarities:
            similarities[word] = 0.05

    return similarities


def compare_llm_rankings(
    words: list[str],
    target_word: str,
    llm_models: list[tuple[str, float]],
    api_key: str,
) -> dict[str, dict[str, float]]:
    """
    Get word rankings from multiple LLM models.

    Args:
        words: List of words to rank
        target_word: The target word to compare against
        llm_models: List of (model_id, price) tuples
        api_key: OpenRouter API key

    Returns:
        {model_id: {word: similarity_score}}
    """
    results = {}

    for model_id, price in llm_models:
        model_short = model_id.split("/")[-1]

        # Check if cached
        cached = load_llm_ranking_from_cache(target_word, words, model_id)
        if cached:
            print(f"  {model_short} (LLM): cached")
        else:
            print(f"  {model_short} (LLM): calling API...")

        try:
            ranking = get_llm_word_ranking(words, target_word, model_id, api_key)
            similarities = llm_ranking_to_similarities(ranking, words)
            results[model_id] = similarities

            # Small delay to avoid rate limits
            if not cached:
                time.sleep(0.5)

        except Exception as e:
            print(f"    Error: {e}")
            results[model_id] = {}

    return results


def get_embeddings(
    words: list[str],
    model: str,
    api_key: str,
) -> dict[str, list[float]]:
    """
    Get embeddings for a list of words using a specific model.
    Uses local cache to avoid redundant API calls.

    Args:
        words: List of words to embed
        model: OpenRouter model ID
        api_key: OpenRouter API key

    Returns:
        Dictionary mapping each word to its embedding vector
    """
    embeddings = {}
    words_to_fetch = []

    # Check cache first
    for word in words:
        cached = load_from_cache(word, model)
        if cached is not None:
            embeddings[word] = cached
        else:
            words_to_fetch.append(word)

    # Fetch uncached words from API
    if words_to_fetch:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": words_to_fetch,
        }

        response = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        # Map words to their embeddings and cache them
        for i, word in enumerate(words_to_fetch):
            embedding = data["data"][i]["embedding"]
            embeddings[word] = embedding
            save_to_cache(word, model, embedding)

    return embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare_word_similarities(
    words: list[str],
    target_word: str,
    model: str,
    api_key: str,
) -> dict[str, float]:
    """
    Compare similarity of each word to a target word.

    Args:
        words: List of words to compare
        target_word: The word to compare against (like the secret word in Semantic Trail)
        model: OpenRouter model ID
        api_key: OpenRouter API key

    Returns:
        Dictionary mapping each word to its similarity score with target
    """
    all_words = list(set(words + [target_word]))
    embeddings = get_embeddings(all_words, model, api_key)

    target_embedding = embeddings[target_word]

    similarities = {}
    for word in words:
        if word != target_word:
            sim = cosine_similarity(embeddings[word], target_embedding)
            similarities[word] = sim

    return similarities


def count_cached(words: list[str], model: str) -> int:
    """Count how many words are already cached for a model."""
    return sum(1 for word in words if load_from_cache(word, model) is not None)


def compare_across_models(
    words: list[str],
    target_word: str,
    models: list[str],
    api_key: str,
) -> dict[str, dict[str, float]]:
    """
    Compare word similarities across multiple embedding models.

    Args:
        words: List of words to compare
        target_word: The target/secret word
        models: List of model IDs to use
        api_key: OpenRouter API key

    Returns:
        Nested dict: {model: {word: similarity}}
    """
    results = {}
    all_words = list(set(words + [target_word]))

    for model in models:
        cached = count_cached(all_words, model)
        to_fetch = len(all_words) - cached
        model_short = model.split("/")[-1]

        if to_fetch == 0:
            print(f"  {model_short}: {cached} cached (no API calls)")
        else:
            print(f"  {model_short}: {cached} cached, fetching {to_fetch}...")

        try:
            similarities = compare_word_similarities(words, target_word, model, api_key)
            results[model] = similarities
        except Exception as e:
            print(f"    Error: {e}")
            results[model] = {}

    return results


def compute_correlation(
    similarities: dict[str, float],
    expected_scores: dict[str, int],
) -> float:
    """
    Compute Spearman rank correlation between model's similarity rankings
    and expected (human-defined) scores.

    Args:
        similarities: Model's cosine similarities {word: similarity}
        expected_scores: Human-defined expected scores {word: score}

    Returns:
        Spearman correlation coefficient (-1 to 1, higher = better match)
    """
    from scipy.stats import spearmanr

    # Get common words
    common_words = set(similarities.keys()) & set(expected_scores.keys())
    if len(common_words) < 2:
        return 0.0

    model_sims = []
    expected_vals = []

    for word in common_words:
        model_sims.append(similarities[word])
        expected_vals.append(expected_scores[word])

    correlation, _ = spearmanr(model_sims, expected_vals)
    return correlation


def print_expected_comparison(
    results: dict[str, dict[str, float]],
    target_word: str,
    expected: dict[str, int],
    model_prices: dict[str, float] = None,
) -> dict[str, float]:
    """
    Print comparison of model rankings vs expected scores with correlation metrics.

    Returns:
        Dictionary mapping model_id to correlation score for aggregation.
    """
    print(f"\n{'='*80}")
    print(f"MODEL ACCURACY vs EXPECTED SCORES for '{target_word}'")
    print(f"{'='*80}\n")

    # Print expected scores first (sorted)
    print("EXPECTED (Human-defined) scores:")
    sorted_expected = sorted(expected.items(), key=lambda x: x[1], reverse=True)
    for i, (word, score) in enumerate(sorted_expected, 1):
        print(f"  {i:>2}. {word:<15} {score:>3}")
    print()

    # Compute and display correlation for each model
    print(f"{'Model':<30} {'Spearman r':>12} {'Price':>12}")
    print("-" * 56)

    model_correlations = {}
    model_metrics = []

    for model, similarities in results.items():
        if not similarities:
            continue
        try:
            corr = compute_correlation(similarities, expected)
        except ImportError:
            corr = 0.0

        model_correlations[model] = corr
        price = model_prices.get(model, 0) if model_prices else 0
        model_short = model.split("/")[-1][:28]
        model_metrics.append((model_short, corr, price))

    # Sort by correlation (best first)
    model_metrics.sort(key=lambda x: x[1], reverse=True)

    for model_short, corr, price in model_metrics:
        print(f"{model_short:<30} {corr:>12.3f} ${price:>10.2f}/M")

    print()
    print("Spearman r: 1.0 = perfect rank match with expected, 0 = no correlation")
    print()

    return model_correlations


def print_ranked_results(
    results: dict[str, dict[str, float]],
    target_word: str,
    expected: dict[str, int],
    model_prices: dict[str, float] = None,
):
    """Print words ranked by similarity for each model in side-by-side columns."""
    print(f"\n{'='*130}")
    print(f"Words ranked by similarity to target: '{target_word}'")
    print(f"{'='*130}\n")

    models = list(results.keys())
    if not models:
        return

    # Get ranked lists for each model
    ranked_by_model = {}
    for model, similarities in results.items():
        if similarities:
            ranked_words = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            ranked_by_model[model] = [word for word, _ in ranked_words]
        else:
            ranked_by_model[model] = []

    # Get model short names - truncate long names
    model_names = [m.split("/")[-1][:16] for m in models]
    col_width = 18

    # Print header with "Expected" first, then model names
    header = f"{'Rank':<6}{'EXPECTED':<{col_width}}"
    for name in model_names:
        header += f"{name:<{col_width}}"
    print(header)

    # Print price row
    if model_prices:
        price_row = f"{'Cost':<6}{'':<{col_width}}"
        for model in models:
            price = model_prices.get(model, 0)
            price_str = f"${price:.3f}/M"
            price_row += f"{price_str:<{col_width}}"
        print(price_row)

    print("-" * len(header))

    # Get expected ranking
    sorted_expected = sorted(expected.items(), key=lambda x: x[1], reverse=True)
    expected_words = [word for word, _ in sorted_expected]

    # Print rows
    max_words = max(
        len(expected_words),
        max((len(words) for words in ranked_by_model.values()), default=0)
    )

    for i in range(max_words):
        row = f"{i+1:<6}"
        # Expected column
        exp_word = expected_words[i] if i < len(expected_words) else ""
        exp_score = expected.get(exp_word, 0) if exp_word else 0
        row += f"{exp_word} ({exp_score})"[:col_width-1].ljust(col_width)

        # Model columns
        for model in models:
            words = ranked_by_model.get(model, [])
            word = words[i] if i < len(words) else ""
            row += f"{word:<{col_width}}"
        print(row)

    print()


def create_ensemble_similarities(
    results: dict[str, dict[str, float]],
    top_n: int = 3,
    model_correlations: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Create ensemble similarities by averaging similarity scores from top N models.

    Args:
        results: {model: {word: similarity}} from all models
        top_n: Number of top models to include in ensemble
        model_correlations: {model: correlation} to determine top models

    Returns:
        {word: averaged_similarity} ensemble similarities
    """
    if model_correlations:
        # Use top N models by correlation
        sorted_models = sorted(model_correlations.items(), key=lambda x: x[1], reverse=True)
        top_models = [m for m, _ in sorted_models[:top_n]]
    else:
        # Just use first N models
        top_models = list(results.keys())[:top_n]

    # Get all words
    all_words = set()
    for model in top_models:
        if model in results:
            all_words.update(results[model].keys())

    # Average similarities across top models
    ensemble_sims = {}
    for word in all_words:
        sims = []
        for model in top_models:
            if model in results and word in results[model]:
                sims.append(results[model][word])
        if sims:
            ensemble_sims[word] = sum(sims) / len(sims)

    return ensemble_sims


def run_ensemble_evaluation(
    all_results: dict[str, dict[str, dict[str, float]]],
    all_correlations: list[dict[str, float]],
    expected_scores: dict[str, dict[str, int]],
    top_n_options: list[int] = [2, 3, 5],
    allowed_models: set[str] | None = None,
    label: str = "",
) -> dict[str, float]:
    """
    Evaluate ensemble approaches by combining top models.
    This can be scoped to a subset of models (e.g., embeddings-only or LLM-only).

    Args:
        all_results: {target_word: {model: {word: similarity}}}
        all_correlations: List of {model: correlation} for each test
        expected_scores: {target_word: {word: expected_score}}
        top_n_options: Different ensemble sizes to try

    Returns:
        {ensemble_name: average_correlation}
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE EVALUATION - Combining top models")
    print("=" * 80 + "\n")

    # Aggregate correlations to find overall top models
    model_totals = {}
    model_counts = {}
    for correlations in all_correlations:
        for model, corr in correlations.items():
            if allowed_models and model not in allowed_models:
                continue
            if model not in model_totals:
                model_totals[model] = 0.0
                model_counts[model] = 0
            model_totals[model] += corr
            model_counts[model] += 1

    overall_rankings = {
        model: model_totals[model] / model_counts[model]
        for model in model_totals
    }

    if not overall_rankings:
        print("No models available for ensemble evaluation with the given filter.")
        return {}

    # Show which models are being combined
    sorted_models = sorted(overall_rankings.items(), key=lambda x: x[1], reverse=True)
    print("Model rankings (used for ensemble selection):")
    for i, (model, corr) in enumerate(sorted_models, 1):
        model_short = model.split("/")[-1][:25]
        print(f"  {i}. {model_short:<27} {corr:.3f}")
    print()

    ensemble_results = {}

    for top_n in top_n_options:
        suffix = f" - {label}" if label else ""
        ensemble_name = f"ENSEMBLE (top {top_n}){suffix}"
        top_model_names = [m.split("/")[-1][:12] for m, _ in sorted_models[:top_n]]
        print(f"{ensemble_name}: {' + '.join(top_model_names)}")

        # Evaluate ensemble on each test
        ensemble_corrs = []
        for target_word, results in all_results.items():
            if target_word not in expected_scores:
                continue

            # Create ensemble similarities
            ensemble_sims = create_ensemble_similarities(
                results,
                top_n=top_n,
                model_correlations=overall_rankings
            )

            # Compute correlation with expected
            try:
                corr = compute_correlation(ensemble_sims, expected_scores[target_word])
                ensemble_corrs.append(corr)
            except:
                pass

        if ensemble_corrs:
            avg_corr = sum(ensemble_corrs) / len(ensemble_corrs)
            ensemble_results[ensemble_name] = avg_corr
            print(f"  Average Spearman r: {avg_corr:.3f}")
        print()

    return ensemble_results


def print_final_leaderboard(
    all_correlations: list[dict[str, float]],
    model_prices: dict[str, float],
    ensemble_results: dict[str, float] = None,
):
    """Print aggregated leaderboard across all tests, including ensembles."""
    print("\n" + "=" * 80)
    print("FINAL LEADERBOARD - Aggregated across all tests")
    print("=" * 80 + "\n")

    # Aggregate scores for individual models
    model_totals = {}
    model_counts = {}

    for correlations in all_correlations:
        for model, corr in correlations.items():
            if model not in model_totals:
                model_totals[model] = 0.0
                model_counts[model] = 0
            model_totals[model] += corr
            model_counts[model] += 1

    # Calculate averages
    model_averages = {
        model: model_totals[model] / model_counts[model]
        for model in model_totals
    }

    # Combine with ensemble results
    all_entries = []

    for model, avg_corr in model_averages.items():
        model_short = model.split("/")[-1][:30]
        price = model_prices.get(model, 0)
        all_entries.append((model_short, avg_corr, price, False))  # False = not ensemble

    if ensemble_results:
        for ensemble_name, avg_corr in ensemble_results.items():
            all_entries.append((ensemble_name, avg_corr, 0, True))  # True = ensemble

    # Sort by average correlation
    all_entries.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6}{'Model':<32}{'Avg Spearman r':>14}{'Price':>14}")
    print("-" * 68)

    for rank, (name, avg_corr, price, is_ensemble) in enumerate(all_entries, 1):
        if is_ensemble:
            # Highlight ensemble entries
            print(f"{rank:<6}{name:<32}{avg_corr:>14.3f}{'(combined)':>14}")
        else:
            print(f"{rank:<6}{name:<32}{avg_corr:>14.3f}${price:>12.3f}/M")

    print()
    print("Higher Spearman r = better match with human-expected semantic rankings")
    print()


def main():
    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return

    # Models to compare (with pricing per million tokens)
    models_to_use = [
        # Commercial models
        ("openai/text-embedding-3-small", 0.02),
        ("openai/text-embedding-3-large", 0.13),
        ("google/gemini-embedding-001", 0.15),
        ("mistralai/mistral-embed-2312", 0.10),
        ("qwen/qwen3-embedding-8b", 0.01),
        # Open-source models
        ("thenlper/gte-large", 0.01),
        ("thenlper/gte-base", 0.008),
        ("intfloat/e5-large-v2", 0.01),
        ("baai/bge-base-en-v1.5", 0.008),
    ]

    # Test words for "void" - related (emptiness, space, nullify) + unrelated
    void_words = [
        # Emptiness/nothingness sense
        "empty",
        "nothing",
        "hollow",
        "blank",
        "vacant",
        "barren",
        # Space/abyss sense
        "abyss",
        "vacuum",
        "chasm",
        "darkness",
        "space",
        "expanse",
        # Legal/nullify sense
        "null",
        "cancel",
        "invalid",
        "nullify",
        "annul",
        "revoke",
        # Programming sense
        "undefined",
        "nil",
        "none",
        # Somewhat related
        "gap",
        "hole",
        "pit",
        "depth",
        # Unrelated words
        "banana",
        "keyboard",
        "happy",
        "running",
        "bicycle",
        "sandwich",
        "purple",
        "whisper",
        "mountain",
        "telephone",
    ]

    # Test words for "skirt" - clothing sense, verb sense (avoid), + unrelated
    skirt_words = [
        # Clothing - very related
        "dress",
        "blouse",
        "garment",
        "hem",
        "fabric",
        "pleated",
        "miniskirt",
        # Clothing - somewhat related
        "pants",
        "shirt",
        "jacket",
        "clothing",
        "wardrobe",
        "outfit",
        "fashion",
        # Verb sense (to go around/avoid)
        "avoid",
        "circumvent",
        "bypass",
        "evade",
        "sidestep",
        # Edge/border sense
        "edge",
        "border",
        "fringe",
        "margin",
        "perimeter",
        "periphery",
        # Unrelated words
        "elephant",
        "thunder",
        "calculate",
        "volcano",
        "algorithm",
        "breakfast",
        "gravity",
        "saxophone",
        "microscope",
        "democracy",
    ]

    # Extract just model IDs for API calls
    model_ids = [m[0] for m in models_to_use]
    model_prices = {m[0]: m[1] for m in models_to_use}

    # Add LLM ranking models to prices
    for llm_id, llm_price in LLM_RANKING_MODELS:
        model_prices[llm_id] = llm_price

    # Test words for "fire" - clear gradient from similar to unrelated
    fire_words = [
        "flame",
        "blaze",
        "burn",
        "heat",
        "smoke",
        "ember",
        "ash",
        "warm",
        "water",
        "pencil",
    ]

    # Test words for "book" - clear gradient from similar to unrelated
    book_words = [
        "novel",
        "text",
        "volume",
        "read",
        "page",
        "author",
        "library",
        "story",
        "magazine",
        "hammer",
    ]

    # Test words for "doctor" - clear gradient from similar to unrelated
    doctor_words = [
        "physician",
        "surgeon",
        "medical",
        "medicine",
        "nurse",
        "hospital",
        "patient",
        "stethoscope",
        "healthy",
        "skateboard",
    ]

    # Test words for "happy" - clear gradient from similar to unrelated
    happy_words = [
        "joyful",
        "cheerful",
        "glad",
        "pleased",
        "content",
        "excited",
        "smile",
        "celebrate",
        "sad",
        "concrete",
    ]

    # Test words for "ocean" - clear gradient from similar to unrelated
    ocean_words = [
        "sea",
        "marine",
        "waves",
        "water",
        "tide",
        "beach",
        "fish",
        "ship",
        "desert",
        "calculator",
    ]

    # Test words for "king" - clear gradient from similar to unrelated
    king_words = [
        "queen",
        "monarch",
        "royal",
        "throne",
        "crown",
        "kingdom",
        "prince",
        "castle",
        "chess",
        "peasant",
        "refrigerator",
    ]

    # Test words for "ocean2" - mixed bag including game bucket labels
    ocean2_words = [
        "airpods",
        "chocolate",
        "cold",
        "cruise",
        "embark",
        "freezing",
        "hot",
        "kindle",
        "mouse",
        "rinse",
        "sail",
        "sale",
        "score",
        "sounds",
        "spider",
        "tepid",
        "warm",
    ]

    print(f"Comparing words across {len(models_to_use)} embedding models + {len(LLM_RANKING_MODELS)} LLM models...")
    print()

    all_correlations = []
    all_results = {}  # {target_word: {model: {word: similarity}}}

    # Test "fire"
    print("=" * 90)
    print("Testing target word: 'fire'")
    fire_results = compare_across_models(
        fire_words,
        "fire",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(fire_words, "fire", LLM_RANKING_MODELS, api_key)
    fire_results.update(llm_results)
    all_results["fire"] = fire_results
    print_ranked_results(fire_results, "fire", EXPECTED_SCORES["fire"], model_prices)
    fire_corrs = print_expected_comparison(fire_results, "fire", EXPECTED_SCORES["fire"], model_prices)
    all_correlations.append(fire_corrs)

    # Test "book"
    print("=" * 90)
    print("Testing target word: 'book'")
    book_results = compare_across_models(
        book_words,
        "book",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(book_words, "book", LLM_RANKING_MODELS, api_key)
    book_results.update(llm_results)
    all_results["book"] = book_results
    print_ranked_results(book_results, "book", EXPECTED_SCORES["book"], model_prices)
    book_corrs = print_expected_comparison(book_results, "book", EXPECTED_SCORES["book"], model_prices)
    all_correlations.append(book_corrs)

    # Test "doctor"
    print("=" * 90)
    print("Testing target word: 'doctor'")
    doctor_results = compare_across_models(
        doctor_words,
        "doctor",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(doctor_words, "doctor", LLM_RANKING_MODELS, api_key)
    doctor_results.update(llm_results)
    all_results["doctor"] = doctor_results
    print_ranked_results(doctor_results, "doctor", EXPECTED_SCORES["doctor"], model_prices)
    doctor_corrs = print_expected_comparison(doctor_results, "doctor", EXPECTED_SCORES["doctor"], model_prices)
    all_correlations.append(doctor_corrs)

    # Test "happy"
    print("=" * 90)
    print("Testing target word: 'happy'")
    happy_results = compare_across_models(
        happy_words,
        "happy",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(happy_words, "happy", LLM_RANKING_MODELS, api_key)
    happy_results.update(llm_results)
    all_results["happy"] = happy_results
    print_ranked_results(happy_results, "happy", EXPECTED_SCORES["happy"], model_prices)
    happy_corrs = print_expected_comparison(happy_results, "happy", EXPECTED_SCORES["happy"], model_prices)
    all_correlations.append(happy_corrs)

    # Test "ocean"
    print("=" * 90)
    print("Testing target word: 'ocean'")
    ocean_results = compare_across_models(
        ocean_words,
        "ocean",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(ocean_words, "ocean", LLM_RANKING_MODELS, api_key)
    ocean_results.update(llm_results)
    all_results["ocean"] = ocean_results
    print_ranked_results(ocean_results, "ocean", EXPECTED_SCORES["ocean"], model_prices)
    ocean_corrs = print_expected_comparison(ocean_results, "ocean", EXPECTED_SCORES["ocean"], model_prices)
    all_correlations.append(ocean_corrs)

    # Test "ocean2" - mixed words including game bucket labels
    print("=" * 90)
    print("Testing target word: 'ocean' (test set 2 - mixed words)")
    ocean2_results = compare_across_models(
        ocean2_words,
        "ocean",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(ocean2_words, "ocean", LLM_RANKING_MODELS, api_key)
    ocean2_results.update(llm_results)
    all_results["ocean2"] = ocean2_results
    print_ranked_results(ocean2_results, "ocean", EXPECTED_SCORES["ocean2"], model_prices)
    ocean2_corrs = print_expected_comparison(ocean2_results, "ocean", EXPECTED_SCORES["ocean2"], model_prices)
    all_correlations.append(ocean2_corrs)

    # Test "king"
    print("=" * 90)
    print("Testing target word: 'king'")
    king_results = compare_across_models(
        king_words,
        "king",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(king_words, "king", LLM_RANKING_MODELS, api_key)
    king_results.update(llm_results)
    all_results["king"] = king_results
    print_ranked_results(king_results, "king", EXPECTED_SCORES["king"], model_prices)
    king_corrs = print_expected_comparison(king_results, "king", EXPECTED_SCORES["king"], model_prices)
    all_correlations.append(king_corrs)

    # Test "void"
    print("=" * 90)
    print("Testing target word: 'void'")
    void_results = compare_across_models(
        void_words,
        "void",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(void_words, "void", LLM_RANKING_MODELS, api_key)
    void_results.update(llm_results)
    all_results["void"] = void_results
    print_ranked_results(void_results, "void", EXPECTED_SCORES["void"], model_prices)
    void_corrs = print_expected_comparison(void_results, "void", EXPECTED_SCORES["void"], model_prices)
    all_correlations.append(void_corrs)

    # Test "skirt"
    print("=" * 90)
    print("Testing target word: 'skirt'")
    skirt_results = compare_across_models(
        skirt_words,
        "skirt",
        model_ids,
        api_key,
    )
    # Add LLM rankings
    llm_results = compare_llm_rankings(skirt_words, "skirt", LLM_RANKING_MODELS, api_key)
    skirt_results.update(llm_results)
    all_results["skirt"] = skirt_results
    print_ranked_results(skirt_results, "skirt", EXPECTED_SCORES["skirt"], model_prices)
    skirt_corrs = print_expected_comparison(skirt_results, "skirt", EXPECTED_SCORES["skirt"], model_prices)
    all_correlations.append(skirt_corrs)

    # Ensemble evaluation - embeddings only
    embedding_models = {m for m, _ in model_ids}
    embedding_ensemble_results = run_ensemble_evaluation(
        all_results,
        all_correlations,
        EXPECTED_SCORES,
        top_n_options=[2, 3, 5],
        allowed_models=embedding_models,
        label="embeddings",
    )

    # Ensemble evaluation - LLM rankings only
    llm_models = {m for m, _ in LLM_RANKING_MODELS}
    llm_ensemble_results = run_ensemble_evaluation(
        all_results,
        all_correlations,
        EXPECTED_SCORES,
        top_n_options=[2, 3],
        allowed_models=llm_models,
        label="llms",
    )

    ensemble_results = {**embedding_ensemble_results, **llm_ensemble_results}

    # Final aggregated leaderboard (includes ensembles)
    print_final_leaderboard(all_correlations, model_prices, ensemble_results)

    # Show embedding dimensions for each model
    print("=" * 80)
    print("Model details:")
    print(f"{'Model':<35} {'Type':>12} {'Dims/Method':>14} {'Price/M':>12}")
    print("-" * 75)
    for model_id, price in models_to_use:
        try:
            embeddings = get_embeddings(["test"], model_id, api_key)
            dim = len(embeddings["test"])
            print(f"{model_id:<35} {'Embedding':>12} {dim:>14} ${price:>11.3f}")
        except Exception as e:
            print(f"{model_id:<35} {'Embedding':>12} {'error':>14} ${price:>11.3f}")

    # Show LLM ranking models
    for model_id, price in LLM_RANKING_MODELS:
        print(f"{model_id:<35} {'LLM':>12} {'ranking':>14} ${price:>11.3f}")


if __name__ == "__main__":
    main()
