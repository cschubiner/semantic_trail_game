#!/usr/bin/env python3
"""
Semantic Trail word guessing game using ensemble embeddings.
Uses top 2 embedding models averaged together for similarity scoring.
"""

import os
import json
import random
import sys
from pathlib import Path

import requests
import numpy as np


# Top 2 models for ensemble (based on benchmark results)
ENSEMBLE_MODELS = [
    "openai/text-embedding-3-large",
    "thenlper/gte-large",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
CACHE_DIR = Path(__file__).parent / ".embedding_cache"
WORD_LIST_PATH = Path(__file__).parent / "google-10000-english-usa.txt"

# Common stop words to exclude
STOP_WORDS = {
    "the", "of", "and", "to", "in", "for", "is", "on", "that", "by",
    "this", "with", "you", "it", "not", "or", "be", "are", "from",
    "at", "as", "your", "all", "have", "new", "more", "an", "was",
    "we", "will", "can", "us", "about", "if", "my", "has", "but",
    "our", "one", "other", "do", "no", "time", "very", "when", "who",
    "way", "may", "been", "would", "could", "should", "into", "than",
    "them", "these", "so", "some", "her", "him", "his", "its", "they",
    "she", "he", "what", "which", "their", "said", "each", "how",
    "were", "had", "also", "did", "just", "over", "such", "any",
    "only", "come", "made", "find", "here", "many", "where", "those",
    "then", "now", "look", "get", "go", "see", "well", "back", "being",
    "there", "much", "through", "use", "before", "because", "good",
    "think", "same", "own", "most", "make", "after", "first", "like",
    "want", "does", "using", "take", "every", "even", "while", "still",
    "between", "under", "last", "never", "another", "around", "however",
    "without", "within", "again", "both", "during", "got", "let",
    "www", "http", "https", "com", "org", "net", "html", "php",
}

# ANSI escape codes
CLEAR_SCREEN = "\033[2J"
MOVE_HOME = "\033[H"
CLEAR_LINE = "\033[2K"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def get_cache_path(word: str, model: str) -> Path:
    """Get the cache file path for a word/model combination."""
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


def get_embedding(word: str, model: str, api_key: str) -> list[float]:
    """Get embedding for a single word, using cache if available."""
    cached = load_from_cache(word, model)
    if cached is not None:
        return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": word,
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    embedding = data["data"][0]["embedding"]

    save_to_cache(word, model, embedding)
    return embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_ensemble_similarity(
    word1: str,
    word2: str,
    api_key: str,
    models: list[str] = ENSEMBLE_MODELS,
) -> float:
    """
    Get similarity between two words using ensemble of models.
    Averages the cosine similarity from each model.
    """
    similarities = []

    for model in models:
        emb1 = get_embedding(word1, model, api_key)
        emb2 = get_embedding(word2, model, api_key)
        sim = cosine_similarity(emb1, emb2)
        similarities.append(sim)

    return sum(similarities) / len(similarities)


def load_word_list() -> list[str]:
    """Load and filter word list."""
    with open(WORD_LIST_PATH, "r") as f:
        words = [line.strip().lower() for line in f]

    # Filter: at least 4 letters, not a stop word, alphabetic only
    filtered = [
        w for w in words
        if len(w) >= 4
        and w not in STOP_WORDS
        and w.isalpha()
    ]

    return filtered


def similarity_to_score(similarity: float) -> int:
    """
    Convert cosine similarity to a game score (0-100).
    """
    min_sim = 0.20
    max_sim = 0.85

    if similarity >= max_sim:
        return 100
    if similarity <= min_sim:
        return 0

    score = (similarity - min_sim) / (max_sim - min_sim) * 100
    return int(round(score))


def score_to_bar(score: int, width: int = 20) -> str:
    """Create a visual progress bar for the score."""
    filled = int(score / 100 * width)
    empty = width - filled

    if score >= 80:
        color = GREEN
    elif score >= 50:
        color = YELLOW
    else:
        color = RED

    bar = color + "█" * filled + DIM + "░" * empty + RESET
    return bar


def get_temperature(score: int) -> str:
    """Get temperature indicator based on score."""
    if score >= 95:
        return f"{GREEN}🔥 BURNING{RESET}"
    elif score >= 80:
        return f"{GREEN}🔥 Hot{RESET}"
    elif score >= 60:
        return f"{YELLOW}☀ Warm{RESET}"
    elif score >= 40:
        return f"{YELLOW}〰 Tepid{RESET}"
    elif score >= 20:
        return f"{CYAN}❄ Cold{RESET}"
    else:
        return f"{CYAN}🧊 Freezing{RESET}"


def render_screen(guesses: list, answer: str, message: str = "", won: bool = False):
    """Render the full game screen."""
    # Clear and move to top
    sys.stdout.write(CLEAR_SCREEN + MOVE_HOME)

    # Header
    print(f"{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  SEMANTIC TRAIL{RESET} - Find the secret word!")
    print(f"{DIM}  Ensemble: {' + '.join(m.split('/')[-1] for m in ENSEMBLE_MODELS)}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}")
    print()

    if won:
        print(f"{GREEN}{BOLD}  🎉 CORRECT! The word was: {answer.upper()} 🎉{RESET}")
        print(f"  You found it in {len(guesses)} guesses!")
        print()

    # Guesses header
    if guesses:
        print(f"{BOLD}  {'#':<4} {'WORD':<18} {'SCORE':<8} {'':22} {RESET}")
        print(f"  {'-' * 54}")

        # Sort by score descending
        sorted_guesses = sorted(guesses, key=lambda x: x[2], reverse=True)

        # Show all guesses (or limit to terminal height)
        display_count = min(len(sorted_guesses), 20)

        for i, (word, similarity, score) in enumerate(sorted_guesses[:display_count]):
            rank = i + 1
            bar = score_to_bar(score)
            temp = get_temperature(score)

            # Highlight the most recent guess
            if word == guesses[-1][0]:
                print(f"{MAGENTA}  {rank:<4} {word:<18} {score:>3}     {bar} {temp}{RESET}")
            else:
                print(f"  {rank:<4} {word:<18} {score:>3}     {bar} {temp}")

        if len(sorted_guesses) > display_count:
            print(f"{DIM}  ... and {len(sorted_guesses) - display_count} more guesses{RESET}")

        print()
    else:
        print(f"{DIM}  No guesses yet. Start typing!{RESET}")
        print()

    # Message area
    if message:
        print(f"  {message}")
        print()

    # Help
    print(f"{DIM}  Commands: 'hint' | 'give up' | 'quit'{RESET}")
    print()


def play_game(api_key: str):
    """Main game loop."""
    # Load words and pick random answer
    words = load_word_list()
    answer = random.choice(words)

    guesses = []  # List of (word, similarity, score)
    message = f"I'm thinking of a word... ({len(words):,} possible words)"

    render_screen(guesses, answer, message)

    while True:
        try:
            user_input = input("  Guess: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nThanks for playing!")
            return False

        if not user_input:
            render_screen(guesses, answer, message)
            continue

        if user_input == "quit":
            render_screen(guesses, answer, f"The word was: {BOLD}{answer.upper()}{RESET}")
            print("  Thanks for playing!\n")
            return False

        if user_input == "give up":
            render_screen(guesses, answer, f"The word was: {BOLD}{answer.upper()}{RESET} - You made {len(guesses)} guesses.")
            return True  # Play again prompt

        if user_input == "hint":
            message = f"Hint: {len(answer)} letters, starts with '{answer[0].upper()}'"
            render_screen(guesses, answer, message)
            continue

        if not user_input.isalpha():
            message = f"{RED}Please enter a valid word (letters only){RESET}"
            render_screen(guesses, answer, message)
            continue

        # Check if already guessed
        if any(g[0] == user_input for g in guesses):
            existing = next(g for g in guesses if g[0] == user_input)
            message = f"{YELLOW}Already guessed '{user_input}' - Score was: {existing[2]}{RESET}"
            render_screen(guesses, answer, message)
            continue

        # Check for win
        if user_input == answer:
            guesses.append((user_input, 1.0, 100))
            render_screen(guesses, answer, "", won=True)
            return True  # Play again prompt

        # Get similarity
        try:
            message = f"{DIM}Thinking...{RESET}"
            render_screen(guesses, answer, message)

            similarity = get_ensemble_similarity(user_input, answer, api_key)
            score = similarity_to_score(similarity)

            guesses.append((user_input, similarity, score))

            # Find rank of new guess
            sorted_guesses = sorted(guesses, key=lambda x: x[2], reverse=True)
            rank = next(i for i, g in enumerate(sorted_guesses, 1) if g[0] == user_input)

            if rank == 1:
                message = f"{GREEN}New best guess!{RESET}"
            else:
                message = f"Ranked #{rank} of {len(guesses)}"

            render_screen(guesses, answer, message)

        except Exception as e:
            message = f"{RED}Error: {e}{RESET}"
            render_screen(guesses, answer, message)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return

    play_again = True
    while play_again:
        result = play_game(api_key)
        if result:
            try:
                response = input("  Play again? (y/n): ").strip().lower()
                play_again = response == "y"
            except (KeyboardInterrupt, EOFError):
                play_again = False
        else:
            play_again = False

    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
