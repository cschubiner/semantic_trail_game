# Semantic Trail - Embedding-Based Word Guessing Game

👉 Play it on GitHub Pages: https://cschubiner.github.io/semantic_trail_game/

Semantic Trail is a word guessing game powered by an ensemble of embedding models. Includes a comprehensive benchmark suite for evaluating embedding models against human-defined semantic similarity judgments.

**Two ways to play:**
- **Web App** — Browser-based game with voice input support (Cloudflare Worker backend)
- **CLI** — Terminal-based game with Python

---

## Web Architecture Overview

The project includes a web app version that anyone can play in a browser.

```
┌──────────────────────────────────────────────────────────────┐
│                    Web Browser (Frontend)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │
│  │  index.html │  │  styles.css │  │      game.js        │   │
│  │   Game UI   │  │ Dark theme  │  │ - Fetch /score      │   │
│  │             │  │ Responsive  │  │ - Web Speech API    │   │
│  └─────────────┘  └─────────────┘  │ - Mic input         │   │
│                                     └─────────────────────┘   │
│                              │                                │
│                              │ POST /score                    │
│                              ▼                                │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                Cloudflare Worker (Backend)                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  POST /score                                            │  │
│  │  1. Validate guess                                      │  │
│  │  2. Get today's secret word (date + salt hash)         │  │
│  │  3. Fetch embeddings (KV cache or OpenRouter API)      │  │
│  │  4. Compute ensemble cosine similarity                 │  │
│  │  5. Return { score, similarity, bucket }               │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │                                │
│              ┌───────────────┴───────────────┐               │
│              ▼                               ▼               │
│  ┌─────────────────────┐         ┌─────────────────────┐    │
│  │   Cloudflare KV     │         │   OpenRouter API    │    │
│  │  (Embedding Cache)  │         │   (if not cached)   │    │
│  │                     │         │                     │    │
│  │  Key: model:word    │         │  text-embedding-3   │    │
│  │  Val: [vector...]   │         │  gte-large          │    │
│  └─────────────────────┘         └─────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Key Features

- **Lazy KV Cache**: Embeddings are cached on-demand. KV starts empty and fills as words are guessed.
- **Deterministic Daily Word**: Same secret word for everyone each day (based on date + secret salt), resets at 5am ET.
- **Voice Input**: Web Speech API enables microphone guessing in supported browsers.
- **CORS Enabled**: Frontend can be hosted anywhere (GitHub Pages, etc.).

---

## How It Works

### The Game (`semantic_trail_game.py`)

You try to guess a secret word. After each guess, you receive a similarity score (0-100) indicating how semantically close your guess is to the target word.

```
════════════════════════════════════════════════════════════
  SEMANTIC TRAIL - Find the secret word!
  Ensemble: text-embedding-3-large + gte-large
════════════════════════════════════════════════════════════

  #    WORD               SCORE
  ------------------------------------------------------
  1    flame              78      ████████████████░░░░ 🔥 Hot
  2    heat               61      ████████████░░░░░░░░ ☀ Warm
  3    water              23      █████░░░░░░░░░░░░░░░ ❄ Cold

  Commands: 'hint' | 'give up' | 'quit'

  Guess: _
```

**The secret sauce**: Instead of using a single embedding model, the game uses an **ensemble of the top 2 performing models** (determined by the benchmark). Similarity scores from both models are averaged, which produces more robust semantic rankings than any single model alone.

### The Benchmark (`embedding_comparison.py`)

Evaluates 9 embedding models against human-defined "ground truth" semantic similarity scores across 8 test words.

#### Models Tested
| Model | Price/M tokens | Notes |
|-------|---------------|-------|
| `openai/text-embedding-3-small` | $0.02 | Fast, cheap |
| `openai/text-embedding-3-large` | $0.13 | Best OpenAI |
| `google/gemini-embedding-001` | $0.15 | Top MTEB benchmark |
| `mistralai/mistral-embed-2312` | $0.10 | 1024 dimensions |
| `qwen/qwen3-embedding-8b` | $0.01 | Cheapest, multilingual |
| `thenlper/gte-large` | $0.01 | Open-source |
| `thenlper/gte-base` | $0.008 | Smaller GTE |
| `intfloat/e5-large-v2` | $0.01 | Open-source |
| `baai/bge-base-en-v1.5` | $0.008 | Open-source |

#### Ground Truth Scoring

I (Claude) defined expected similarity scores for each test word based on semantic knowledge. For example, for the target word **"fire"**:

| Score | Word | Reasoning |
|-------|------|-----------|
| 95 | flame | Direct synonym |
| 90 | blaze | Synonym for large fire |
| 82 | burn | What fire does |
| 70 | heat | Property of fire |
| 62 | smoke | Byproduct |
| 58 | ember | Remnant of fire |
| 50 | ash | Result of fire |
| 38 | warm | Effect (adjective) |
| 15 | water | Opposite (extinguishes fire) |
| 2 | pencil | Completely unrelated |

Models are evaluated using **Spearman rank correlation** — how well does the model's ranking match the expected ranking?

#### Ensemble Evaluation

The benchmark also tests combining top models:
- **ENSEMBLE (top 2)**: Averages similarity scores from the 2 best-performing models
- **ENSEMBLE (top 3)**: Averages from top 3
- **ENSEMBLE (top 5)**: Averages from top 5

Ensembles often outperform individual models by capturing different semantic aspects.

## Setup

This repo has two main usage modes:

1. **Local Python CLI + Benchmarks** — Play in the terminal and run model benchmarks
2. **Web App (Cloudflare Worker + GitHub Pages)** — Host a public web game

---

### Python CLI Setup

#### Requirements

```bash
pip install requests numpy scipy
```

#### API Key

Get an API key from [OpenRouter](https://openrouter.ai/) and set it:

```bash
export OPENROUTER_API_KEY='your-key-here'
```

#### Word List

The game uses `google-10000-english-usa.txt` (10,000 most common English words). This file should be in the same directory as the scripts.

---

### Web App Setup (Cloudflare Worker)

The web app consists of:
- `worker/` — Cloudflare Worker backend
- `web/` — Static HTML/CSS/JS frontend

#### 1. Install Wrangler CLI

```bash
npm install -g wrangler
wrangler login
```

#### 2. Create KV Namespace

```bash
cd worker
wrangler kv:namespace create EMBED_CACHE
```

Copy the output ID and update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "EMBED_CACHE"
id = "your-kv-namespace-id"
```

#### 3. Set Environment Variables

Edit `wrangler.toml` to set a random `SECRET_SALT`:

```toml
[vars]
SECRET_SALT = "your-random-secret-string"
```

Set your OpenRouter API key as a secret:

```bash
wrangler secret put OPENROUTER_API_KEY
```

#### 4. Common Deploy Commands (Worker)

```bash
# One-time: log in to Cloudflare
wrangler login

# One-time: create KV namespace and paste ID into wrangler.toml
cd worker
wrangler kv:namespace create EMBED_CACHE

# One-time: set secrets (interactive prompt)
npx wrangler secret put OPENROUTER_API_KEY

# One-time: set vars in wrangler.toml (non-secret)
# [vars]
# SECRET_SALT = "your-random-secret-string"
# ALLOWED_ORIGINS = "https://cschubiner.github.io"

# Deploy worker
npx wrangler deploy          # or: npm run deploy

# Tail logs (useful for debugging)
npx wrangler tail

# Local dev server
npx wrangler dev             # or: npm run dev
```

Wrangler will print your Worker URL, e.g.:
```
https://semantic-trail-backend.<your-subdomain>.workers.dev
```

#### 5. Configure Frontend

Update `web/game.js` with your Worker URL:

```javascript
const API_URL = 'https://semantic-trail-backend.<your-subdomain>.workers.dev/score';
const DEMO_MODE = false;  // Set to false to use real backend
```

#### 6. Host Frontend (GitHub Pages)

This repo includes a GitHub Actions workflow (`.github/workflows/pages.yml`) that publishes `web/` to Pages.

Steps:
1. In GitHub → Settings → Pages, set Source to “GitHub Actions.”
2. Push to `master`/`main` — the workflow uploads `web/` as the Pages artifact and deploys.
3. Confirm the Pages URL (e.g., `https://cschubiner.github.io/semantic_trail_game/`).

For local testing:

```bash
cd web
python3 -m http.server 8080
# Open http://localhost:8080
```

## Usage

### Play the Game

```bash
python semantic_trail_game.py
```

**Commands:**
- Type any word to guess
- `hint` — Shows word length and first letter
- `give up` — Reveals the answer
- `quit` — Exit

**Scoring:**
- 🔥 BURNING (95+) — Almost there!
- 🔥 Hot (80-94) — Very close
- ☀ Warm (60-79) — Getting warmer
- 〰 Tepid (40-59) — Lukewarm
- ❄ Cold (20-39) — Not close
- 🧊 Freezing (0-19) — Way off

### Run the Benchmark

```bash
python embedding_comparison.py
```

This will:
1. Test 8 target words across 9 models
2. Show ranked results compared to expected scores
3. Evaluate ensemble combinations
4. Print a final leaderboard

## Caching

Embeddings are cached to `.embedding_cache/` to avoid redundant API calls:

```
.embedding_cache/
├── openai_text-embedding-3-large/
│   ├── fire.json
│   ├── flame.json
│   └── ...
├── thenlper_gte-large/
│   └── ...
└── ...
```

Subsequent runs (benchmark or game) will be much faster as cached embeddings are reused.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   User Guess    │────▶│  OpenRouter API  │
│   "flame"       │     │  (if not cached) │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Embedding Cache      │
                    │   .embedding_cache/    │
                    └────────────┬───────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌───────────────────┐                         ┌───────────────────┐
│ text-embedding-3  │                         │    gte-large      │
│     (large)       │                         │                   │
│                   │                         │                   │
│ cosine_sim = 0.72 │                         │ cosine_sim = 0.68 │
└─────────┬─────────┘                         └─────────┬─────────┘
          │                                             │
          └──────────────────┬──────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Ensemble Average │
                    │  (0.72 + 0.68)/2 │
                    │     = 0.70       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Score: 77/100  │
                    │    ☀ Warm       │
                    └─────────────────┘
```

## Test Words

The benchmark evaluates these 8 target words:

| Word | # Test Words | Categories Tested |
|------|-------------|-------------------|
| fire | 10 | synonyms, properties, byproducts, opposites |
| book | 10 | synonyms, components, related concepts |
| doctor | 10 | synonyms, workplace, tools, patients |
| happy | 10 | synonyms, related emotions, antonyms |
| ocean | 10 | synonyms, features, inhabitants |
| king | 11 | synonyms, royalty, symbols, opposites |
| void | 28 | emptiness, legal, programming, physics |
| skirt | 27 | clothing, verb sense (avoid), edge/border |

## How Similarity Scoring Works

1. **Cosine Similarity**: Measures angle between embedding vectors (0 to 1)
2. **Score Mapping**: Cosine similarity is mapped to 0-100 scale
   - `similarity < 0.20` → Score 0
   - `similarity > 0.85` → Score 100
   - Linear interpolation in between

```python
score = (similarity - 0.20) / (0.85 - 0.20) * 100
```

## Files

```
word_guessing_game/
├── README.md                    # This file
├── semantic_trail_game.py       # CLI game (Python)
├── embedding_comparison.py      # Benchmark suite
├── google-10000-english-usa.txt # Word list (10k words)
├── .embedding_cache/            # Python CLI cache (auto-created)
│
├── web/                         # Web frontend
│   ├── index.html               # Game UI
│   ├── styles.css               # Dark theme styling
│   └── game.js                  # Game logic + Web Speech API
│
└── worker/                      # Cloudflare Worker backend
    ├── wrangler.toml            # Wrangler configuration
    ├── package.json             # Node dependencies
    ├── tsconfig.json            # TypeScript config
    └── src/
        ├── index.ts             # Worker entry point (/score endpoint)
        └── wordlist.ts          # Filtered word list (8825 words)
```

## Worker API Reference

### `POST /score`

Score a guess against today's secret word.

**Request:**
```json
{
  "guess": "flame"
}
```

**Response (success):**
```json
{
  "guess": "flame",
  "similarity": 0.72,
  "score": 77,
  "bucket": "Hot",
  "isCorrect": false
}
```

**Response (correct guess):**
```json
{
  "guess": "fire",
  "similarity": 1.0,
  "score": 100,
  "bucket": "CORRECT!",
  "isCorrect": true
}
```

**Response (error):**
```json
{
  "error": "Missing guess field"
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Credits

- **Embeddings**: Served via [OpenRouter](https://openrouter.ai/)
- **Word List**: [google-10000-english](https://github.com/first20hours/google-10000-english)
- **Inspiration**: [Semantle](https://semantle.com/) by David Turner
