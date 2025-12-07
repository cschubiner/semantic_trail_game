# Semantic Trail Worker - Development Notes

## Deployment

**IMPORTANT: Always deploy the worker after making any changes to the backend code!**

```bash
cd /Users/canal/word_guessing_game/worker
npm run deploy
```

The worker is deployed to: https://semantic-trail-backend.cschubiner.workers.dev

## Architecture

- **Cloudflare Worker** with KV storage for embedding cache
- **Ensemble embeddings**: google/gemini-embedding-001 + thenlper/gte-base
- **LLM re-ranking**: google/gemini-2.5-flash (with $1/hour budget protection)

## API Endpoints

- `POST /score` - Score a guess (accepts `game` param for multiple games per day)
- `POST /rerank` - LLM re-ranking of top guesses
- `GET /hint?type=letter|length&game=N` - Get hints
- `GET /reveal?game=N` - Reveal the secret word
- `GET /health` - Health check

## Environment Variables

- `OPENROUTER_API_KEY` - API key for OpenRouter
- `SECRET_SALT` - Salt for deterministic word selection
- `ALLOWED_ORIGINS` - CORS allowed origins
- `EMBED_CACHE` - KV namespace for caching embeddings
