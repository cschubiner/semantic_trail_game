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
- **Single embedding model**: thenlper/gte-base (GTE-base)
- **LLM re-ranking**: google/gemini-2.5-flash (with $2/hour budget protection)
- **LLM hints**: anthropic/claude-sonnet-4.5 (shares same $2/hour budget)
- **Validation**: Min 3 chars, no stop words

## API Endpoints

- `POST /score` - Score a guess (accepts `game` param for multiple games per day)
- `POST /rerank` - LLM re-ranking of top guesses (Gemini 2.5 Flash)
- `POST /llm-hint` - Get a vague hint from Claude based on top guesses (Claude Sonnet 4.5)
- `GET /hint?type=letter|length&game=N` - Get hints (first letter or word length)
- `GET /reveal?game=N` - Reveal the secret word
- `GET /health` - Health check

## Environment Variables

- `OPENROUTER_API_KEY` - API key for OpenRouter
- `SECRET_SALT` - Salt for deterministic word selection
- `ALLOWED_ORIGINS` - CORS allowed origins
- `EMBED_CACHE` - KV namespace for caching embeddings

## R2 Embeddings Setup (Optional - for better performance)

The worker code is already prepared to use R2 when available, with graceful fallback to API.

To precompute embeddings and store in R2:

1. **Enable R2** in Cloudflare Dashboard (free tier: 10GB storage, 10M ops/month)

2. **Generate embeddings** (~3 hours for 8800 words):
   ```bash
   cd /Users/canal/word_guessing_game/worker
   OPENROUTER_API_KEY=your-key node scripts/generate-embeddings.js
   ```

3. **Upload to R2**:
   ```bash
   chmod +x scripts/upload-to-r2.sh
   ./scripts/upload-to-r2.sh
   ```

4. **Uncomment R2 binding** in wrangler.toml:
   ```toml
   [[r2_buckets]]
   binding = "EMBEDDINGS_BUCKET"
   bucket_name = "semantic-trail-embeddings"
   ```

5. **Deploy**: `npm run deploy`

Once R2 is set up, the worker will automatically use precomputed embeddings for faster responses.
