/**
 * Semantic Trail Backend - Cloudflare Worker
 *
 * Exposes POST /score endpoint for the word guessing game.
 * Uses ensemble embeddings with lazy KV caching.
 */

import { WORD_LIST } from './wordlist';

// Ensemble models (production pair)
const ENSEMBLE_MODELS = [
  'google/gemini-embedding-001',
  'thenlper/gte-base',
];

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings';
const OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions';

// LLM for re-ranking
const RERANK_MODEL = 'google/gemini-2.5-flash';

// LLM for hints
const HINT_MODEL = 'anthropic/claude-sonnet-4.5';

// Cost protection: $2/hour budget
const HOURLY_BUDGET_CENTS = 200; // $2.00 in cents
// Gemini 2.5 Flash pricing (approximate): $0.15/1M input, $0.60/1M output
// Estimate ~500 tokens per rerank call, ~$0.0003 per call = 0.03 cents
const ESTIMATED_COST_PER_RERANK_CENTS = 0.05; // Conservative estimate
// Claude Sonnet 4.5 pricing: ~$3/1M input, $15/1M output
// Estimate ~300 tokens per hint call = ~0.5 cents
const ESTIMATED_COST_PER_HINT_CENTS = 0.5;

// Similarity to score mapping (non-linear)
const MIN_SIM = 0.10; // anything below this is ~0
const MAX_SIM = 0.80; // anything above this is ~100
const SCORE_CURVE = 1.75; // >1 makes scores drop off faster (fewer "Warm" results)

const EASTERN_FORMATTER = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  hour12: false,
});

interface Env {
  EMBED_CACHE: KVNamespace;
  OPENROUTER_API_KEY: string;
  SECRET_SALT: string;
  ALLOWED_ORIGINS?: string;
}

interface ScoreRequest {
  guess: string;
  game?: number; // 0 = word of the day, 1+ = random games
}

interface ScoreResponse {
  guess: string;
  similarity: number;
  score: number;
  isCorrect?: boolean;
}

interface ErrorResponse {
  error: string;
}

interface HintResponse {
  hint: string;
  hintType: 'first_letter' | 'length';
}

interface RevealResponse {
  word: string;
  message: string;
}

interface RerankRequest {
  guesses: Array<{
    word: string;
    score: number;
  }>;
  game?: number;
}

interface RerankResponse {
  rankings: Array<{
    word: string;
    rank: number;
    llmScore?: number;
  }>;
  rateLimited?: boolean;
  message?: string;
}

interface LLMHintRequest {
  topGuesses: Array<{ word: string; score: number }>;
  game?: number;
}

interface LLMHintResponse {
  hint: string;
  rateLimited?: boolean;
}

/**
 * Build CORS headers based on allowed origins list.
 * Defaults to "*" if ALLOWED_ORIGINS is not set.
 */
function getCorsHeaders(request: Request, env: Env): Record<string, string> {
  const allowed = (env.ALLOWED_ORIGINS ?? '*')
    .split(',')
    .map(origin => origin.trim())
    .filter(Boolean);

  const origin = request.headers.get('Origin');
  let allowOrigin = '*';

  if (allowed.length && allowed[0] !== '*') {
    if (origin && allowed.includes(origin)) {
      allowOrigin = origin;
    } else {
      // Default to first allowed origin if incoming origin is missing/unmatched
      allowOrigin = allowed[0];
    }
  }

  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

/**
 * Convert cosine similarity to 0-100 score
 */
function similarityToScore(similarity: number): number {
  if (similarity >= MAX_SIM) return 100;
  if (similarity <= MIN_SIM) return 0;

  const normalized = (similarity - MIN_SIM) / (MAX_SIM - MIN_SIM);
  const curved = Math.pow(normalized, SCORE_CURVE);
  const score = Math.round(curved * 100);

  if (score < 0) return 0;
  if (score > 100) return 100;
  return score;
}

/**
 * Get year/month/day/hour in America/New_York.
 */
function getEasternParts(date: Date): { year: number; month: string; day: string; hour: number } {
  const parts = EASTERN_FORMATTER.formatToParts(date);
  const lookup: Record<string, string> = {};
  for (const { type, value } of parts) {
    if (type !== 'literal') {
      lookup[type] = value;
    }
  }
  return {
    year: Number(lookup.year),
    month: lookup.month,
    day: lookup.day,
    hour: Number(lookup.hour),
  };
}

/**
 * Get the date string (YYYY-MM-DD) that represents the "game day".
 * The day rolls over at 5am Eastern Time.
 */
function getGameDateString(): string {
  const now = new Date();
  const parts = getEasternParts(now);

  if (parts.hour < 5) {
    const prev = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const prevParts = getEasternParts(prev);
    return `${prevParts.year}-${prevParts.month}-${prevParts.day}`;
  }

  return `${parts.year}-${parts.month}-${parts.day}`;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Simple hash function for deterministic secret word selection
 */
async function hashString(str: string): Promise<number> {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = new Uint8Array(hashBuffer);

  // Use first 4 bytes as a 32-bit integer
  let hash = 0;
  for (let i = 0; i < 4; i++) {
    hash = (hash << 8) | hashArray[i];
  }
  return Math.abs(hash);
}

/**
 * Get today's secret word deterministically
 * @param salt - Secret salt for hashing
 * @param gameIndex - 0 for word of the day, 1+ for additional random games
 */
async function getSecretWord(salt: string, gameIndex: number = 0): Promise<string> {
  const gameDay = getGameDateString(); // YYYY-MM-DD in ET with 5am cutoff
  // Include gameIndex in hash so each game gets a different word
  const hash = await hashString(gameDay + salt + ':' + gameIndex);
  const index = hash % WORD_LIST.length;
  return WORD_LIST[index];
}

/**
 * Build KV cache key for a word/model combination
 */
function getCacheKey(word: string, model: string): string {
  return `${model}:${word}`;
}

/**
 * Get embedding from cache or fetch from OpenRouter
 */
async function getEmbedding(
  word: string,
  model: string,
  env: Env
): Promise<number[]> {
  const cacheKey = getCacheKey(word, model);

  // Try cache first
  const cached = await env.EMBED_CACHE.get(cacheKey, 'json');
  if (cached) {
    return cached as number[];
  }

  // Fetch from OpenRouter
  const response = await fetch(OPENROUTER_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: model,
      input: word,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenRouter API error: ${response.status} - ${text}`);
  }

  const data = await response.json() as { data: Array<{ embedding: number[] }> };
  const embedding = data.data[0].embedding;

  // Cache for future use (no expiration)
  await env.EMBED_CACHE.put(cacheKey, JSON.stringify(embedding));

  return embedding;
}

/**
 * Get ensemble similarity between two words
 */
async function getEnsembleSimilarity(
  word1: string,
  word2: string,
  env: Env
): Promise<number> {
  const similarities: number[] = [];

  for (const model of ENSEMBLE_MODELS) {
    const [emb1, emb2] = await Promise.all([
      getEmbedding(word1, model, env),
      getEmbedding(word2, model, env),
    ]);

    const sim = cosineSimilarity(emb1, emb2);
    similarities.push(sim);
  }

  // Average similarities
  return similarities.reduce((a, b) => a + b, 0) / similarities.length;
}

/**
 * Get the current hour key for cost tracking (YYYY-MM-DD-HH in UTC)
 */
function getHourKey(): string {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, '0');
  const day = String(now.getUTCDate()).padStart(2, '0');
  const hour = String(now.getUTCHours()).padStart(2, '0');
  return `cost:${year}-${month}-${day}-${hour}`;
}

/**
 * Check if we're under budget and increment cost if so
 * Returns true if the request can proceed, false if rate limited
 */
async function checkAndIncrementCost(env: Env, costCents: number = ESTIMATED_COST_PER_RERANK_CENTS): Promise<boolean> {
  const hourKey = getHourKey();

  // Get current spend for this hour
  const currentSpendStr = await env.EMBED_CACHE.get(hourKey);
  const currentSpend = currentSpendStr ? parseFloat(currentSpendStr) : 0;

  // Check if we'd exceed budget
  if (currentSpend + costCents > HOURLY_BUDGET_CENTS) {
    console.log(`Rate limited: current spend ${currentSpend} cents, budget ${HOURLY_BUDGET_CENTS} cents`);
    return false;
  }

  // Increment and save (with 2 hour TTL so old keys expire)
  const newSpend = currentSpend + costCents;
  await env.EMBED_CACHE.put(hourKey, newSpend.toString(), { expirationTtl: 7200 });

  return true;
}

/**
 * Call LLM to re-rank guesses based on semantic similarity to secret word
 */
async function rerankWithLLM(
  guesses: Array<{ word: string; score: number }>,
  secret: string,
  env: Env
): Promise<Array<{ word: string; rank: number; llmScore?: number }>> {
  // Build the prompt for ranking
  const wordList = guesses.map(g => g.word).join(', ');

  const prompt = `You are helping with a word guessing game. The secret word is "${secret}".

Given these guessed words, rank them from most to least semantically similar to "${secret}".
Consider meaning, context, associations, and conceptual relationships.

Words to rank: ${wordList}

Respond with ONLY a JSON array of objects with "word" and "score" (1-100, higher = more similar).
Example format: [{"word": "example", "score": 85}, ...]

Be precise and consistent. Output valid JSON only, no explanation.`;

  const response = await fetch(OPENROUTER_CHAT_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: RERANK_MODEL,
      messages: [
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 1000,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`LLM API error: ${response.status} - ${text}`);
  }

  const data = await response.json() as {
    choices: Array<{ message: { content: string } }>;
  };

  const content = data.choices[0]?.message?.content || '';

  // Parse JSON from response (handle potential markdown code blocks)
  let jsonStr = content.trim();
  if (jsonStr.startsWith('```')) {
    jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
  }

  const rankings = JSON.parse(jsonStr) as Array<{ word: string; score: number }>;

  // Sort by score descending and assign ranks
  rankings.sort((a, b) => b.score - a.score);

  return rankings.map((r, idx) => ({
    word: r.word.toLowerCase(),
    rank: idx + 1,
    llmScore: r.score,
  }));
}

/**
 * Handle POST /score requests
 */
async function handleScore(request: Request, env: Env): Promise<Response> {
  // Parse request body
  let body: ScoreRequest;
  try {
    body = await request.json() as ScoreRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' }, 400, request, env);
  }

  // Validate guess
  const guess = body.guess?.toLowerCase().trim();
  if (!guess) {
    return jsonResponse({ error: 'Missing guess field' }, 400, request, env);
  }

  if (!/^[a-z]+$/.test(guess)) {
    return jsonResponse({ error: 'Guess must contain only letters' }, 400, request, env);
  }

  // Get game index (0 = word of the day, 1+ = random games)
  const gameIndex = body.game ?? 0;

  // Get today's secret word
  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  // Check for exact match
  if (guess === secret) {
    const response: ScoreResponse = {
      guess,
      similarity: 1.0,
      score: 100,
      isCorrect: true,
    };
    return jsonResponse(response, 200, request, env);
  }

  // Check if guess is in word list (optional validation)
  // For now, we allow any word but could restrict to WORD_LIST

  // Get ensemble similarity
  try {
    const similarity = await getEnsembleSimilarity(guess, secret, env);
    const score = similarityToScore(similarity);

    const response: ScoreResponse = {
      guess,
      similarity: Math.round(similarity * 1000) / 1000,
      score,
      isCorrect: false,
    };

    return jsonResponse(response, 200, request, env);
  } catch (error) {
    console.error('Error getting similarity:', error);
    return jsonResponse(
      { error: `Failed to compute similarity: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500,
      request,
      env
    );
  }
}

/**
 * Handle GET /hint?type=letter or /hint?type=length
 */
async function handleHint(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const hintType = url.searchParams.get('type') || 'letter';
  const gameIndex = parseInt(url.searchParams.get('game') || '0', 10);

  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  let hint: string;
  let type: 'first_letter' | 'length';

  if (hintType === 'length') {
    hint = `The word has ${secret.length} letters`;
    type = 'length';
  } else {
    hint = `The word starts with "${secret[0].toUpperCase()}"`;
    type = 'first_letter';
  }

  const response: HintResponse = { hint, hintType: type };
  return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
}

/**
 * Handle GET /reveal
 */
async function handleReveal(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const gameIndex = parseInt(url.searchParams.get('game') || '0', 10);

  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  const response: RevealResponse = {
    word: secret,
    message: `The secret word was: ${secret.toUpperCase()}`,
  };
  return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
}

/**
 * Handle POST /llm-hint - Get a vague hint from Claude based on top guesses
 */
async function handleLLMHint(request: Request, env: Env): Promise<Response> {
  // Parse request body
  let body: LLMHintRequest;
  try {
    body = await request.json() as LLMHintRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' } as ErrorResponse, 400, request, env);
  }

  // Validate
  if (!body.topGuesses || !Array.isArray(body.topGuesses) || body.topGuesses.length === 0) {
    return jsonResponse({ error: 'Missing or empty topGuesses array' } as ErrorResponse, 400, request, env);
  }

  // Get game index
  const gameIndex = body.game ?? 0;

  // Check budget before making LLM call
  const canProceed = await checkAndIncrementCost(env, ESTIMATED_COST_PER_HINT_CENTS);
  if (!canProceed) {
    const response: LLMHintResponse = {
      hint: 'Hint unavailable (rate limited)',
      rateLimited: true,
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
  }

  // Get secret word
  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  // Build the prompt
  const topWords = body.topGuesses.slice(0, 5).map(g => g.word).join(', ');

  const prompt = `You are helping with a word guessing game. The secret word is "${secret}".

The player's top guesses so far are: ${topWords}

Give them ONE extremely vague hint to nudge them in the right direction. The hint must be:
- Very general and indirect (NOT obvious)
- One short sentence only
- Do NOT use the secret word or any close synonyms
- Do NOT say "think about X" or "consider Y"
- Be cryptic but fair

Example good hints for "ocean": "Vast and blue, sailors know it well." or "Where waves meet the shore."
Example bad hints: "It's a large body of water" (too obvious) or "Think about the sea" (too direct)

Your hint:`;

  try {
    const response = await fetch(OPENROUTER_CHAT_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: HINT_MODEL,
        messages: [
          { role: 'user', content: prompt }
        ],
        temperature: 0.7,
        max_tokens: 100,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`LLM API error: ${response.status} - ${text}`);
    }

    const data = await response.json() as {
      choices: Array<{ message: { content: string } }>;
    };

    const hint = data.choices[0]?.message?.content?.trim() || 'No hint available';

    const hintResponse: LLMHintResponse = {
      hint,
      rateLimited: false,
    };
    return jsonResponse(hintResponse as unknown as ScoreResponse, 200, request, env);

  } catch (error) {
    console.error('Error in LLM hint:', error);
    return jsonResponse(
      { error: `LLM hint error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500,
      request,
      env
    );
  }
}

/**
 * Handle POST /rerank - Re-rank top guesses using LLM
 */
async function handleRerank(request: Request, env: Env): Promise<Response> {
  // Parse request body
  let body: RerankRequest;
  try {
    body = await request.json() as RerankRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' } as ErrorResponse, 400, request, env);
  }

  // Validate guesses
  if (!body.guesses || !Array.isArray(body.guesses) || body.guesses.length === 0) {
    return jsonResponse({ error: 'Missing or empty guesses array' } as ErrorResponse, 400, request, env);
  }

  // Limit to top 20
  const topGuesses = body.guesses.slice(0, 20);

  // Get game index
  const gameIndex = body.game ?? 0;

  // Check budget before making LLM call
  const canProceed = await checkAndIncrementCost(env);
  if (!canProceed) {
    // Return rate limited response - frontend should use embedding scores
    const response: RerankResponse = {
      rankings: topGuesses.map((g, idx) => ({
        word: g.word,
        rank: idx + 1,
      })),
      rateLimited: true,
      message: 'LLM rate limited due to hourly budget. Using embedding scores.',
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
  }

  // Get secret word for LLM ranking
  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  try {
    const rankings = await rerankWithLLM(topGuesses, secret, env);

    const response: RerankResponse = {
      rankings,
      rateLimited: false,
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
  } catch (error) {
    console.error('Error in LLM rerank:', error);
    // Return error - don't populate with fallback rankings
    return jsonResponse(
      { error: `LLM error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500,
      request,
      env
    );
  }
}

/**
 * Create JSON response with CORS headers
 */
function jsonResponse(
  data: ScoreResponse | ErrorResponse | HintResponse | RevealResponse,
  status = 200,
  request?: Request,
  env?: Env
): Response {
  const corsHeaders = request && env ? getCorsHeaders(request, env) : {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };

  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders,
    },
  });
}

/**
 * Handle CORS preflight requests
 */
function handleOptions(request: Request, env: Env): Response {
  const corsHeaders = getCorsHeaders(request, env);
  return new Response(null, {
    status: 204,
    headers: {
      ...corsHeaders,
    },
  });
}

/**
 * Main Worker entry point
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return handleOptions(request, env);
    }

    // Route requests
    if (url.pathname === '/score' && request.method === 'POST') {
      return handleScore(request, env);
    }

    // Hint endpoint
    if (url.pathname === '/hint' && request.method === 'GET') {
      return handleHint(request, env);
    }

    // Reveal endpoint
    if (url.pathname === '/reveal' && request.method === 'GET') {
      return handleReveal(request, env);
    }

    // Rerank endpoint (LLM-based re-ranking of top guesses)
    if (url.pathname === '/rerank' && request.method === 'POST') {
      return handleRerank(request, env);
    }

    // LLM hint endpoint
    if (url.pathname === '/llm-hint' && request.method === 'POST') {
      return handleLLMHint(request, env);
    }

    // Health check endpoint
    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({ status: 'ok' } as unknown as ScoreResponse, 200, request, env);
    }

    // 404 for unknown routes
    return jsonResponse({ error: 'Not found' }, 404, request, env);
  },
};
