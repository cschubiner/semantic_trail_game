/**
 * Semantic Trail Backend - Cloudflare Worker
 *
 * Exposes POST /score endpoint for the word guessing game.
 * Uses ensemble embeddings with lazy KV caching.
 */

import { WORD_LIST } from './wordlist';

// Ensemble models (top 2 from benchmark)
const ENSEMBLE_MODELS = [
  'openai/text-embedding-3-large',
  'thenlper/gte-large',
];

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings';

// Similarity to score mapping (same as Python version)
const MIN_SIM = 0.20;
const MAX_SIM = 0.85;

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
}

interface ScoreResponse {
  guess: string;
  similarity: number;
  score: number;
  bucket: string;
  rank?: number;
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

/**
 * Get bucket label based on score
 */
function getBucket(score: number): string {
  if (score >= 95) return '🔥 BURNING';
  if (score >= 80) return '🔥 Hot';
  if (score >= 60) return '☀ Warm';
  if (score >= 40) return '〰 Tepid';
  if (score >= 20) return '❄ Cold';
  return '🧊 Freezing';
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
  return Math.round((similarity - MIN_SIM) / (MAX_SIM - MIN_SIM) * 100);
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
 */
async function getSecretWord(salt: string): Promise<string> {
  const gameDay = getGameDateString(); // YYYY-MM-DD in ET with 5am cutoff
  const hash = await hashString(gameDay + salt);
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

  // Get today's secret word
  const secret = await getSecretWord(env.SECRET_SALT);

  // Check for exact match
  if (guess === secret) {
    const response: ScoreResponse = {
      guess,
      similarity: 1.0,
      score: 100,
      bucket: '🎉 CORRECT!',
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
    const bucket = getBucket(score);

    const response: ScoreResponse = {
      guess,
      similarity: Math.round(similarity * 1000) / 1000,
      score,
      bucket,
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

  const secret = await getSecretWord(env.SECRET_SALT);

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
  const secret = await getSecretWord(env.SECRET_SALT);

  const response: RevealResponse = {
    word: secret,
    message: `The secret word was: ${secret.toUpperCase()}`,
  };
  return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
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

    // Health check endpoint
    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({ status: 'ok' } as unknown as ScoreResponse, 200, request, env);
    }

    // 404 for unknown routes
    return jsonResponse({ error: 'Not found' }, 404, request, env);
  },
};
