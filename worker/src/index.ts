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

interface Env {
  EMBED_CACHE: KVNamespace;
  OPENROUTER_API_KEY: string;
  SECRET_SALT: string;
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
 * Convert cosine similarity to 0-100 score
 */
function similarityToScore(similarity: number): number {
  if (similarity >= MAX_SIM) return 100;
  if (similarity <= MIN_SIM) return 0;
  return Math.round((similarity - MIN_SIM) / (MAX_SIM - MIN_SIM) * 100);
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
  const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD in UTC
  const hash = await hashString(today + salt);
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
    return jsonResponse({ error: 'Invalid JSON body' }, 400);
  }

  // Validate guess
  const guess = body.guess?.toLowerCase().trim();
  if (!guess) {
    return jsonResponse({ error: 'Missing guess field' }, 400);
  }

  if (!/^[a-z]+$/.test(guess)) {
    return jsonResponse({ error: 'Guess must contain only letters' }, 400);
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
    return jsonResponse(response);
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

    return jsonResponse(response);
  } catch (error) {
    console.error('Error getting similarity:', error);
    return jsonResponse(
      { error: `Failed to compute similarity: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500
    );
  }
}

/**
 * Create JSON response with CORS headers
 */
function jsonResponse(data: ScoreResponse | ErrorResponse, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

/**
 * Handle CORS preflight requests
 */
function handleOptions(): Response {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
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
      return handleOptions();
    }

    // Route requests
    if (url.pathname === '/score' && request.method === 'POST') {
      return handleScore(request, env);
    }

    // Health check endpoint
    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({ status: 'ok' } as unknown as ScoreResponse);
    }

    // 404 for unknown routes
    return jsonResponse({ error: 'Not found' }, 404);
  },
};
