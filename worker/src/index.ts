/**
 * Semantic Trail Backend - Cloudflare Worker
 *
 * Exposes POST /score endpoint for the word guessing game.
 * Uses Google Gemini embedding only (no ensemble).
 */

import { WORD_LIST } from './wordlist';

// Single embedding model (Gemini)
const EMBEDDING_MODEL = 'google/gemini-embedding-001';

// Stop words that cannot be guessed
const STOP_WORDS = new Set([
  'a', 'an', 'the',
  'i', 'you', 'he', 'she', 'it', 'we', 'they',
  'me', 'him', 'her', 'us', 'them',
  'and', 'or', 'but', 'if', 'then', 'so',
  'for', 'nor', 'yet',
  'to', 'of', 'in', 'on', 'at', 'by', 'as',
  'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
  'this', 'that', 'these', 'those',
  'my', 'your', 'his', 'its', 'our', 'their',
  'what', 'which', 'who', 'whom', 'whose',
  'do', 'does', 'did', 'have', 'has', 'had',
  'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
  'not', 'no', 'yes', 'all', 'any', 'some', 'each', 'every',
  'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
  'here', 'there', 'when', 'where', 'why', 'how',
  'both', 'few', 'more', 'most', 'other', 'such', 'only', 'own', 'same',
  'than', 'too', 'very', 'just', 'now',
]);

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings';
const OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions';

// LLM for re-ranking and question parsing
const RERANK_MODEL = 'google/gemini-2.5-flash';

// LLM for hints
const HINT_MODEL = 'anthropic/claude-sonnet-4.5';

// LLM for answering questions in 20 Questions mode
const QUESTIONS_ANSWER_MODEL = 'deepseek/deepseek-v3.2-exp';

// Cost protection: $2/hour budget
const HOURLY_BUDGET_CENTS = 200; // $2.00 in cents
// Gemini 2.5 Flash pricing (approximate): $0.15/1M input, $0.60/1M output
// Estimate ~500 tokens per rerank call, ~$0.0003 per call = 0.03 cents
const ESTIMATED_COST_PER_RERANK_CENTS = 0.05; // Conservative estimate
// Claude Sonnet 4.5 pricing: ~$3/1M input, $15/1M output
// Estimate ~300 tokens per hint call = ~0.5 cents
const ESTIMATED_COST_PER_HINT_CENTS = 0.5;

// Questions mode cost estimates
// Whisper pricing: $0.006 per minute of audio, 10s = ~$0.001 = 0.1 cents
const ESTIMATED_COST_PER_TRANSCRIBE_CENTS = 0.15;
// Gemini 2.5 Flash for question parsing: ~200 tokens = ~0.02 cents
const ESTIMATED_COST_PER_PARSE_CENTS = 0.03;
// Gemini 2.5 Flash for question answering: ~300 tokens = ~0.03 cents
const ESTIMATED_COST_PER_ANSWER_CENTS = 0.05;

// OpenAI Transcription API (GPT-4o-mini-transcribe)
const OPENAI_TRANSCRIPTION_URL = 'https://api.openai.com/v1/audio/transcriptions';
const TRANSCRIPTION_MODEL = 'gpt-4o-mini-transcribe';

// Similarity to score mapping (non-linear)
// Piecewise similarity->score mapping tuned for numeric words (less generous)
// Control points are linearly interpolated; adjust to retune.
// Control points tuned for Gemini-only similarities
const SCORE_POINTS: Array<{ sim: number; score: number }> = [
  { sim: 0.10, score: 0 },
  { sim: 0.40, score: 10 },
  { sim: 0.50, score: 25 },
  { sim: 0.60, score: 50 },
  { sim: 0.68, score: 75 },
  { sim: 0.75, score: 95 },
  { sim: 0.82, score: 100 },
];

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
  EMBEDDINGS_BUCKET?: R2Bucket; // Optional R2 bucket for precomputed embeddings
  OPENROUTER_API_KEY: string;
  OPENAI_API_KEY?: string; // For Whisper transcription (Questions mode)
  SECRET_SALT: string;
  ALLOWED_ORIGINS?: string;
}

// (R2 caches were for GTE-base; kept for future reuse but unused by Gemini-only path)
let r2EmbeddingsLoaded = false;
let r2WordIndex: Record<string, number> | null = null;
let r2EmbeddingsData: Float32Array | null = null;
const EMBEDDING_DIM = 768;

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

// === Questions Mode Types ===

type QuestionAnswer = 'yes' | 'no' | 'maybe' | 'so close' | 'N/A';

interface AskRequest {
  audioBase64?: string;
  mimeType?: string;
  textQuestions?: string[];
  previousTranscript?: string;  // Previous buffer's transcript for context
  recentQuestions?: string[];   // Last 10 questions for duplicate detection
  game?: number;
}

interface AskResponse {
  transcribedText?: string;
  answers: Array<{
    question: string;
    answer: QuestionAnswer;
  }>;
  rateLimited?: boolean;
  won?: boolean;
  secretWord?: string;
  error?: string;
}

interface QuestionsHintRequest {
  recentQA: Array<{ question: string; answer: string }>;
  game?: number;
}

interface QuestionsHintResponse {
  hint: string;
  rateLimited?: boolean;
}

// === Similarity Mode Transcription Types ===

interface TranscribeGuessRequest {
  audioBase64: string;
  mimeType: string;
}

interface TranscribeGuessResponse {
  transcribedText?: string;
  words: string[];
  rateLimited?: boolean;
  error?: string;
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
 * Convert cosine similarity to 0-100 score using piecewise linear control points.
 * Keeps monotonicity while letting us tune specific regions (e.g., numeric words).
 */
function similarityToScore(similarity: number): number {
  // Clamp below/above
  if (similarity <= SCORE_POINTS[0].sim) return 0;
  if (similarity >= SCORE_POINTS[SCORE_POINTS.length - 1].sim) return 100;

  // Find the segment and linearly interpolate
  for (let i = 0; i < SCORE_POINTS.length - 1; i++) {
    const a = SCORE_POINTS[i];
    const b = SCORE_POINTS[i + 1];
    if (similarity >= a.sim && similarity <= b.sim) {
      const t = (similarity - a.sim) / (b.sim - a.sim);
      return Math.round(a.score + t * (b.score - a.score));
    }
  }

  // Fallback (should not hit)
  return 0;
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
 * Get embedding from cache or fetch from OpenRouter (Gemini only).
 */
async function getEmbedding(
  word: string,
  env: Env
): Promise<number[]> {
  const model = EMBEDDING_MODEL;
  const cacheKey = getCacheKey(word, model);

  // Try cache first
  try {
    const cached = await env.EMBED_CACHE.get(cacheKey, 'json');
    if (cached) {
      return cached as number[];
    }
  } catch (e) {
    console.warn('KV get failed, continuing without cache:', e);
  }

  // Fetch from OpenRouter
  const response = await fetch(OPENROUTER_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      input: word,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenRouter API error (${model}): ${response.status} - ${text}`);
  }

  const data = await response.json() as { data: Array<{ embedding: number[] }> };
  const embedding = data.data[0].embedding;

  // Try to cache for future use (non-fatal if it fails)
  try {
    await env.EMBED_CACHE.put(cacheKey, JSON.stringify(embedding));
  } catch (e) {
    console.warn('KV put failed, continuing without cache:', e);
  }

  return embedding;
}

/**
 * Load precomputed embeddings from R2 (called once per worker instance)
 */
async function loadR2Embeddings(env: Env): Promise<boolean> {
  if (r2EmbeddingsLoaded) return r2WordIndex !== null;
  r2EmbeddingsLoaded = true;

  if (!env.EMBEDDINGS_BUCKET) {
    console.log('R2 bucket not configured, using API fallback');
    return false;
  }

  try {
    // Load word index
    const indexObj = await env.EMBEDDINGS_BUCKET.get('embeddings-index.json');
    if (!indexObj) {
      console.warn('R2: embeddings-index.json not found');
      return false;
    }
    r2WordIndex = await indexObj.json();

    // Load binary embeddings
    const embObj = await env.EMBEDDINGS_BUCKET.get('embeddings.bin');
    if (!embObj) {
      console.warn('R2: embeddings.bin not found');
      r2WordIndex = null;
      return false;
    }
    const arrayBuffer = await embObj.arrayBuffer();
    r2EmbeddingsData = new Float32Array(arrayBuffer);

    console.log(`R2: Loaded ${Object.keys(r2WordIndex).length} word embeddings`);
    return true;
  } catch (e) {
    console.error('Failed to load R2 embeddings:', e);
    r2WordIndex = null;
    r2EmbeddingsData = null;
    return false;
  }
}

/**
// NOTE: R2 embedding path disabled for Gemini-only mode
function getR2Embedding(_word: string): number[] | null {
  if (!r2WordIndex || !r2EmbeddingsData) return null;

  const wordLower = word.toLowerCase();
  const index = r2WordIndex[wordLower];
  if (index === undefined) return null;

  const start = index * EMBEDDING_DIM;
  const end = start + EMBEDDING_DIM;
  return Array.from(r2EmbeddingsData.slice(start, end));
}

/**
 * Get similarity between two words using the Gemini embedding.
 */
async function getSimilarity(
  word1: string,
  word2: string,
  env: Env
): Promise<number> {
  const [emb1, emb2] = await Promise.all([
    getEmbedding(word1, env),
    getEmbedding(word2, env),
  ]);
  return cosineSimilarity(emb1, emb2);
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

  try {
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
  } catch (e) {
    // If KV fails (e.g., daily limit exceeded), allow the request but log warning
    console.warn('Cost tracking failed (KV error), allowing request:', e);
    return true;
  }
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

// === Questions Mode Functions ===

/**
 * Transcribe audio using OpenAI Whisper API
 */
async function transcribeAudio(
  audioBase64: string,
  mimeType: string,
  env: Env
): Promise<string> {
  if (!env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY not configured');
  }

  // Convert base64 to binary
  const binaryString = atob(audioBase64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Determine file extension from mime type
  const ext = mimeType.includes('webm') ? 'webm' : mimeType.includes('mp4') ? 'mp4' : 'webm';

  // Create form data for Whisper API
  const formData = new FormData();
  const blob = new Blob([bytes], { type: mimeType });
  formData.append('file', blob, `audio.${ext}`);
  formData.append('model', TRANSCRIPTION_MODEL);
  formData.append('language', 'en');

  const response = await fetch(OPENAI_TRANSCRIPTION_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENAI_API_KEY}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Whisper API error: ${response.status} - ${text}`);
  }

  const data = await response.json() as { text: string };
  return data.text;
}

/**
 * Parse natural language text into discrete questions using DeepSeek
 */
async function parseQuestionsWithLLM(text: string, env: Env): Promise<string[]> {
  const prompt = `Extract all questions from this transcribed speech.
Return ONLY questions that are asking about properties or characteristics of something (yes/no questions).
Ignore statements, commands, or off-topic content.
Clean up any transcription errors to make proper questions.

Transcribed text: "${text}"

Respond with a JSON object with a "questions" array containing the extracted questions.
If no valid questions found, return {"questions": []}

Example input: "is it an animal um does it have four legs what about can it fly"
Example output: {"questions": ["Is it an animal?", "Does it have four legs?", "Can it fly?"]}`;

  const response = await fetch(OPENROUTER_CHAT_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: QUESTIONS_ANSWER_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 500,
      response_format: { type: 'json_object' },
    }),
  });

  if (!response.ok) {
    console.error('Parse questions LLM error:', await response.text());
    return [];
  }

  const data = await response.json() as { choices: Array<{ message: { content: string } }> };
  const content = data.choices[0]?.message?.content || '{"questions":[]}';

  try {
    // Handle potential markdown code blocks
    let jsonStr = content.trim();
    if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
    }
    const parsed = JSON.parse(jsonStr);
    return parsed.questions || [];
  } catch {
    console.error('Failed to parse questions JSON:', content);
    return [];
  }
}

/**
 * Check if a question is a duplicate of recently asked questions using Gemini
 */
async function checkDuplicateQuestion(
  newQuestion: string,
  recentQuestions: string[],
  env: Env
): Promise<boolean> {
  if (recentQuestions.length === 0) return false;

  const recentList = recentQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n');

  const prompt = `Determine if the new question is essentially asking the same thing as any of the recent questions.
Consider semantic similarity - questions worded differently but asking the same thing ARE duplicates.

Recent questions:
${recentList}

New question: "${newQuestion}"

Is this new question a duplicate of any recent question?
Respond with ONLY a JSON object: {"isDuplicate": true} or {"isDuplicate": false}`;

  const response = await fetch(OPENROUTER_CHAT_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: RERANK_MODEL, // Gemini 2.5 Flash for duplicate detection
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 50,
      response_format: { type: 'json_object' },
    }),
  });

  if (!response.ok) {
    console.error('Duplicate check LLM error:', await response.text());
    return false; // On error, assume not duplicate
  }

  const data = await response.json() as { choices: Array<{ message: { content: string } }> };
  const content = data.choices[0]?.message?.content || '{"isDuplicate":false}';

  try {
    let jsonStr = content.trim();
    if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
    }
    const parsed = JSON.parse(jsonStr);
    return parsed.isDuplicate === true;
  } catch {
    console.error('Failed to parse duplicate check JSON:', content);
    return false;
  }
}

/**
 * Answer a single question about the secret word using Gemini
 * Returns the answer and whether the player won
 */
async function answerQuestionWithLLM(
  question: string,
  secret: string,
  env: Env
): Promise<{ answer: QuestionAnswer; won: boolean }> {
  // Check if question directly asks about the secret word
  const questionLower = question.toLowerCase();
  const secretLower = secret.toLowerCase();

  // Check for win condition: "is it [word]?" pattern
  const isItPattern = /is\s+it\s+(?:a\s+|an\s+|the\s+)?(\w+)\??$/i;
  const match = questionLower.match(isItPattern);
  if (match && match[1] === secretLower) {
    return { answer: 'yes', won: true };
  }

  const prompt = `You are playing a word guessing game (like 20 Questions). The secret word is "${secret}".

The player asks: "${question}"

IMPORTANT: For questions about the WORD ITSELF (spelling, letters, length, alphabet position), think step-by-step:
- First letter of "${secret}" is "${secret[0].toUpperCase()}"
- Word length is ${secret.length} letters
- The first half of the alphabet is A-M (letters 1-13), second half is N-Z (letters 14-26)
- "${secret[0].toUpperCase()}" is letter #${'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.indexOf(secret[0].toUpperCase()) + 1} of 26

Answer with ONLY one of these responses:
- "yes" - if the answer is clearly yes
- "no" - if the answer is clearly no
- "maybe" - if the answer is uncertain, partially true, or depends on context
- "so close" - ONLY if the question reveals they are VERY close to guessing (e.g., asking about a very specific category the word belongs to, or a near-synonym)
- "N/A" - if the question cannot be answered with yes/no, is unclear, or is not relevant

Be accurate and precise. For factual questions, verify your answer carefully.

Respond with ONLY a JSON object: {"answer": "your_answer_here"}`;

  const response = await fetch(OPENROUTER_CHAT_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: QUESTIONS_ANSWER_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 50,
      response_format: { type: 'json_object' },
    }),
  });

  if (!response.ok) {
    console.error('Answer question LLM error:', await response.text());
    return { answer: 'N/A', won: false };
  }

  const data = await response.json() as { choices: Array<{ message: { content: string } }> };
  const content = data.choices[0]?.message?.content || '{"answer":"N/A"}';

  try {
    let jsonStr = content.trim();
    if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
    }
    const parsed = JSON.parse(jsonStr);
    const answerRaw = (parsed.answer || '').toString().toLowerCase().trim();

    // Normalize various formats DeepSeek might return
    let answer: QuestionAnswer;
    if (answerRaw === 'yes') answer = 'yes';
    else if (answerRaw === 'no') answer = 'no';
    else if (answerRaw === 'maybe') answer = 'maybe';
    else if (answerRaw === 'so close' || answerRaw === 'soclose') answer = 'so close';
    else answer = 'N/A';

    return { answer, won: false };
  } catch {
    console.error('Failed to parse answer JSON:', content);
    return { answer: 'N/A', won: false };
  }
}

/**
 * Handle POST /ask - Combined endpoint for Questions mode
 * Handles transcription, parsing, duplicate detection, and answering
 */
async function handleAsk(request: Request, env: Env): Promise<Response> {
  let body: AskRequest;
  try {
    body = await request.json() as AskRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' } as ErrorResponse, 400, request, env);
  }

  const gameIndex = body.game ?? 0;
  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);
  const recentQuestions = body.recentQuestions || [];

  const allQuestions: string[] = [...(body.textQuestions || [])];
  let transcribedText: string | undefined;
  let rateLimited = false;
  let won = false;
  let secretWord: string | undefined;

  // Step 1: If audio provided, transcribe it
  if (body.audioBase64 && body.mimeType) {
    const canTranscribe = await checkAndIncrementCost(env, ESTIMATED_COST_PER_TRANSCRIBE_CENTS);
    if (!canTranscribe) {
      rateLimited = true;
    } else {
      try {
        transcribedText = await transcribeAudio(body.audioBase64, body.mimeType, env);
      } catch (e) {
        console.error('Transcription error:', e);
        // Continue without transcription, still process text questions
      }
    }
  }

  // Step 2: Parse transcribed text into questions
  // Combine with previous transcript for better context
  if (transcribedText && transcribedText.trim() && !rateLimited) {
    const canParse = await checkAndIncrementCost(env, ESTIMATED_COST_PER_PARSE_CENTS);
    if (canParse) {
      try {
        // Combine previous transcript with current for more context
        const combinedText = body.previousTranscript
          ? `${body.previousTranscript} ${transcribedText}`
          : transcribedText;
        const parsedQuestions = await parseQuestionsWithLLM(combinedText, env);
        allQuestions.push(...parsedQuestions);
      } catch (e) {
        console.error('Parse questions error:', e);
      }
    } else {
      rateLimited = true;
    }
  }

  // Step 3: Answer each question (with duplicate detection)
  const answers: Array<{ question: string; answer: QuestionAnswer }> = [];

  for (const question of allQuestions) {
    if (won) {
      // Already won, skip remaining questions
      break;
    }

    // Step 3a: Check for duplicate question using Gemini
    const canCheckDuplicate = await checkAndIncrementCost(env, ESTIMATED_COST_PER_PARSE_CENTS);
    if (canCheckDuplicate) {
      try {
        const isDuplicate = await checkDuplicateQuestion(question, recentQuestions, env);
        if (isDuplicate) {
          console.log(`Skipping duplicate question: "${question}"`);
          continue; // Skip this question, don't answer it
        }
      } catch (e) {
        console.error('Duplicate check error:', e);
        // On error, continue to answer the question
      }
    }

    // Step 3b: Answer the question
    const canAnswer = await checkAndIncrementCost(env, ESTIMATED_COST_PER_ANSWER_CENTS);
    if (!canAnswer) {
      rateLimited = true;
      break;
    }

    try {
      const result = await answerQuestionWithLLM(question, secret, env);
      answers.push({ question, answer: result.answer });

      if (result.won) {
        won = true;
        secretWord = secret;
      }
    } catch (e) {
      console.error('Answer question error:', e);
      answers.push({ question, answer: 'N/A' });
    }
  }

  const response: AskResponse = {
    transcribedText,
    answers,
    rateLimited,
    won,
    secretWord,
  };

  return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
}

/**
 * Handle POST /questions-hint - Get a vague hint for Questions mode based on Q&A history
 */
async function handleQuestionsHint(request: Request, env: Env): Promise<Response> {
  let body: QuestionsHintRequest;
  try {
    body = await request.json() as QuestionsHintRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' } as ErrorResponse, 400, request, env);
  }

  if (!body.recentQA || !Array.isArray(body.recentQA)) {
    return jsonResponse({ error: 'Missing recentQA array' } as ErrorResponse, 400, request, env);
  }

  const gameIndex = body.game ?? 0;

  // Check budget before making LLM call
  const canProceed = await checkAndIncrementCost(env, ESTIMATED_COST_PER_HINT_CENTS);
  if (!canProceed) {
    const response: QuestionsHintResponse = {
      hint: 'Hint unavailable (rate limited)',
      rateLimited: true,
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
  }

  const secret = await getSecretWord(env.SECRET_SALT, gameIndex);

  // Build Q&A summary for the prompt
  const qaList = body.recentQA.slice(-10).map(qa => `Q: ${qa.question} → ${qa.answer}`).join('\n');

  const prompt = `You are playing 20 Questions. The secret word is "${secret}".

Here are the player's recent questions and answers:
${qaList}

Your job: Give ONE cryptic, almost-useless hint that gently redirects them. They need a nudge in the right direction.

RULES (strict):
- Maximum 5-7 words
- Be EXTREMELY vague and abstract
- Do NOT describe what the word IS or DOES directly
- Do NOT use synonyms, rhymes, or obvious category words
- Reference a distant association, mood, or abstract quality only
- Consider what they've learned from YES/NO answers so far

EXAMPLES of good hints (cryptic, poetic, indirect):
- "Think broader, not deeper"
- "The answer hides in plain sight"
- "Less abstract, more tangible"
- "Consider what connects them all"

Use their Q&A history to gently redirect without giving it away.

Hint:`;

  try {
    const response = await fetch(OPENROUTER_CHAT_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: HINT_MODEL,
        messages: [{ role: 'user', content: prompt }],
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

    const hintResponse: QuestionsHintResponse = {
      hint,
      rateLimited: false,
    };
    return jsonResponse(hintResponse as unknown as ScoreResponse, 200, request, env);

  } catch (error) {
    console.error('Error in Questions hint:', error);
    return jsonResponse(
      { error: `LLM hint error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500,
      request,
      env
    );
  }
}

/**
 * Handle POST /transcribe-guess - Transcribe audio and extract word guesses for Similarity mode
 */
async function handleTranscribeGuess(request: Request, env: Env): Promise<Response> {
  let body: TranscribeGuessRequest;
  try {
    body = await request.json() as TranscribeGuessRequest;
  } catch {
    return jsonResponse({ error: 'Invalid JSON body' } as ErrorResponse, 400, request, env);
  }

  if (!body.audioBase64 || !body.mimeType) {
    return jsonResponse({ error: 'Missing audioBase64 or mimeType' } as ErrorResponse, 400, request, env);
  }

  // Check budget before transcription
  const canTranscribe = await checkAndIncrementCost(env, ESTIMATED_COST_PER_TRANSCRIBE_CENTS);
  if (!canTranscribe) {
    const response: TranscribeGuessResponse = {
      words: [],
      rateLimited: true,
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);
  }

  try {
    const transcribedText = await transcribeAudio(body.audioBase64, body.mimeType, env);

    // Extract words from transcription (only valid guess words)
    const words: string[] = [];
    if (transcribedText) {
      const rawWords = transcribedText.toLowerCase().split(/\s+/);
      for (const word of rawWords) {
        // Only keep words that are valid guesses (letters only, 3+ chars, not stop words)
        const cleaned = word.replace(/[^a-z]/g, '');
        if (cleaned.length >= 3 && /^[a-z]+$/.test(cleaned) && !STOP_WORDS.has(cleaned)) {
          words.push(cleaned);
        }
      }
    }

    const response: TranscribeGuessResponse = {
      transcribedText,
      words,
      rateLimited: false,
    };
    return jsonResponse(response as unknown as ScoreResponse, 200, request, env);

  } catch (error) {
    console.error('Transcription error:', error);
    return jsonResponse(
      { error: `Transcription error: ${error instanceof Error ? error.message : 'Unknown error'}` },
      500,
      request,
      env
    );
  }
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

  // Minimum length check
  if (guess.length < 3) {
    return jsonResponse({ error: 'Guess must be at least 3 letters' }, 400, request, env);
  }

  // Block stop words
  if (STOP_WORDS.has(guess)) {
    return jsonResponse({ error: 'That word is too common' }, 400, request, env);
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

  // Get similarity using GTE-base
  try {
    const similarity = await getSimilarity(guess, secret, env);
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

  const prompt = `Word guessing game. Secret: "${secret}". Player's guesses: ${topWords}

Your job: Give ONE cryptic, almost-useless hint that gently redirects them. They're on the wrong track.

RULES (strict):
- Maximum 5-7 words
- Be EXTREMELY vague and abstract
- Do NOT describe what the word IS or DOES
- Do NOT use synonyms, rhymes, or category words
- Do NOT say what it's "like" or "related to"
- Reference a distant association, mood, or abstract quality only

EXAMPLES (secret word, their guesses -> good hint):

Secret "sing", guesses "ocean, blue, water, wave, sea":
BAD: "Birds do this" / "Musical expression" (too obvious)
GOOD: "Less wet, more from within" / "Abandon the depths, find your throat"

Secret "ocean", guesses "music, song, voice, melody, sound":
BAD: "Large body of water" / "Where fish swim" (too obvious)
GOOD: "Deeper than any note" / "Scale up from sound to vastness"

Secret "knife", guesses "happy, joy, smile, laugh, love":
BAD: "Sharp cutting tool" / "Found in kitchens" (too obvious)
GOOD: "Edges exist beyond emotion" / "Joy has no point—this does"

Secret "dream", guesses "rock, stone, mountain, earth, ground":
BAD: "What happens when sleeping" (too obvious)
GOOD: "Softer than any stone" / "Close your eyes to move"

Secret "bridge", guesses "fire, heat, flame, burn, hot":
BAD: "Connects two places" / "Crosses water" (too obvious)
GOOD: "Cool the flames, span the gap" / "Not destruction—connection"

Secret "time", guesses "tree, forest, leaf, branch, wood":
BAD: "Clocks measure it" / "Hours and minutes" (too obvious)
GOOD: "Rings mark its passage silently" / "What trees count but cannot keep"

Secret "fire", guesses "cold, ice, winter, snow, freeze":
BAD: "Hot flames" / "Burns things" (too obvious)
GOOD: "The opposite hungers nearby" / "Melt your assumptions"

Use their guesses to gently redirect. Be poetic and obscure.

Hint:`;

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

    // Questions mode endpoint
    if (url.pathname === '/ask' && request.method === 'POST') {
      return handleAsk(request, env);
    }

    // Questions mode hint endpoint
    if (url.pathname === '/questions-hint' && request.method === 'POST') {
      return handleQuestionsHint(request, env);
    }

    // Similarity mode transcription endpoint
    if (url.pathname === '/transcribe-guess' && request.method === 'POST') {
      return handleTranscribeGuess(request, env);
    }

    // Health check endpoint
    if (url.pathname === '/health' && request.method === 'GET') {
      return jsonResponse({ status: 'ok' } as unknown as ScoreResponse, 200, request, env);
    }

    // 404 for unknown routes
    return jsonResponse({ error: 'Not found' }, 404, request, env);
  },
};
