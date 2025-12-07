#!/usr/bin/env node
// Probe the current scoring algorithm against live OpenRouter embeddings.
// Usage: OPENROUTER_API_KEY=sk-or-... node scripts/score_probe.js

const API_KEY = process.env.OPENROUTER_API_KEY;
if (!API_KEY) {
  console.error('Missing OPENROUTER_API_KEY. Export it before running.');
  process.exit(1);
}

// Match Worker defaults
const ENSEMBLE_MODELS = [
  { model: 'google/gemini-embedding-001', weight: 0.7 },
  { model: 'thenlper/gte-base', weight: 0.3 },
];

// Default scoring parameters from worker/src/index.ts
const CURVES = [
  { name: 'old-current', min: 0.10, max: 0.80, curve: 1.75 },
  { name: 'balanced', min: 0.10, max: 0.85, curve: 2.5 },
];

// Piecewise mapping matching worker/src/index.ts
const SCORE_POINTS = [
  { sim: 0.10, score: 0 },
  { sim: 0.40, score: 5 },
  { sim: 0.50, score: 20 },
  { sim: 0.60, score: 45 },
  { sim: 0.66, score: 65 },
  { sim: 0.72, score: 95 },
  { sim: 0.86, score: 100 },
];

// Quick test set; edit as needed
const TESTS = [
  {
    secret: 'nines',
    guesses: ['human', 'number', 'nine', 'digits', 'cat'],
  },
  {
    secret: 'enforcement',
    guesses: ['tiger', 'human', 'man', 'number', 'nine', 'science', 'red', 'green', 'blasphemy', 'encryption', 'particle', 'aphrodisiac', 'burning'],
  },
  {
    secret: 'bridge',
    guesses: ['connection', 'river', 'fire', 'ladder', 'train'],
  },
  {
    secret: 'happy',
    guesses: ['joy', 'sad', 'knife', 'smile', 'pain'],
  },
];

const cache = new Map();

async function fetchEmbedding(word, model) {
  const key = `${model}:${word}`;
  if (cache.has(key)) return cache.get(key);

  const response = await fetch('https://openrouter.ai/api/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model, input: word }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Embedding error ${response.status}: ${text}`);
  }

  const data = await response.json();
  const embedding = data.data[0].embedding;
  cache.set(key, embedding);
  return embedding;
}

function cosineSimilarity(a, b) {
  if (a.length !== b.length) throw new Error('Vector length mismatch');
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function similarityToScore(similarity, { min, max, curve }) {
  if (similarity >= max) return 100;
  if (similarity <= min) return 0;
  const normalized = (similarity - min) / (max - min);
  const curved = Math.pow(normalized, curve);
  const score = Math.round(curved * 100);
  if (score < 0) return 0;
  if (score > 100) return 100;
  return score;
}

function scorePiecewise(similarity) {
  if (similarity <= SCORE_POINTS[0].sim) return 0;
  if (similarity >= SCORE_POINTS[SCORE_POINTS.length - 1].sim) return 100;
  for (let i = 0; i < SCORE_POINTS.length - 1; i++) {
    const a = SCORE_POINTS[i];
    const b = SCORE_POINTS[i + 1];
    if (similarity >= a.sim && similarity <= b.sim) {
      const t = (similarity - a.sim) / (b.sim - a.sim);
      return Math.round(a.score + t * (b.score - a.score));
    }
  }
  return 0;
}

async function evaluatePair(secret, guess) {
  // Launch both models in parallel to measure which would win the race
  const tasks = ENSEMBLE_MODELS.map(async ({ model, weight }) => {
    const t0 = performance.now();
    const [a, b] = await Promise.all([
      fetchEmbedding(secret, model),
      fetchEmbedding(guess, model),
    ]);
    const elapsedMs = performance.now() - t0;
    const similarity = cosineSimilarity(a, b);
    return { model, weight, similarity, elapsedMs };
  });

  const results = await Promise.all(tasks);
  const winner = results.reduce((best, r) => (r.elapsedMs < best.elapsedMs ? r : best), results[0]);
  const weightedSim = results.reduce((sum, r) => sum + r.similarity * r.weight, 0);

  // Compute scores for each curve option
  const perCurve = CURVES.map(cfg => ({
    name: cfg.name,
    raceScore: similarityToScore(winner.similarity, cfg),
    perModel: results.map(r => ({ model: r.model, score: similarityToScore(r.similarity, cfg) })),
    weightedScore: similarityToScore(weightedSim, cfg),
  }));

  const piecewise = {
    name: 'piecewise (worker)',
    raceScore: scorePiecewise(winner.similarity),
    perModel: results.map(r => ({ model: r.model, score: scorePiecewise(r.similarity) })),
    weightedScore: scorePiecewise(weightedSim),
  };

  return { results, winner, perCurve, piecewise, weightedSim };
}

async function main() {
  for (const test of TESTS) {
    console.log(`\nSecret: ${test.secret}`);
    for (const guess of test.guesses) {
      try {
        const { results, winner, perCurve, piecewise, weightedSim } = await evaluatePair(test.secret, guess);
        const sims = results.map(r => `${r.model}: ${r.similarity.toFixed(3)} (${r.elapsedMs.toFixed(0)}ms)`).join(' | ');
        console.log(`  Guess: ${guess}`);
        console.log(`    Sims: ${sims} | weighted: ${weightedSim.toFixed(3)}`);
        perCurve.forEach(cfg => {
          const modelScores = cfg.perModel.map(m => `${m.model}:${m.score}`).join(', ');
          console.log(`    Curve ${cfg.name}: race=${cfg.raceScore} | weighted=${cfg.weightedScore} | ${modelScores}`);
        });
        const pwScores = piecewise.perModel.map(m => `${m.model}:${m.score}`).join(', ');
        console.log(`    ${piecewise.name}: race=${piecewise.raceScore} | weighted=${piecewise.weightedScore} | ${pwScores}`);
      } catch (err) {
        console.error(`  Guess: ${guess} -> error: ${err.message}`);
      }
    }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
