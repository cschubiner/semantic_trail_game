#!/usr/bin/env node
/**
 * Embedding Generator Script (Node.js)
 *
 * Generates GTE-base embeddings for all words in the word list.
 * Run with: node scripts/generate-embeddings.js
 *
 * Set OPENROUTER_API_KEY in .env file or as environment variable
 *
 * Output: data/embeddings.bin + data/embeddings-index.json
 */

const fs = require('fs');
const path = require('path');

// Load .env file if it exists
const envPath = path.join(__dirname, '../.env');
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8');
  for (const line of envContent.split('\n')) {
    const match = line.match(/^\s*([^#][^=]*?)\s*=\s*(.*)$/);
    if (match && !process.env[match[1]]) {
      process.env[match[1]] = match[2].trim();
    }
  }
  console.log('Loaded .env file');
}

// Read word list from TypeScript file
const wordlistPath = path.join(__dirname, '../src/wordlist.ts');
const wordlistContent = fs.readFileSync(wordlistPath, 'utf-8');
const wordListMatch = wordlistContent.match(/export const WORD_LIST[^=]*=\s*\[([\s\S]*?)\];/);
if (!wordListMatch) {
  console.error('Could not parse word list from wordlist.ts');
  process.exit(1);
}
const WORD_LIST = wordListMatch[1]
  .split('\n')
  .map(line => line.match(/'([^']+)'/))
  .filter(Boolean)
  .map(m => m[1]);

console.log(`Parsed ${WORD_LIST.length} words from wordlist.ts`);

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings';
const EMBEDDING_MODEL = 'thenlper/gte-base';
const EMBEDDING_DIM = 768;
const BATCH_SIZE = 100;  // Larger batches
const CONCURRENCY = 5;   // Parallel requests
const DELAY_BETWEEN_WAVES_MS = 200;  // Delay between concurrent waves

async function getEmbeddings(words, apiKey) {
  const response = await fetch(OPENROUTER_API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: words,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`OpenRouter API error: ${response.status} - ${text}`);
  }

  const data = await response.json();
  return data.data.map(d => d.embedding);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('Error: OPENROUTER_API_KEY environment variable is required');
    console.error('Usage: OPENROUTER_API_KEY=your-key node scripts/generate-embeddings.js');
    process.exit(1);
  }

  console.log(`\nGenerating embeddings for ${WORD_LIST.length} words...`);
  console.log(`Model: ${EMBEDDING_MODEL}`);
  console.log(`Embedding dimension: ${EMBEDDING_DIM}`);
  console.log(`Batch size: ${BATCH_SIZE}, Concurrency: ${CONCURRENCY}`);

  // Results stored by batch index to maintain order
  const batchResults = new Map();

  // Process in batches
  const batches = [];
  for (let i = 0; i < WORD_LIST.length; i += BATCH_SIZE) {
    batches.push({ index: batches.length, words: WORD_LIST.slice(i, i + BATCH_SIZE) });
  }

  console.log(`\nProcessing ${batches.length} batches with ${CONCURRENCY}x parallelization...`);
  const startTime = Date.now();
  let completedBatches = 0;

  // Process a single batch with retry
  async function processBatch(batch, retries = 3) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const embeddings = await getEmbeddings(batch.words, apiKey);
        return { index: batch.index, words: batch.words, embeddings };
      } catch (error) {
        if (attempt === retries) {
          console.error(`\nBatch ${batch.index + 1} failed after ${retries} attempts: ${error.message}`);
          return { index: batch.index, words: batch.words, embeddings: null, error: error.message };
        }
        await sleep(1000 * attempt); // Exponential backoff
      }
    }
  }

  // Process batches in waves of CONCURRENCY
  for (let wave = 0; wave < batches.length; wave += CONCURRENCY) {
    const waveBatches = batches.slice(wave, wave + CONCURRENCY);

    // Process wave in parallel
    const results = await Promise.all(waveBatches.map(b => processBatch(b)));

    // Store results
    for (const result of results) {
      if (result.embeddings) {
        batchResults.set(result.index, result);
      }
      completedBatches++;
    }

    // Progress update
    const progress = (completedBatches / batches.length * 100).toFixed(1);
    const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    const rate = completedBatches / ((Date.now() - startTime) / 1000);
    const remaining = batches.length - completedBatches;
    const eta = rate > 0 ? (remaining / rate / 60).toFixed(1) : '?';

    process.stdout.write(`\rBatches: ${completedBatches}/${batches.length} (${progress}%) - Elapsed: ${elapsed}m, ETA: ${eta}m, Rate: ${(rate * 60).toFixed(0)} batches/min    `);

    // Small delay between waves to avoid rate limiting
    if (wave + CONCURRENCY < batches.length) {
      await sleep(DELAY_BETWEEN_WAVES_MS);
    }
  }

  // Build final arrays in correct order
  const wordIndex = {};
  const allEmbeddings = [];

  for (let i = 0; i < batches.length; i++) {
    const result = batchResults.get(i);
    if (!result || !result.embeddings) continue;

    for (let j = 0; j < result.words.length; j++) {
      const word = result.words[j];
      const embedding = result.embeddings[j];

      if (!embedding || embedding.length !== EMBEDDING_DIM) {
        console.warn(`\nWarning: ${word} has unexpected embedding dimension`);
        continue;
      }

      wordIndex[word] = allEmbeddings.length / EMBEDDING_DIM;
      allEmbeddings.push(...embedding);
    }
  }

  console.log('\n');

  // Create output directory
  const outputDir = path.join(__dirname, '../data');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Save binary embeddings
  const float32Array = new Float32Array(allEmbeddings);
  const buffer = Buffer.from(float32Array.buffer);
  const binPath = path.join(outputDir, 'embeddings.bin');
  fs.writeFileSync(binPath, buffer);
  console.log(`Saved binary embeddings to: ${binPath}`);
  console.log(`Binary size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

  // Save word index
  const indexPath = path.join(outputDir, 'embeddings-index.json');
  fs.writeFileSync(indexPath, JSON.stringify(wordIndex));
  console.log(`Saved word index to: ${indexPath}`);
  console.log(`Total words indexed: ${Object.keys(wordIndex).length}`);

  // Summary
  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  console.log(`\n--- Complete ---`);
  console.log(`Total time: ${totalTime} minutes`);
  console.log(`Words processed: ${Object.keys(wordIndex).length}/${WORD_LIST.length}`);
}

main().catch(console.error);
