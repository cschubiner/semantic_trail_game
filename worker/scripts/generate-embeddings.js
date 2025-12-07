#!/usr/bin/env node
/**
 * Embedding Generator Script (Node.js)
 *
 * Generates GTE-base embeddings for all words in the word list.
 * Run with: OPENROUTER_API_KEY=your-key node scripts/generate-embeddings.js
 *
 * Output: data/embeddings.bin + data/embeddings-index.json
 */

const fs = require('fs');
const path = require('path');

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
const BATCH_SIZE = 50;
const DELAY_BETWEEN_BATCHES_MS = 1000;

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
  console.log(`Batch size: ${BATCH_SIZE}`);

  const wordIndex = {};
  const allEmbeddings = [];

  // Process in batches
  const batches = [];
  for (let i = 0; i < WORD_LIST.length; i += BATCH_SIZE) {
    batches.push(WORD_LIST.slice(i, i + BATCH_SIZE));
  }

  console.log(`\nProcessing ${batches.length} batches...`);
  const startTime = Date.now();

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i];
    const progress = ((i + 1) / batches.length * 100).toFixed(1);
    const elapsed = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    const eta = i > 0 ? ((Date.now() - startTime) / (i + 1) * (batches.length - i - 1) / 1000 / 60).toFixed(1) : '?';

    process.stdout.write(`\rBatch ${i + 1}/${batches.length} (${progress}%) - Elapsed: ${elapsed}m, ETA: ${eta}m    `);

    try {
      const embeddings = await getEmbeddings(batch, apiKey);

      for (let j = 0; j < batch.length; j++) {
        const word = batch[j];
        const embedding = embeddings[j];

        if (!embedding || embedding.length !== EMBEDDING_DIM) {
          console.warn(`\nWarning: ${word} has unexpected embedding dimension`);
          continue;
        }

        wordIndex[word] = allEmbeddings.length / EMBEDDING_DIM;
        allEmbeddings.push(...embedding);
      }
    } catch (error) {
      console.error(`\nError processing batch ${i + 1}:`, error.message);
      // Wait longer on error (rate limiting)
      await sleep(5000);
    }

    // Rate limiting
    if (i < batches.length - 1) {
      await sleep(DELAY_BETWEEN_BATCHES_MS);
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
