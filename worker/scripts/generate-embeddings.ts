/**
 * Embedding Generator Script
 *
 * Generates GTE-base embeddings for all words in the word list.
 * Run with: npx ts-node scripts/generate-embeddings.ts
 *
 * Requires OPENROUTER_API_KEY environment variable.
 *
 * Output: embeddings.bin (binary Float32Array) + embeddings-index.json (word -> offset mapping)
 */

import * as fs from 'fs';
import * as path from 'path';

// Import word list
import { WORD_LIST } from '../src/wordlist';

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings';
const EMBEDDING_MODEL = 'thenlper/gte-base';
const EMBEDDING_DIM = 768; // GTE-base dimension
const BATCH_SIZE = 50; // Process words in batches
const DELAY_BETWEEN_BATCHES_MS = 1000; // Rate limiting

interface EmbeddingResponse {
  data: Array<{ embedding: number[] }>;
}

async function getEmbeddings(words: string[], apiKey: string): Promise<number[][]> {
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

  const data = await response.json() as EmbeddingResponse;
  return data.data.map(d => d.embedding);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('Error: OPENROUTER_API_KEY environment variable is required');
    process.exit(1);
  }

  console.log(`Generating embeddings for ${WORD_LIST.length} words...`);
  console.log(`Model: ${EMBEDDING_MODEL}`);
  console.log(`Embedding dimension: ${EMBEDDING_DIM}`);

  // Create index mapping: word -> offset in binary file
  const wordIndex: Record<string, number> = {};
  const allEmbeddings: number[] = [];

  // Process in batches
  const batches = [];
  for (let i = 0; i < WORD_LIST.length; i += BATCH_SIZE) {
    batches.push(WORD_LIST.slice(i, i + BATCH_SIZE));
  }

  console.log(`Processing ${batches.length} batches of ${BATCH_SIZE} words each...`);

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i];
    console.log(`Batch ${i + 1}/${batches.length}: ${batch[0]} ... ${batch[batch.length - 1]}`);

    try {
      const embeddings = await getEmbeddings(batch, apiKey);

      for (let j = 0; j < batch.length; j++) {
        const word = batch[j];
        const embedding = embeddings[j];

        if (embedding.length !== EMBEDDING_DIM) {
          console.warn(`Warning: ${word} has unexpected embedding dimension: ${embedding.length}`);
        }

        // Store offset (position in flat array, not byte offset)
        wordIndex[word] = allEmbeddings.length / EMBEDDING_DIM;
        allEmbeddings.push(...embedding);
      }
    } catch (error) {
      console.error(`Error processing batch ${i + 1}:`, error);
      // Continue with next batch
    }

    // Rate limiting
    if (i < batches.length - 1) {
      await sleep(DELAY_BETWEEN_BATCHES_MS);
    }
  }

  // Save binary embeddings
  const outputDir = path.join(__dirname, '../data');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Save as Float32Array binary
  const float32Array = new Float32Array(allEmbeddings);
  const buffer = Buffer.from(float32Array.buffer);
  const binPath = path.join(outputDir, 'embeddings.bin');
  fs.writeFileSync(binPath, buffer);
  console.log(`\nSaved binary embeddings to: ${binPath}`);
  console.log(`Binary size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

  // Save word index
  const indexPath = path.join(outputDir, 'embeddings-index.json');
  fs.writeFileSync(indexPath, JSON.stringify(wordIndex, null, 2));
  console.log(`Saved word index to: ${indexPath}`);
  console.log(`Total words indexed: ${Object.keys(wordIndex).length}`);

  // Verification
  console.log('\n--- Verification ---');
  console.log(`Expected embeddings: ${WORD_LIST.length}`);
  console.log(`Actual embeddings: ${Object.keys(wordIndex).length}`);
  console.log(`Expected binary size: ${(WORD_LIST.length * EMBEDDING_DIM * 4 / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Actual binary size: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);
}

main().catch(console.error);
