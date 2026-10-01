import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { webcrypto } from 'node:crypto';
import ts from 'typescript';

// Exercise the actual Worker handler without live credentials or API charges.
const buildDir = await mkdtemp(join(tmpdir(), 'semantic-trail-test-'));
for (const name of ['index', 'wordlist']) {
  const source = await readFile(new URL(`../src/${name}.ts`, import.meta.url), 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2021, module: ts.ModuleKind.ES2022 },
  });
  await writeFile(join(buildDir, `${name}.mjs`), outputText.replace("from './wordlist'", "from './wordlist.mjs'"));
}
const { default: worker } = await import(pathToFileURL(join(buildDir, 'index.mjs')));
globalThis.crypto ??= webcrypto;
after(() => rm(buildDir, { recursive: true, force: true }));

const env = {
  SECRET_SALT: 'test-salt',
  ALLOWED_ORIGINS: 'https://cschubiner.github.io',
  OPENROUTER_API_KEY: 'test-only',
  EMBED_CACHE: { get: async () => null, put: async () => {} },
};
const request = () => new Request('https://example.test/score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', Origin: 'https://cschubiner.github.io' },
  body: JSON.stringify({ guess: 'bread' }),
});

for (const [status, expected] of [[401, 'credentials need updating'], [403, 'credentials need updating'], [402, 'needs more credits'], [429, 'temporarily unavailable'], [500, 'temporarily unavailable']]) {
  test(`provider HTTP ${status} returns an actionable 503 without raw provider details`, async (t) => {
    t.mock.method(globalThis, 'fetch', async () => new Response('private provider diagnostic', { status }));
    t.mock.method(console, 'error', () => {});
    const response = await worker.fetch(request(), env);
    assert.equal(response.status, 503);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), 'https://cschubiner.github.io');
    const body = await response.json();
    assert.ok(body.error.includes(expected));
    assert.ok(!body.error.includes('private provider diagnostic'));
  });
}

test('successful embeddings still return a real similarity score and populate the cache', async (t) => {
  const cached = [];
  t.mock.method(globalThis, 'fetch', async () => Response.json({ data: [{ embedding: [1, 0, 0] }] }));
  const response = await worker.fetch(request(), {
    ...env,
    EMBED_CACHE: { get: async () => null, put: async (key) => cached.push(key) },
  });
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.guess, 'bread');
  assert.equal(body.similarity, 1);
  assert.equal(body.isCorrect, false);
  assert.equal(cached.length, 2);
});
