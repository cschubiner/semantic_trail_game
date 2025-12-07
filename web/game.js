/**
 * Semantic Trail Web Game - Frontend JavaScript
 *
 * Handles:
 * - Submitting guesses to the Worker backend
 * - Updating the guess table UI
 * - Web Speech API microphone input
 */

// Configuration
// Cloudflare Worker backend URL
const API_BASE = 'https://semantic-trail-backend.cschubiner.workers.dev';
const API_URL = API_BASE + '/score';

// For demo/testing without backend, set this to true
const DEMO_MODE = false;

// Game state
let guesses = []; // Array of { word, similarity, score, bucket, isCorrect, llmRank?, llmScore? }
let recentAttempts = []; // Track last 10 attempts (including duplicates) for display
let gameWon = false;
let guessQueue = []; // Queue of words currently being processed
let currentGameIndex = 0; // 0 = word of the day, 1+ = random games

// LocalStorage key (includes date so it resets daily)
const STORAGE_KEY = 'semanticTrail_' + new Date().toISOString().split('T')[0];

// LLM Re-ranking state
let rerankInterval = null;
let lastGuessTime = null;
const RERANK_INTERVAL_MS = 15000; // 15 seconds - check frequently for new words
const INACTIVITY_TIMEOUT_MS = 60000; // Stop re-ranking after 60s of no guesses
let previousTop20Hash = ''; // Track changes to top 20
let rerankGroupId = 0; // Unique ID for each batch sent to LLM

// Timer state
let timerStarted = false;
let timerStartTime = null;
let timerPenaltyMs = 0; // Penalty time from hints (in ms)
let timerInterval = null;
let finalTime = null;

// Score tiers (time in seconds -> tier name)
const SCORE_TIERS = [
  { maxSeconds: 180, name: 'Genius', emoji: '🧠', color: '#ff44ff' },      // 3 min
  { maxSeconds: 240, name: 'Brilliant', emoji: '💎', color: '#44ffff' },   // 4 min
  { maxSeconds: 300, name: 'Sharp', emoji: '🎯', color: '#44ff44' },       // 5 min
  { maxSeconds: 480, name: 'Clever', emoji: '🦊', color: '#ffcc00' },      // 8 min
  { maxSeconds: 600, name: 'Good', emoji: '👍', color: '#ff8844' },        // 10 min
  { maxSeconds: 900, name: 'Solid', emoji: '🪨', color: '#ff6666' },       // 15 min
  { maxSeconds: Infinity, name: 'Persistent', emoji: '🐢', color: '#888888' }, // 20+ min
];

// DOM elements
const guessInput = document.getElementById('guess-input');
const guessBtn = document.getElementById('guess-btn');
const micBtn = document.getElementById('mic-btn');
const micStatus = document.getElementById('mic-status');
const statusEl = document.getElementById('status');
const guessTbody = document.getElementById('guess-tbody');
const noGuessesEl = document.getElementById('no-guesses');
const winBanner = document.getElementById('win-banner');
const winWord = document.getElementById('win-word');
const winGuesses = document.getElementById('win-guesses');
const guessTable = document.getElementById('guess-table');
const hintLetterBtn = document.getElementById('hint-letter-btn');
const hintLengthBtn = document.getElementById('hint-length-btn');
const hintLLMBtn = document.getElementById('hint-llm-btn');
const revealBtn = document.getElementById('reveal-btn');
const hintDisplay = document.getElementById('hint-display');
const recentGuesses = document.getElementById('recent-guesses');
const recentList = document.getElementById('recent-list');
const timerDisplay = document.getElementById('timer-display');
const timerTime = document.getElementById('timer-time');
const timerTier = document.getElementById('timer-tier');
const timerProgress = document.getElementById('timer-progress');
const timerNextTier = document.getElementById('timer-next-tier');
const winTime = document.getElementById('win-time');
const winTierEl = document.getElementById('win-tier');

// ============================================================
// Timer Functions
// ============================================================

/**
 * Get elapsed time in seconds (including penalty)
 */
function getElapsedSeconds() {
  if (!timerStartTime) return 0;
  const now = finalTime || Date.now();
  return Math.floor((now - timerStartTime + timerPenaltyMs) / 1000);
}

/**
 * Get current tier based on elapsed time
 */
function getCurrentTier(seconds) {
  for (const tier of SCORE_TIERS) {
    if (seconds < tier.maxSeconds) {
      return tier;
    }
  }
  return SCORE_TIERS[SCORE_TIERS.length - 1];
}

/**
 * Get next tier (the one you'll drop to if time continues)
 */
function getNextTier(seconds) {
  for (let i = 0; i < SCORE_TIERS.length; i++) {
    if (seconds < SCORE_TIERS[i].maxSeconds) {
      return i < SCORE_TIERS.length - 1 ? SCORE_TIERS[i + 1] : null;
    }
  }
  return null;
}

/**
 * Format seconds as MM:SS
 */
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Start the timer
 */
function startTimer() {
  if (timerStarted) return;
  timerStarted = true;
  timerStartTime = Date.now();
  timerDisplay.classList.remove('hidden');

  timerInterval = setInterval(updateTimerDisplay, 100);
  updateTimerDisplay();
}

/**
 * Stop the timer
 */
function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  finalTime = Date.now();
}

/**
 * Add penalty time (for using hints)
 */
function addTimerPenalty(seconds) {
  timerPenaltyMs += seconds * 1000;
  updateTimerDisplay();
  showStatus(`+${seconds}s penalty added`, 'info');
}

/**
 * Update the timer display UI
 */
function updateTimerDisplay() {
  const elapsed = getElapsedSeconds();
  const currentTier = getCurrentTier(elapsed);
  const nextTier = getNextTier(elapsed);

  // Update time display
  timerTime.textContent = formatTime(elapsed);

  // Update current tier
  timerTier.textContent = `${currentTier.emoji} ${currentTier.name}`;
  timerTier.style.color = currentTier.color;

  // Update progress bar and next tier info
  if (nextTier && !gameWon) {
    const tierStart = SCORE_TIERS[SCORE_TIERS.indexOf(currentTier) - 1]?.maxSeconds || 0;
    const tierEnd = currentTier.maxSeconds;
    const progress = ((elapsed - tierStart) / (tierEnd - tierStart)) * 100;

    timerProgress.style.width = `${Math.min(progress, 100)}%`;
    timerProgress.style.backgroundColor = currentTier.color;

    const secondsLeft = currentTier.maxSeconds - elapsed;
    timerNextTier.textContent = `${formatTime(secondsLeft)} until ${nextTier.name}`;
    timerNextTier.classList.remove('hidden');
  } else {
    timerProgress.style.width = '100%';
    timerProgress.style.backgroundColor = currentTier.color;
    timerNextTier.classList.add('hidden');
  }
}

/**
 * Reset the timer
 */
function resetTimer() {
  timerStarted = false;
  timerStartTime = null;
  timerPenaltyMs = 0;
  finalTime = null;
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerDisplay.classList.add('hidden');
}

// ============================================================
// LocalStorage Persistence
// ============================================================

/**
 * Save game state to localStorage
 */
function saveGameState() {
  const state = {
    guesses,
    recentAttempts,
    gameWon,
    timerStarted,
    timerStartTime,
    timerPenaltyMs,
    finalTime,
    rerankGroupId,
    lastGuessTime,
    currentGameIndex,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.error('Failed to save game state:', e);
  }
}

/**
 * Load game state from localStorage
 */
function loadGameState() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return false;

    const state = JSON.parse(saved);
    guesses = state.guesses || [];
    recentAttempts = state.recentAttempts || [];
    gameWon = state.gameWon || false;
    timerStarted = state.timerStarted || false;
    timerStartTime = state.timerStartTime;
    timerPenaltyMs = state.timerPenaltyMs || 0;
    finalTime = state.finalTime;
    rerankGroupId = state.rerankGroupId || 0;
    lastGuessTime = state.lastGuessTime;
    currentGameIndex = state.currentGameIndex || 0;

    return true;
  } catch (e) {
    console.error('Failed to load game state:', e);
    return false;
  }
}

/**
 * Clear old localStorage keys from previous days
 */
function clearOldStorage() {
  const todayKey = STORAGE_KEY;
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('semanticTrail_') && key !== todayKey) {
      localStorage.removeItem(key);
    }
  }
}

// ============================================================
// LLM Re-ranking: Smart Anchor + Blended Score (Hybrid)
// ============================================================

const LLM_WEIGHT = 0.7; // 70% LLM, 30% Embeddings - safety net against hallucinations

/**
 * Start the re-ranking interval
 */
function startRerankInterval() {
  if (rerankInterval || DEMO_MODE) return;
  console.log('Starting Smart Anchor Re-ranker');
  rerankInterval = setInterval(checkAndRerank, RERANK_INTERVAL_MS);
}

/**
 * Stop the re-ranking interval
 */
function stopRerankInterval() {
  if (rerankInterval) {
    clearInterval(rerankInterval);
    rerankInterval = null;
  }
}

/**
 * Reset re-ranking state - restore to original embedding scores
 */
function resetRerankState() {
  stopRerankInterval();
  lastGuessTime = null;
  rerankGroupId = 0;
  previousTop20Hash = '';
  // Reset state to original embeddings
  for (const g of guesses) {
    g.score = g.embeddingScore || g.score;
    g.llmScore = null;
    g.llmRank = undefined;
    g.rerankGroupId = 0;
    g.bucket = getBucket(g.score);
  }
}

/**
 * Main Loop: Check if we need to re-rank
 */
async function checkAndRerank() {
  // Need enough words to make sorting meaningful
  if (gameWon || guesses.length < 3) return;

  // Inactivity check
  if (lastGuessTime && Date.now() - lastGuessTime > INACTIVITY_TIMEOUT_MS) {
    stopRerankInterval();
    return;
  }

  // Build the "Swiss Cheese" Batch - skips stable middles, prioritizes new words
  const batch = buildSmartBatch(20);

  // If we only have anchors (size < 3) and no new words, skip
  if (batch.length < 3) return;

  // Hash check to prevent spamming the exact same request
  const batchHash = batch.map(g => g.word).join(',');
  if (batchHash === previousTop20Hash) return;

  console.log('Smart Rerank batch:', batch.map(b => `${b.word}(${b.rerankGroupId || 'new'})`).join(', '));
  await performSmartRerank(batch, batchHash);
}

/**
 * SELECTOR: "Swiss Cheese" Logic
 * Iterates down the list. If it sees a block of words sharing the same
 * rerankGroupId, it grabs the Head & Tail (Anchors) and skips the Body.
 * This naturally self-heals when new words break up stable blocks.
 */
function buildSmartBatch(maxSize) {
  // Sort by current effective score
  const sorted = [...guesses]
    .filter(g => !g.isCorrect)
    .sort((a, b) => b.score - a.score);

  const batch = [];
  let i = 0;

  while (batch.length < maxSize && i < sorted.length) {
    const current = sorted[i];

    // Ensure state validity (for existing words loaded from storage)
    current.embeddingScore = current.embeddingScore || current.score;

    // Case A: Word is new or part of a mixed/broken group. Add it.
    if (!current.rerankGroupId) {
      batch.push(current);
      i++;
      continue;
    }

    // Case B: Word is part of a stable group. Check block size.
    let blockEnd = i;
    while (
      blockEnd + 1 < sorted.length &&
      sorted[blockEnd + 1].rerankGroupId === current.rerankGroupId
    ) {
      blockEnd++;
    }

    const blockSize = blockEnd - i + 1;

    // "The 3-8 Logic": If block is big (>=3), skip the middle
    if (blockSize >= 3) {
      // Add Head (Top Anchor)
      batch.push(sorted[i]);

      // Add Tail (Bottom Anchor) if space
      if (batch.length < maxSize) {
        batch.push(sorted[blockEnd]);
      }

      // Skip the body (words i+1 to blockEnd-1)
      i = blockEnd + 1;
    } else {
      // Block too small or fragmented? Just add the word to re-verify it.
      batch.push(current);
      i++;
    }
  }

  return batch;
}

/**
 * EXECUTION: Rerank -> MinMax -> Blend
 * The blend is the safety net - embeddings always have 30% influence
 */
async function performSmartRerank(batch, batchHash) {
  // Determine Range (Ceiling & Floor) from the selected batch's current scores
  const scores = batch.map(g => g.score);
  const ceiling = Math.max(...scores);
  const floor = Math.min(...scores);

  const requestBody = {
    guesses: batch.map(g => ({ word: g.word, score: g.score })),
    game: currentGameIndex
  };

  try {
    const response = await fetch(`${API_BASE}/rerank`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      console.error('Rerank request failed:', response.status);
      return;
    }

    const data = await response.json();

    if (data.rateLimited) {
      console.log('LLM rate limited:', data.message);
      return;
    }

    // Increment Group ID: These words are now a new "stable" block
    rerankGroupId++;

    const resultCount = data.rankings.length;

    // Apply Min-Max Normalization + Blending
    for (let index = 0; index < data.rankings.length; index++) {
      const r = data.rankings[index];
      const g = guesses.find(x => x.word.toLowerCase() === r.word.toLowerCase());
      if (!g) continue;

      // Ensure embeddingScore exists
      g.embeddingScore = g.embeddingScore || g.score;

      // 1. Min-Max Normalization (Rank -> Target Score)
      let target;
      if (resultCount <= 1) {
        target = ceiling;
      } else {
        // Rank 1 gets Ceiling, Rank N gets Floor
        const ratio = (resultCount - 1 - index) / (resultCount - 1);
        target = floor + (ratio * (ceiling - floor));
      }

      // 2. Smooth the LLM opinion (50/50 blend of old vs new)
      if (g.llmScore === null) {
        g.llmScore = target;
      } else {
        g.llmScore = (0.5 * g.llmScore) + (0.5 * target);
      }

      // 3. THE BLEND: Combine LLM + Embedding (safety net)
      g.score = Number((
        (LLM_WEIGHT * g.llmScore) +
        ((1 - LLM_WEIGHT) * g.embeddingScore)
      ).toFixed(2));

      // 4. Update State
      g.rerankGroupId = rerankGroupId;
      g.llmRank = index + 1;
      g.bucket = getBucket(g.score);
    }

    previousTop20Hash = batchHash;
    renderGuesses();
    saveGameState();
    console.log(`Smart rerank complete: ${resultCount} words, ceiling=${ceiling.toFixed(1)}, floor=${floor.toFixed(1)}`);

  } catch (error) {
    console.error('Error in performSmartRerank:', error);
  }
}

// ============================================================
// Core Game Logic
// ============================================================

/**
 * Add an attempt to the recent attempts list (keeps last 10)
 */
function addRecentAttempt(word, score, isCorrect = false) {
  recentAttempts.push({ word, score, isCorrect, timestamp: Date.now() });
  // Keep only last 10
  if (recentAttempts.length > 10) {
    recentAttempts = recentAttempts.slice(-10);
  }
}

/**
 * Submit a guess to the backend (processes in parallel)
 */
async function submitGuess(word = null) {
  const guess = (word || guessInput.value).trim().toLowerCase();

  // Clear input immediately so user can keep typing
  if (!word) {
    guessInput.value = '';
  }

  if (!guess) {
    return;
  }

  if (!/^[a-z]+$/.test(guess)) {
    showStatus('Please enter letters only', 'error');
    return;
  }

  // Check if already guessed or already processing
  const existingGuess = guesses.find(g => g.word === guess);
  if (existingGuess || guessQueue.includes(guess)) {
    // Still add to recent attempts so user sees what they typed
    if (existingGuess) {
      addRecentAttempt(existingGuess.word, existingGuess.score, existingGuess.isCorrect);
      renderRecentGuesses();
    }
    showStatus(`Already guessed "${guess}"`, 'info');
    return;
  }

  // Add to queue and process (fire and forget - parallel processing)
  guessQueue.push(guess);
  processGuess(guess); // Don't await - let it run in parallel
}

/**
 * Process a single guess (runs in parallel with other guesses)
 */
async function processGuess(guess) {
  // Start timer on first guess
  if (!timerStarted) {
    startTimer();
  }

  try {
    let result;

    if (DEMO_MODE) {
      // Demo mode: generate fake scores for testing without backend
      result = generateDemoScore(guess);
    } else {
      // Real mode: call the Worker backend
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ guess, game: currentGameIndex }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Request failed');
      }

      result = await response.json();
    }

    // Remove from queue
    const queueIndex = guessQueue.indexOf(guess);
    if (queueIndex > -1) {
      guessQueue.splice(queueIndex, 1);
    }

    // Add to guesses (compute bucket on frontend)
    // HYBRID STATE: track embedding score separately as permanent anchor
    const newGuess = {
      word: result.guess,
      similarity: result.similarity,
      score: result.score,              // Effective score (starts as embedding, gets blended)
      embeddingScore: result.score,     // Permanent anchor - never changes
      llmScore: null,                   // LLM's opinion (null = not yet ranked)
      rerankGroupId: 0,                 // 0 = never sorted by LLM
      bucket: getBucket(result.score),
      isCorrect: result.isCorrect || false,
    };
    guesses.push(newGuess);

    // Also add to recent attempts for display
    addRecentAttempt(newGuess.word, newGuess.score, newGuess.isCorrect);

    // Track last guess time and start re-rank interval
    lastGuessTime = Date.now();
    if (!rerankInterval && !DEMO_MODE) {
      startRerankInterval();
    }

    // Check for win
    if (result.isCorrect) {
      handleWin(result.guess);
      guessQueue = []; // Clear queue on win
    } else {
      // Show word, score, bucket, and rank in status
      const rank = getRank(result.score);
      const bucketLabel = getBucket(result.score);
      const pendingInfo = guessQueue.length > 0 ? ` (${guessQueue.length} pending)` : '';
      showStatus(`"${result.guess}" → ${result.score}/100 (${bucketLabel}) — Rank #${rank}/${guesses.length}${pendingInfo}`, 'info');
    }

    // Update UI
    renderGuesses();
    saveGameState();

  } catch (error) {
    console.error('Error submitting guess:', error);
    // Remove from queue on error too
    const queueIndex = guessQueue.indexOf(guess);
    if (queueIndex > -1) {
      guessQueue.splice(queueIndex, 1);
    }
    showStatus(`Error for "${guess}": ${error.message}`, 'error');
  }
}

/**
 * Generate a demo score for testing without backend
 */
function generateDemoScore(guess) {
  // Simulate some semantic similarity based on word characteristics
  // This is just for demo purposes - real scores come from embeddings

  // Secret word for demo mode
  const secretWord = 'ocean';

  if (guess === secretWord) {
    return {
      guess,
      similarity: 1.0,
      score: 100,
      bucket: 'CORRECT!',
      isCorrect: true,
    };
  }

  // Related words get higher scores
  const relatedWords = {
    'sea': 92,
    'water': 85,
    'wave': 82,
    'beach': 78,
    'marine': 75,
    'fish': 70,
    'ship': 65,
    'blue': 55,
    'salt': 52,
    'deep': 48,
    'swim': 45,
    'lake': 60,
    'river': 55,
    'boat': 58,
    'whale': 72,
    'dolphin': 70,
    'coral': 68,
    'island': 62,
    'coast': 65,
    'tide': 75,
  };

  let score;
  if (relatedWords[guess]) {
    score = relatedWords[guess];
  } else {
    // Random score between 5-40 for unrelated words
    score = Math.floor(Math.random() * 35) + 5;
  }

  const similarity = 0.20 + (score / 100) * 0.65;

  return {
    guess,
    similarity: Math.round(similarity * 1000) / 1000,
    score,
    bucket: getBucket(score),
    isCorrect: false,
  };
}

/**
 * Get bucket label from score
 */
function getBucket(score) {
  if (score >= 95) return 'BURNING';
  if (score >= 80) return 'Hot';
  if (score >= 60) return 'Warm';
  if (score >= 40) return 'Tepid';
  if (score >= 20) return 'Cold';
  return 'Freezing';
}

/**
 * Get rank of a score among all guesses
 */
function getRank(score) {
  const sortedScores = guesses.map(g => g.score).sort((a, b) => b - a);
  return sortedScores.indexOf(score) + 1;
}

/**
 * Handle winning the game
 */
function handleWin(word) {
  gameWon = true;

  // Stop the timer and re-ranking
  stopTimer();
  stopRerankInterval();
  const elapsed = getElapsedSeconds();
  const finalTier = getCurrentTier(elapsed);

  winWord.textContent = word.toUpperCase();
  winGuesses.textContent = guesses.length;
  winTime.textContent = formatTime(elapsed);
  winTierEl.textContent = `${finalTier.emoji} ${finalTier.name}`;
  winTierEl.style.color = finalTier.color;

  winBanner.classList.remove('hidden');
  setInputDisabled(true);
  showStatus('', '');
  saveGameState();
}

/**
 * Reset the game (same word)
 */
function resetGame() {
  guesses = [];
  recentAttempts = [];
  gameWon = false;
  guessQueue = [];
  winBanner.classList.add('hidden');
  hintDisplay.classList.add('hidden');
  hintDisplay.textContent = '';
  resetTimer();
  resetRerankState();
  setInputDisabled(false);
  showStatus('Game reset!', 'success');
  renderGuesses();
  saveGameState();
  guessInput.focus();

  // Clear status after a moment
  setTimeout(() => {
    if (!gameWon) showStatus('', '');
  }, 2000);
}

/**
 * Start a new game with a different random word
 */
function newRandomGame() {
  currentGameIndex++;
  guesses = [];
  recentAttempts = [];
  gameWon = false;
  guessQueue = [];
  winBanner.classList.add('hidden');
  hintDisplay.classList.add('hidden');
  hintDisplay.textContent = '';
  resetTimer();
  resetRerankState();
  setInputDisabled(false);
  showStatus(`New word #${currentGameIndex + 1}!`, 'success');
  renderGuesses();
  saveGameState();
  guessInput.focus();

  // Clear status after a moment
  setTimeout(() => {
    if (!gameWon) showStatus('', '');
  }, 2000);
}

/**
 * Request a hint from the backend
 */
async function getHint(type) {
  if (gameWon) {
    showStatus('Game already won!', 'info');
    return;
  }

  // Start timer if not started (penalty still applies)
  if (!timerStarted) {
    startTimer();
  }

  // Add 60 second penalty for using hints
  addTimerPenalty(60);

  if (DEMO_MODE) {
    // Demo mode hints for "ocean"
    if (type === 'letter') {
      showHint('The word starts with "O"');
    } else {
      showHint('The word has 5 letters');
    }
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/hint?type=${type}&game=${currentGameIndex}`);
    if (!response.ok) {
      throw new Error('Failed to get hint');
    }
    const data = await response.json();
    showHint(data.hint);
  } catch (error) {
    console.error('Error getting hint:', error);
    showStatus('Failed to get hint', 'error');
  }
}

/**
 * Reveal the secret word (give up)
 */
async function revealWord() {
  if (gameWon) {
    showStatus('Game already won!', 'info');
    return;
  }

  if (DEMO_MODE) {
    showHint('The secret word was: OCEAN');
    gameWon = true;
    setInputDisabled(true);
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/reveal?game=${currentGameIndex}`);
    if (!response.ok) {
      throw new Error('Failed to reveal word');
    }
    const data = await response.json();
    showHint(data.message);
    gameWon = true;
    setInputDisabled(true);
    saveGameState();
  } catch (error) {
    console.error('Error revealing word:', error);
    showStatus('Failed to reveal word', 'error');
  }
}

/**
 * Request an LLM-generated hint based on top guesses (2 minute penalty)
 */
async function getLLMHint() {
  if (gameWon) {
    showStatus('Game already won!', 'info');
    return;
  }

  if (guesses.length < 1) {
    showStatus('Make at least one guess first!', 'info');
    return;
  }

  // Start timer if not started (penalty still applies)
  if (!timerStarted) {
    startTimer();
  }

  // Add 2 minute (120 second) penalty for LLM hint
  addTimerPenalty(120);

  if (DEMO_MODE) {
    showHint('Demo hint: Think about vast, blue depths...');
    return;
  }

  // Get top 5 guesses sorted by score (g.score is now the blended score)
  const topGuesses = [...guesses]
    .filter(g => !g.isCorrect)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map(g => ({ word: g.word, score: g.score }));

  try {
    hintLLMBtn.disabled = true;
    showStatus('Getting hint from AI...', 'info');

    const response = await fetch(`${API_BASE}/llm-hint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topGuesses, game: currentGameIndex }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get LLM hint');
    }

    const data = await response.json();

    if (data.rateLimited) {
      showStatus('LLM hint rate limited, try again later', 'info');
    } else {
      showHint(data.hint);
      showStatus('', '');
    }
  } catch (error) {
    console.error('Error getting LLM hint:', error);
    showStatus(`Failed to get LLM hint: ${error.message}`, 'error');
  } finally {
    hintLLMBtn.disabled = false;
  }
}

/**
 * Display a hint message
 */
function showHint(message) {
  hintDisplay.textContent = message;
  hintDisplay.classList.remove('hidden');
}

// ============================================================
// UI Rendering
// ============================================================

/**
 * Render the 10 most recent attempts (includes duplicates to show what was tried)
 */
function renderRecentGuesses() {
  if (recentAttempts.length === 0) {
    recentGuesses.classList.add('hidden');
    return;
  }

  recentGuesses.classList.remove('hidden');

  // Get recent attempts (most recent first)
  const recent = [...recentAttempts].reverse();

  recentList.innerHTML = recent.map(g => {
    const score = Math.round(g.score);
    const bucketClass = getBucketClass(score, g.isCorrect);
    const isCorrect = g.isCorrect ? ' correct' : '';

    return `
      <div class="recent-item${isCorrect}">
        <span class="recent-word">${escapeHtml(g.word)}</span>
        <span class="recent-score ${bucketClass}">${score}</span>
      </div>
    `;
  }).join('');
}

/**
 * Render the guesses table
 */
function renderGuesses() {
  // Render recent guesses (last 5)
  renderRecentGuesses();

  if (guesses.length === 0) {
    noGuessesEl.classList.remove('hidden');
    guessTable.classList.add('hidden');
    return;
  }

  noGuessesEl.classList.add('hidden');
  guessTable.classList.remove('hidden');

  // Sort by effective score (g.score is now the blended score)
  const sorted = [...guesses].sort((a, b) => b.score - a.score);

  // Build table rows
  guessTbody.innerHTML = sorted.map((g, index) => {
    const isLatest = g.word === guesses[guesses.length - 1].word;
    // g.score is now the blended effective score
    const displayScore = g.score;
    const bucketLabel = g.isCorrect ? 'CORRECT!' : (g.bucket || getBucket(displayScore));
    const bucketClass = getBucketClass(displayScore, g.isCorrect);
    const barClass = getBarClass(displayScore, g.isCorrect);

    // Show LLM score if available (yellow, rounded) - this is the raw LLM opinion
    const llmDisplay = g.llmScore !== null && g.llmScore !== undefined
      ? `<span class="llm-score">${Math.round(g.llmScore)}</span>`
      : '<span class="llm-pending">—</span>';

    return `
      <tr class="${isLatest ? 'latest' : ''}">
        <td>${index + 1}</td>
        <td class="word-cell">${escapeHtml(g.word)}</td>
        <td class="score-cell">${Math.round(displayScore)}</td>
        <td class="llm-cell">${llmDisplay}</td>
        <td>
          <div class="score-bar">
            <div class="score-bar-fill ${barClass}" style="width: ${displayScore}%"></div>
          </div>
          <span class="bucket-cell ${bucketClass}">${bucketLabel}</span>
        </td>
      </tr>
    `;
  }).join('');
}

/**
 * Get CSS class for bucket text
 */
function getBucketClass(score, isCorrect) {
  if (isCorrect) return 'bucket-correct';
  if (score >= 95) return 'bucket-burning';
  if (score >= 80) return 'bucket-hot';
  if (score >= 60) return 'bucket-warm';
  if (score >= 40) return 'bucket-tepid';
  if (score >= 20) return 'bucket-cold';
  return 'bucket-freezing';
}

/**
 * Get CSS class for score bar
 */
function getBarClass(score, isCorrect) {
  if (isCorrect) return 'bar-correct';
  if (score >= 95) return 'bar-burning';
  if (score >= 80) return 'bar-hot';
  if (score >= 60) return 'bar-warm';
  if (score >= 40) return 'bar-tepid';
  if (score >= 20) return 'bar-cold';
  return 'bar-freezing';
}

/**
 * Show a status message
 */
function showStatus(message, type) {
  statusEl.textContent = message;
  statusEl.className = 'status';
  if (type) {
    statusEl.classList.add(type);
  }
}

/**
 * Enable/disable input elements
 */
function setInputDisabled(disabled) {
  guessInput.disabled = disabled;
  guessBtn.disabled = disabled;
  micBtn.disabled = disabled;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ============================================================
// Web Speech API - Microphone Input (Continuous Mode)
// ============================================================

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let micShouldBeActive = false;  // User's intent
let micIsRunning = false;       // Actual recognition state
let restartTimeout = null;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.continuous = true;       // Keep listening
  recognition.interimResults = false;  // Only final results
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    console.log('Recognition started');
    micIsRunning = true;
    micBtn.classList.add('listening');
    micStatus.classList.remove('hidden');
    micStatus.textContent = 'Listening... (click mic to stop)';
  };

  recognition.onend = () => {
    console.log('Recognition ended, micShouldBeActive:', micShouldBeActive);
    micIsRunning = false;

    // Clear any pending restart
    if (restartTimeout) {
      clearTimeout(restartTimeout);
      restartTimeout = null;
    }

    // If user still wants mic active, restart after a delay
    if (micShouldBeActive && !gameWon) {
      micStatus.textContent = 'Restarting...';
      restartTimeout = setTimeout(() => {
        if (micShouldBeActive && !gameWon && !micIsRunning) {
          console.log('Attempting to restart recognition...');
          try {
            recognition.start();
          } catch (e) {
            console.error('Restart failed:', e);
            // If it fails because already started, that's actually fine
            if (e.name !== 'InvalidStateError') {
              micShouldBeActive = false;
              micBtn.classList.remove('listening');
              micStatus.classList.add('hidden');
              showStatus('Mic restart failed - click mic to try again', 'error');
            }
          }
        }
      }, 250);
    } else {
      micBtn.classList.remove('listening');
      micStatus.classList.add('hidden');
    }
  };

  recognition.onresult = (event) => {
    // Process all results from this event
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        const transcript = event.results[i][0].transcript;
        const words = transcript.trim().toLowerCase().split(/\s+/);
        console.log('Speech recognized:', transcript, '-> words:', words);

        for (const word of words) {
          if (word && /^[a-z]+$/.test(word)) {
            if (!guesses.some(g => g.word === word)) {
              submitGuess(word);
            } else {
              console.log(`Skipping "${word}" - already guessed`);
            }
          }
        }
      }
    }

    // Confirm still listening
    if (micShouldBeActive) {
      micStatus.textContent = 'Listening... (click mic to stop)';
    }
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);

    if (event.error === 'not-allowed') {
      micShouldBeActive = false;
      micIsRunning = false;
      micBtn.classList.remove('listening');
      micStatus.classList.add('hidden');
      showStatus('Microphone access denied', 'error');
    } else if (event.error === 'no-speech') {
      // This is fine, keep listening
      micStatus.textContent = 'No speech detected... keep talking!';
    } else if (event.error === 'aborted') {
      // User or system stopped, handled in onend
    } else if (event.error === 'network') {
      micStatus.textContent = 'Network error... retrying';
    } else {
      console.log('Unhandled speech error:', event.error);
    }
  };
} else {
  micBtn.style.display = 'none';
  console.log('Web Speech API not supported in this browser');
}

/**
 * Toggle microphone recognition on/off
 */
function startMicGuess() {
  if (!recognition) {
    showStatus('Speech recognition not supported', 'error');
    return;
  }

  if (gameWon) {
    showStatus('Game over! Click "Play Again" to continue', 'info');
    return;
  }

  console.log('Mic button clicked. shouldBeActive:', micShouldBeActive, 'isRunning:', micIsRunning);

  if (micShouldBeActive) {
    // User wants to stop
    micShouldBeActive = false;
    if (restartTimeout) {
      clearTimeout(restartTimeout);
      restartTimeout = null;
    }
    try {
      recognition.abort();  // Use abort() for immediate stop
    } catch (e) {
      console.log('Abort error (ok):', e);
    }
    micIsRunning = false;
    micBtn.classList.remove('listening');
    micStatus.classList.add('hidden');
    showStatus('Microphone stopped', 'info');
  } else {
    // User wants to start
    micShouldBeActive = true;

    // Force stop first to ensure clean state
    try {
      recognition.abort();
    } catch (e) {
      // Ignore - might not be running
    }

    // Small delay then start fresh
    setTimeout(() => {
      if (micShouldBeActive) {
        try {
          recognition.start();
        } catch (e) {
          console.error('Start error:', e);
          micShouldBeActive = false;
          showStatus('Failed to start microphone', 'error');
        }
      }
    }, 100);
  }
}

// ============================================================
// Event Listeners
// ============================================================

// Submit on Enter key
guessInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !guessInput.disabled) {
    submitGuess();
  }
});

// Submit button click
guessBtn.addEventListener('click', () => submitGuess());

// Mic button click
micBtn.addEventListener('click', startMicGuess);

// Hint and reveal button clicks
hintLetterBtn.addEventListener('click', () => getHint('letter'));
hintLengthBtn.addEventListener('click', () => getHint('length'));
hintLLMBtn.addEventListener('click', getLLMHint);
revealBtn.addEventListener('click', revealWord);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  // Clear old day's storage
  clearOldStorage();

  // Try to restore saved state
  if (loadGameState()) {
    console.log('Restored game state:', guesses.length, 'guesses');
    renderGuesses();

    // Restore timer display if game was in progress
    if (timerStarted && !gameWon) {
      timerDisplay.classList.remove('hidden');
      timerInterval = setInterval(updateTimerDisplay, 100);
      updateTimerDisplay();
      // Restart rerank interval if there were guesses
      if (guesses.length >= 3) {
        startRerankInterval();
      }
    }

    // Restore win banner if game was won
    if (gameWon) {
      const winningGuess = guesses.find(g => g.isCorrect);
      if (winningGuess) {
        winWord.textContent = winningGuess.word.toUpperCase();
        winGuesses.textContent = guesses.length;
        const elapsed = getElapsedSeconds();
        const finalTier = getCurrentTier(elapsed);
        winTime.textContent = formatTime(elapsed);
        winTierEl.textContent = `${finalTier.emoji} ${finalTier.name}`;
        winTierEl.style.color = finalTier.color;
        winBanner.classList.remove('hidden');
        timerDisplay.classList.remove('hidden');
        updateTimerDisplay();
        setInputDisabled(true);
      }
    }
  }

  guessInput.focus();

  // Show demo mode notice
  if (DEMO_MODE) {
    showStatus('Demo mode: Try to guess "ocean"! (Backend not connected)', 'info');
  }
});

// Expose functions to window for button onclick
window.resetGame = resetGame;
window.newRandomGame = newRandomGame;
window.submitGuess = submitGuess;
window.startMicGuess = startMicGuess;
