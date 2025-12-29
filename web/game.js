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

  // Minimum length check
  if (guess.length < 3) {
    showStatus('Guesses must be at least 3 letters', 'error');
    return;
  }

  // Block stop words
  if (STOP_WORDS.has(guess)) {
    showStatus('That word is too common. Try a more specific word.', 'error');
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

  // Stop Similarity mode transcribe recording if active
  if (typeof stopSimRecording === 'function') {
    stopSimRecording();
  }

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
  // Generate a random game ID (1 to 1 billion) instead of incrementing
  // Game 0 is reserved for the deterministic word of the day
  currentGameIndex = Math.floor(Math.random() * 1_000_000_000) + 1;
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
  showStatus('New random word!', 'success');
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
  // Only show in Similarity mode
  if (recentAttempts.length === 0 || currentMode === 'questions') {
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
// GPT-4o Transcribe Recording (Similarity Mode)
// Uses OpenAI Realtime transcription streaming (single buffer)
// ============================================================

// Shared helper: encode Float32 audio to base64 PCM16
function encodeFloat32ToPcm16Base64(inputData) {
  const pcm16 = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    const s = Math.max(-1, Math.min(1, inputData[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer)));
}

// Transcribe recording state (Similarity mode)
let simIsRecording = false;
let simRealtimeWs = null;       // WebSocket to OpenAI
let simAudioWorklet = null;     // { source, processor }
let simStreamingTranscript = ''; // Running transcript (for debugging)

// DOM elements for transcribe recording
const transcribeRecordBtn = document.getElementById('transcribe-record-btn');
const transcribeStatus = document.getElementById('transcribe-status');
const transcribeIndicator = document.getElementById('transcribe-indicator');
const transcribeStatusText = document.getElementById('transcribe-status-text');

/**
 * Toggle Similarity mode transcribe recording
 */
async function toggleSimRecording() {
  if (simIsRecording) {
    stopSimRecording();
  } else {
    await startSimRecording();
  }
}

/**
 * Start Similarity mode transcribe recording (single streaming buffer)
 */
async function startSimRecording() {
  if (gameWon) {
    showStatus('Game over! Start a new game.', 'info');
    return;
  }

  // Stop Web Speech API mic if active
  if (micShouldBeActive) {
    micShouldBeActive = false;
    try { recognition?.abort(); } catch (e) {}
    micBtn.classList.remove('listening');
    micStatus.classList.add('hidden');
  }

  simIsRecording = true;
  simStreamingTranscript = '';

  // Update UI
  if (transcribeRecordBtn) {
    transcribeRecordBtn.classList.add('recording');
    transcribeRecordBtn.innerHTML = '<span class="record-icon">&#x23F9;</span> Stop Recording';
  }
  if (transcribeIndicator) transcribeIndicator.classList.add('active');
  if (transcribeStatus) transcribeStatus.classList.remove('hidden');
  if (transcribeStatusText) transcribeStatusText.textContent = 'Connecting...';

  // Start timer on first recording
  if (!timerStarted) {
    startTimer();
  }

  // Initialize shared audio system if needed
  if (!audioStream || !audioContext) {
    const success = await initAudioSystem();
    if (!success) {
      simIsRecording = false;
      if (transcribeRecordBtn) {
        transcribeRecordBtn.classList.remove('recording');
        transcribeRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> GPT-4o Transcribe';
      }
      if (transcribeIndicator) transcribeIndicator.classList.remove('active');
      if (transcribeStatus) transcribeStatus.classList.add('hidden');
      return;
    }
  }

  // Resume audio context if suspended
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  // Connect to OpenAI
  const connected = await connectSimRealtimeWebSocket();
  if (!connected) {
    simIsRecording = false;
    if (transcribeStatusText) transcribeStatusText.textContent = 'Connection failed';
    if (transcribeRecordBtn) {
      transcribeRecordBtn.classList.remove('recording');
      transcribeRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> GPT-4o Transcribe';
    }
    if (transcribeIndicator) transcribeIndicator.classList.remove('active');
    if (transcribeStatus) transcribeStatus.classList.add('hidden');
    return;
  }

  // Wait for WebSocket to be ready, then start streaming
  const waitForOpen = setInterval(() => {
    if (simRealtimeWs && simRealtimeWs.readyState === WebSocket.OPEN) {
      clearInterval(waitForOpen);
      startSimAudioStreaming();
      if (transcribeStatusText) {
        transcribeStatusText.textContent = 'Streaming... speak your guesses';
      }
    }
  }, 100);
}

/**
 * Stop Similarity mode transcribe recording
 */
function stopSimRecording() {
  if (!simIsRecording) return;

  simIsRecording = false;
  simStreamingTranscript = '';

  // Stop audio processing
  if (simAudioWorklet) {
    try {
      simAudioWorklet.source.disconnect();
      simAudioWorklet.processor.disconnect();
    } catch (e) {
      console.error('Error disconnecting Similarity audio worklet:', e);
    }
    simAudioWorklet = null;
  }

  // Close WebSocket
  if (simRealtimeWs) {
    simRealtimeWs.close();
    simRealtimeWs = null;
  }

  // Update UI
  if (transcribeRecordBtn) {
    transcribeRecordBtn.classList.remove('recording');
    transcribeRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> GPT-4o Transcribe';
  }
  if (transcribeIndicator) transcribeIndicator.classList.remove('active');
  if (transcribeStatus) transcribeStatus.classList.add('hidden');
}

/**
 * Connect to OpenAI Realtime Transcription API for Similarity mode
 */
async function connectSimRealtimeWebSocket() {
  try {
    const token = await getRealtimeToken();

    simRealtimeWs = new WebSocket(
      'wss://api.openai.com/v1/realtime?intent=transcription',
      ['realtime', `openai-insecure-api-key.${token}`, 'openai-beta.realtime-v1']
    );

    simRealtimeWs.onopen = () => {
      console.log('Connected to OpenAI Realtime Transcription API (Similarity)');

      // Configure transcription session
      simRealtimeWs.send(JSON.stringify({
        type: 'transcription_session.update',
        session: {
          input_audio_format: 'pcm16',
          input_audio_transcription: {
            model: 'gpt-4o-mini-transcribe',
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 800
          }
        }
      }));
    };

    simRealtimeWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleSimRealtimeMessage(data);
    };

    simRealtimeWs.onerror = (error) => {
      console.error('Similarity WebSocket error:', error);
      showStatus('Connection error', 'error');
      if (transcribeStatusText && simIsRecording) {
        transcribeStatusText.textContent = 'Connection error';
      }
    };

    simRealtimeWs.onclose = () => {
      console.log('Similarity WebSocket closed');
      if (simIsRecording && !gameWon) {
        setTimeout(() => {
          if (simIsRecording && !gameWon) {
            connectSimRealtimeWebSocket();
          }
        }, 1000);
      }
    };

    return true;
  } catch (error) {
    console.error('Failed to connect Similarity WebSocket:', error);
    return false;
  }
}

/**
 * Handle messages from OpenAI Realtime Transcription API (Similarity)
 */
function handleSimRealtimeMessage(data) {
  switch (data.type) {
    case 'transcription_session.created':
    case 'transcription_session.updated':
      console.log('Similarity transcription session configured:', data.type);
      if (transcribeStatusText && simIsRecording) {
        transcribeStatusText.textContent = 'Streaming... speak your guesses';
      }
      break;

    case 'conversation.item.input_audio_transcription.delta':
      if (data.delta) {
        simStreamingTranscript += data.delta;
      }
      break;

    case 'conversation.item.input_audio_transcription.completed':
      if (data.transcript) {
        console.log('Similarity completed transcript:', data.transcript);
        processSimCompletedTranscript(data.transcript);
      }
      break;

    case 'input_audio_buffer.speech_started':
      if (transcribeStatusText && simIsRecording) {
        transcribeStatusText.textContent = 'Listening...';
      }
      break;

    case 'input_audio_buffer.speech_stopped':
      if (transcribeStatusText && simIsRecording) {
        transcribeStatusText.textContent = 'Processing...';
      }
      break;

    case 'error':
      console.error('Similarity realtime error:', data);
      showStatus('Transcription error', 'error');
      if (transcribeStatusText && simIsRecording) {
        transcribeStatusText.textContent = 'Transcription error';
      }
      break;
  }
}

/**
 * Process a completed transcript segment into similarity guesses
 */
function processSimCompletedTranscript(transcript) {
  if (!transcript || !transcript.trim() || gameWon) return;

  const rawWords = transcript.toLowerCase().split(/\s+/);
  const words = [];

  for (const word of rawWords) {
    const cleaned = word.replace(/[^a-z]/g, '');
    if (cleaned.length >= 3 && /^[a-z]+$/.test(cleaned) && !STOP_WORDS.has(cleaned)) {
      words.push(cleaned);
    }
  }

  if (words.length > 0) {
    console.log('Similarity transcript words:', words);
    for (const word of words) {
      if (!guesses.some(g => g.word === word) && !guessQueue.includes(word)) {
        submitGuess(word);
      }
    }
  }

  simStreamingTranscript = '';
  if (transcribeStatusText && simIsRecording) {
    transcribeStatusText.textContent = 'Streaming... speak your guesses';
  }
}

/**
 * Start streaming audio to OpenAI for Similarity mode
 */
async function startSimAudioStreaming() {
  if (!audioContext || !audioStream || !simRealtimeWs) return;

  const source = audioContext.createMediaStreamSource(audioStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!simIsRecording || !simRealtimeWs || simRealtimeWs.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);
    const base64 = encodeFloat32ToPcm16Base64(inputData);

    simRealtimeWs.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: base64
    }));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  simAudioWorklet = { source, processor };
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

// Transcribe record button click
if (transcribeRecordBtn) {
  transcribeRecordBtn.addEventListener('click', toggleSimRecording);
}

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

// ============================================================
// Questions Mode
// ============================================================

// Questions mode state
let currentMode = 'similarity'; // 'similarity' or 'questions'
let qaHistory = []; // Array of { question, answer, timestamp, won }
let questionsWon = false;
let questionsSecretWord = null;

// Audio recording state - WebSocket streaming to OpenAI Realtime API
let isRecording = false;
let audioStream = null;
let audioContext = null;
let audioWorklet = null;
let realtimeWs = null;  // WebSocket to OpenAI
let realtimeToken = null;
let realtimeTokenExpiry = 0;
let streamingTranscript = '';  // Running transcript for display

// Anchor & Append: Smart multi-transcript question combining
// Sends complete questions immediately, buffers fragments, sends one clean revision
let currentQuestion = null;  // { id, text, lastUpdated, sent }
let revisionBuffer = [];     // Fragments waiting to be appended
let revisionTimer = null;    // Debounce timer for revisions

// Configuration for transcript combining
const REVISION_WINDOW_MS = 2500;  // How long after Q1 can we append continuations?
const DEBOUNCE_MS = 600;          // Wait time to stitch fragmented thoughts
const CONNECTOR_REGEX = /^(like|um|so|and|but|or|because|i mean|you know|specifically|actually|well|basically|essentially)\b/i;

// Questions mode DOM elements (lazy loaded)
let questionsContainer, audioRecordBtn, audioStatusEl, audioStatusText;
let audioIndicator, questionInput, askBtn, qaHistoryEl, noQuestionsEl;
let transcriptionPreview, transcriptionText, questionCountEl;
let modeSimBtn, modeQuestionsBtn;
let questionsHintBtn, questionsHintDisplay;

/**
 * Initialize Questions mode DOM references
 */
function initQuestionsModeDOM() {
  questionsContainer = document.getElementById('questions-container');
  audioRecordBtn = document.getElementById('audio-record-btn');
  audioStatusEl = document.getElementById('audio-status');
  audioStatusText = document.getElementById('audio-status-text');
  audioIndicator = document.getElementById('audio-indicator');
  questionInput = document.getElementById('question-input');
  askBtn = document.getElementById('ask-btn');
  qaHistoryEl = document.getElementById('qa-history');
  noQuestionsEl = document.getElementById('no-questions');
  transcriptionPreview = document.getElementById('transcription-preview');
  transcriptionText = document.getElementById('transcription-text');
  questionCountEl = document.getElementById('question-count');
  modeSimBtn = document.getElementById('mode-similarity');
  modeQuestionsBtn = document.getElementById('mode-questions');
  questionsHintBtn = document.getElementById('questions-hint-btn');
  questionsHintDisplay = document.getElementById('questions-hint-display');

  // Set up event listeners
  if (audioRecordBtn) {
    audioRecordBtn.addEventListener('click', toggleRecording);
  }
  if (askBtn) {
    askBtn.addEventListener('click', () => submitQuestion());
  }
  if (questionInput) {
    questionInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') submitQuestion();
    });
  }
  if (modeSimBtn) {
    modeSimBtn.addEventListener('click', () => switchMode('similarity'));
  }
  if (modeQuestionsBtn) {
    modeQuestionsBtn.addEventListener('click', () => switchMode('questions'));
  }
  if (questionsHintBtn) {
    questionsHintBtn.addEventListener('click', getQuestionsHint);
  }
}

/**
 * Switch between Similarity and Questions modes
 */
function switchMode(mode) {
  currentMode = mode;

  // Update toggle buttons
  if (modeSimBtn) modeSimBtn.classList.toggle('active', mode === 'similarity');
  if (modeQuestionsBtn) modeQuestionsBtn.classList.toggle('active', mode === 'questions');

  // Elements to show/hide for Similarity mode
  const similarityElements = [
    document.querySelector('.input-area'),
    document.querySelector('.hint-buttons'),
    document.querySelector('.guess-table-container'),
    document.getElementById('recent-guesses'),
    document.getElementById('similarity-help'),
    document.getElementById('mic-status'),
    document.getElementById('transcribe-status'),     // Transcribe status indicator
    document.getElementById('hint-display'),          // Similarity hint display
  ];

  if (mode === 'similarity') {
    // Show similarity mode elements
    similarityElements.forEach(el => el?.classList.remove('hidden'));
    if (questionsContainer) questionsContainer.classList.add('hidden');

    // Stop Questions mode recording if active
    if (isRecording) {
      stopRecording();
    }

    // Clear any pending question state
    if (revisionTimer) {
      clearTimeout(revisionTimer);
      revisionTimer = null;
    }
    currentQuestion = null;
    revisionBuffer = [];
  } else {
    // Hide similarity mode elements
    similarityElements.forEach(el => el?.classList.add('hidden'));
    if (questionsContainer) questionsContainer.classList.remove('hidden');

    // Stop similarity mode Web Speech API mic if active
    if (micShouldBeActive) {
      micShouldBeActive = false;
      try { recognition?.abort(); } catch (e) {}
      micBtn?.classList.remove('listening');
    }

    // Stop similarity mode GPT-4o transcribe recording if active
    if (typeof stopSimRecording === 'function' && simIsRecording) {
      stopSimRecording();
    }

    // Render Q&A history
    renderQAHistory();
  }

  saveGameState();
}

// ============================================================
// Audio Recording - WebSocket Streaming to OpenAI Realtime API
// ============================================================

/**
 * Get ephemeral token from our backend for OpenAI Realtime API
 */
async function getRealtimeToken() {
  // Check if we have a valid token
  if (realtimeToken && Date.now() < realtimeTokenExpiry - 60000) {
    return realtimeToken;
  }

  try {
    const response = await fetch(`${API_BASE}/realtime-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error('Failed to get realtime token');
    }

    const data = await response.json();
    realtimeToken = data.clientSecret;
    realtimeTokenExpiry = data.expiresAt * 1000; // Convert to ms
    return realtimeToken;
  } catch (error) {
    console.error('Error getting realtime token:', error);
    throw error;
  }
}

/**
 * Initialize audio stream and AudioContext for PCM capture
 */
async function initAudioSystem() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    audioContext = new AudioContext({ sampleRate: 24000 });
    return true;
  } catch (error) {
    console.error('Failed to initialize audio system:', error);
    showStatus('Microphone access denied', 'error');
    return false;
  }
}

/**
 * Connect to OpenAI Realtime API via WebSocket
 */
async function connectRealtimeWebSocket() {
  try {
    const token = await getRealtimeToken();

    // Connect with subprotocols for browser auth
    // Use intent=transcription for transcription-only mode (20x cheaper!)
    realtimeWs = new WebSocket(
      'wss://api.openai.com/v1/realtime?intent=transcription',
      ['realtime', `openai-insecure-api-key.${token}`, 'openai-beta.realtime-v1']
    );

    realtimeWs.onopen = () => {
      console.log('Connected to OpenAI Realtime Transcription API');

      // Configure transcription session
      realtimeWs.send(JSON.stringify({
        type: 'transcription_session.update',
        session: {
          input_audio_format: 'pcm16',
          input_audio_transcription: {
            model: 'gpt-4o-mini-transcribe',
          },
          turn_detection: {
            type: 'server_vad',
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 800
          }
        }
      }));
    };

    realtimeWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealtimeMessage(data);
    };

    realtimeWs.onerror = (error) => {
      console.error('WebSocket error:', error);
      showStatus('Connection error', 'error');
    };

    realtimeWs.onclose = () => {
      console.log('WebSocket closed');
      if (isRecording) {
        // Reconnect if still recording
        setTimeout(() => {
          if (isRecording && !questionsWon) {
            connectRealtimeWebSocket();
          }
        }, 1000);
      }
    };

    return true;
  } catch (error) {
    console.error('Failed to connect WebSocket:', error);
    return false;
  }
}

/**
 * Handle messages from OpenAI Realtime Transcription API
 */
function handleRealtimeMessage(data) {
  switch (data.type) {
    case 'transcription_session.created':
    case 'transcription_session.updated':
      console.log('Transcription session configured:', data.type);
      if (data.type === 'transcription_session.updated') {
        audioStatusText.textContent = 'Streaming... speak your questions';
      }
      break;

    case 'conversation.item.input_audio_transcription.delta':
      // Incremental transcript update
      if (data.delta) {
        streamingTranscript += data.delta;
        updateTranscriptionPreview();
      }
      break;

    case 'conversation.item.input_audio_transcription.completed':
      // Final transcript for this speech segment
      if (data.transcript) {
        console.log('Completed transcript:', data.transcript);
        processCompletedTranscript(data.transcript);
      }
      break;

    case 'input_audio_buffer.speech_started':
      audioStatusText.textContent = 'Listening...';
      break;

    case 'input_audio_buffer.speech_stopped':
      audioStatusText.textContent = 'Processing...';
      break;

    case 'error':
      console.error('Realtime API error:', data.error);
      if (data.error?.message) {
        showStatus(`Error: ${data.error.message}`, 'error');
      }
      break;

    default:
      console.log('Realtime event:', data.type);
  }
}

/**
 * Update the transcription preview with recent 30 words
 */
function updateTranscriptionPreview() {
  if (!transcriptionText || !transcriptionPreview) return;

  // Show last 30 words
  const words = streamingTranscript.trim().split(/\s+/);
  const recent = words.slice(-30).join(' ');

  transcriptionText.textContent = recent || 'Listening...';
  transcriptionPreview.classList.remove('hidden');
}

/**
 * Check if a transcript looks like a complete question
 * Returns true if it should be sent immediately
 */
function looksLikeCompleteQuestion(text) {
  const t = text.trim();
  // Ends with question mark = definitely complete
  if (t.endsWith('?')) return true;
  // Very short without punctuation = fragment
  if (t.split(/\s+/).length <= 2) return false;
  // Has sentence-ending punctuation
  if (/[.!]$/.test(t) && t.split(/\s+/).length > 3) return true;
  // Starts with question word and is reasonably long
  if (/^(is|are|does|do|can|will|would|could|should|was|were|has|have|did|what|how|why|where|when|who)\b/i.test(t) && t.split(/\s+/).length > 3) return true;
  // Default: treat as incomplete if short
  return t.split(/\s+/).length > 4;
}

/**
 * Check if a transcript looks like a fragment/continuation
 */
function isFragment(text) {
  const t = text.trim();
  const words = t.split(/\s+/);
  // Short = fragment
  if (words.length < 4) return true;
  // Lacks sentence-ending punctuation = fragment
  if (!/[.?!]$/.test(t)) return true;
  return false;
}

/**
 * Check if this transcript is a continuation of the current question
 */
function isContinuation(text, now) {
  if (!currentQuestion) return false;
  if (now - currentQuestion.lastUpdated > REVISION_WINDOW_MS) return false;

  const t = text.trim();
  // Starts with connector word = continuation
  if (CONNECTOR_REGEX.test(t)) return true;
  // Is a fragment = continuation
  if (isFragment(t)) return true;
  return false;
}

/**
 * Generate a simple unique ID for question tracking
 */
function generateQuestionId() {
  return 'q_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
}

/**
 * Send a revision combining the anchor question + buffered fragments
 */
async function sendRevision() {
  if (!currentQuestion || revisionBuffer.length === 0) return;

  // Combine anchor + buffer
  const combinedText = `${currentQuestion.text} ${revisionBuffer.join(' ')}`;
  console.log(`🔄 Sending Revision [${currentQuestion.id}]: "${combinedText}"`);

  // Update the anchor
  currentQuestion.text = combinedText;
  currentQuestion.lastUpdated = Date.now();
  revisionBuffer = [];

  // Send to backend
  await sendQuestionToBackend(combinedText, currentQuestion.id);
}

/**
 * Send question text to backend for parsing and answering
 */
/**
 * Ask a single question to all models in parallel
 * Returns the results array for win/hint detection
 */
async function askSingleQuestion(question) {
  // Check for duplicates - same question text within 30 seconds
  const isDuplicate = qaHistory.some(
    existing => existing.question.toLowerCase() === question.toLowerCase() &&
                Date.now() - existing.timestamp < 30000
  );
  if (isDuplicate) return [];

  // Create placeholder entry with loading state
  const qaIndex = qaHistory.length;
  const placeholderModelAnswers = QUESTION_MODELS.map(model => ({
    model,
    answer: 'loading'
  }));

  qaHistory.push({
    question,
    answer: 'loading',
    modelAnswers: placeholderModelAnswers,
    timestamp: Date.now(),
    won: false
  });
  renderQAHistory();

  // Fire off parallel requests to all models
  const modelPromises = QUESTION_MODELS.map(async (model) => {
    try {
      const response = await fetch(`${API_BASE}/ask-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          model,
          game: currentGameIndex,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get answer');
      }

      const data = await response.json();
      updateModelAnswer(qaIndex, model, data.answer, data.won, data.secretWord);
      return data;
    } catch (error) {
      console.error(`Error asking ${model}:`, error);
      updateModelAnswer(qaIndex, model, 'N/A', false);
      return { model, answer: 'N/A', won: false };
    }
  });

  const results = await Promise.all(modelPromises);

  // Check if any model detected a hint request
  const hintResult = results.find(r => r.answer?.toLowerCase() === 'hint');
  if (hintResult) {
    qaHistory.splice(qaIndex, 1);
    renderQAHistory();
    return [{ isHint: true }];
  }

  return results;
}

async function sendQuestionToBackend(text, questionId) {
  if (questionsWon) return;

  // First, parse the transcript into individual questions
  try {
    const parseResponse = await fetch(`${API_BASE}/parse-questions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!parseResponse.ok) {
      throw new Error('Failed to parse questions');
    }

    const parseData = await parseResponse.json();

    if (parseData.rateLimited) {
      showStatus('Rate limited - try again in a minute', 'info');
      return;
    }

    const questions = parseData.questions || [];
    if (questions.length === 0) {
      // No valid questions found
      return;
    }

    // Process each question
    for (const question of questions) {
      if (questionsWon) break;

      const results = await askSingleQuestion(question);

      // Check for hint
      if (results.some(r => r.isHint)) {
        showStatus('Getting hint...', 'info');
        getQuestionsHint();
        return;
      }

      // Check for win
      const winResult = results.find(r => r.won);
      if (winResult) {
        handleQuestionsWin(winResult.secretWord);
        break;
      }

      // Check for rate limiting
      const rateLimited = results.some(r => r.rateLimited);
      if (rateLimited) {
        showStatus('Rate limited - try again in a minute', 'info');
        break;
      }
    }

    saveGameState();

  } catch (error) {
    console.error('Error sending question to backend:', error);
  }
}

/**
 * Process a completed transcript segment using Anchor & Append strategy
 * - Complete questions → send immediately (0ms latency)
 * - Fragments/continuations → buffer and send as one revision
 */
async function processCompletedTranscript(transcript) {
  if (!transcript || !transcript.trim() || questionsWon) return;

  const text = transcript.trim();
  const now = Date.now();

  // Reset streaming transcript for next segment
  streamingTranscript = '';

  console.log(`📝 Transcript: "${text}" | isContinuation: ${isContinuation(text, now)} | looksComplete: ${looksLikeCompleteQuestion(text)}`);

  // CASE 1: This is a continuation of the current question
  if (isContinuation(text, now)) {
    revisionBuffer.push(text);

    // If we already sent the anchor, schedule a debounced revision
    if (currentQuestion.sent) {
      if (revisionTimer) clearTimeout(revisionTimer);
      revisionTimer = setTimeout(sendRevision, DEBOUNCE_MS);
      console.log(`🐢 Buffered continuation, will revise in ${DEBOUNCE_MS}ms`);
    } else {
      // Anchor not sent yet - check if combined text is now complete
      const combinedText = `${currentQuestion.text} ${revisionBuffer.join(' ')}`;
      if (looksLikeCompleteQuestion(combinedText)) {
        // Combined is complete! Send it now
        currentQuestion.text = combinedText;
        currentQuestion.sent = true;
        currentQuestion.lastUpdated = now;
        revisionBuffer = [];
        console.log(`🚀 Combined question complete, sending: "${combinedText}"`);
        await sendQuestionToBackend(combinedText, currentQuestion.id);
      } else {
        // Still incomplete, wait for more
        if (revisionTimer) clearTimeout(revisionTimer);
        revisionTimer = setTimeout(async () => {
          // Timeout - send what we have even if incomplete
          const finalText = `${currentQuestion.text} ${revisionBuffer.join(' ')}`;
          currentQuestion.text = finalText;
          currentQuestion.sent = true;
          revisionBuffer = [];
          console.log(`⏰ Timeout, sending incomplete: "${finalText}"`);
          await sendQuestionToBackend(finalText, currentQuestion.id);
        }, DEBOUNCE_MS);
      }
    }
    return;
  }

  // CASE 2: New question (not a continuation)

  // First, flush any pending revision for the old question
  if (revisionBuffer.length > 0 && currentQuestion) {
    if (revisionTimer) clearTimeout(revisionTimer);
    await sendRevision();
  }

  // Start a new question
  const newId = generateQuestionId();
  currentQuestion = {
    id: newId,
    text: text,
    lastUpdated: now,
    sent: false
  };
  revisionBuffer = [];

  // Check if it looks complete
  if (looksLikeCompleteQuestion(text)) {
    // FAST PATH: Send immediately
    currentQuestion.sent = true;
    console.log(`🚀 Complete question, sending immediately: "${text}"`);
    await sendQuestionToBackend(text, newId);
  } else {
    // Incomplete - wait for more with a timeout
    console.log(`⏳ Incomplete question, waiting for more: "${text}"`);
    if (revisionTimer) clearTimeout(revisionTimer);
    revisionTimer = setTimeout(async () => {
      if (currentQuestion && !currentQuestion.sent) {
        const finalText = revisionBuffer.length > 0
          ? `${currentQuestion.text} ${revisionBuffer.join(' ')}`
          : currentQuestion.text;
        currentQuestion.text = finalText;
        currentQuestion.sent = true;
        revisionBuffer = [];
        console.log(`⏰ Timeout, sending: "${finalText}"`);
        await sendQuestionToBackend(finalText, currentQuestion.id);
      }
    }, DEBOUNCE_MS);
  }
}

/**
 * Start streaming audio to OpenAI
 */
async function startAudioStreaming() {
  if (!audioContext || !audioStream || !realtimeWs) return;

  const source = audioContext.createMediaStreamSource(audioStream);

  // Create ScriptProcessor for PCM capture (deprecated but widely supported)
  // AudioWorklet would be better but requires separate file
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!isRecording || !realtimeWs || realtimeWs.readyState !== WebSocket.OPEN) return;

    const inputData = e.inputBuffer.getChannelData(0);
    const base64 = encodeFloat32ToPcm16Base64(inputData);

    // Send to OpenAI
    realtimeWs.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: base64
    }));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  audioWorklet = { source, processor };
}

/**
 * Toggle audio recording on/off
 */
async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

/**
 * Start streaming recording with OpenAI Realtime API
 */
async function startRecording() {
  if (questionsWon) {
    showStatus('Game already won! Start a new game.', 'info');
    return;
  }

  // Update UI immediately
  audioRecordBtn.classList.add('recording');
  audioRecordBtn.innerHTML = '<span class="record-icon">&#x23F9;</span> Stop';
  audioIndicator.classList.add('active');
  audioStatusText.textContent = 'Connecting...';

  // Initialize audio if needed
  if (!audioStream) {
    const success = await initAudioSystem();
    if (!success) {
      audioRecordBtn.classList.remove('recording');
      audioRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> Start Recording';
      audioIndicator.classList.remove('active');
      return;
    }
  }

  // Resume audio context if suspended
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  // Connect to OpenAI
  const connected = await connectRealtimeWebSocket();
  if (!connected) {
    audioRecordBtn.classList.remove('recording');
    audioRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> Start Recording';
    audioIndicator.classList.remove('active');
    audioStatusText.textContent = 'Connection failed';
    return;
  }

  isRecording = true;
  streamingTranscript = '';

  // Wait for WebSocket to be ready, then start streaming
  const waitForOpen = setInterval(() => {
    if (realtimeWs && realtimeWs.readyState === WebSocket.OPEN) {
      clearInterval(waitForOpen);
      startAudioStreaming();
      audioStatusText.textContent = 'Streaming... speak your questions';
    }
  }, 100);

  // Start timer on first recording
  if (!timerStarted) {
    startTimer();
  }
}

/**
 * Stop streaming recording
 */
function stopRecording() {
  if (!isRecording) return;

  isRecording = false;

  // Flush any pending question/revision before stopping
  if (revisionTimer) {
    clearTimeout(revisionTimer);
    revisionTimer = null;
  }
  if (currentQuestion && !currentQuestion.sent) {
    // Send what we have
    const finalText = revisionBuffer.length > 0
      ? `${currentQuestion.text} ${revisionBuffer.join(' ')}`
      : currentQuestion.text;
    if (finalText.trim()) {
      console.log(`🛑 Recording stopped, flushing: "${finalText}"`);
      sendQuestionToBackend(finalText, currentQuestion.id);
    }
  } else if (revisionBuffer.length > 0 && currentQuestion) {
    // Anchor was sent but we have buffered revisions
    sendRevision();
  }
  // Clear state
  currentQuestion = null;
  revisionBuffer = [];

  // Stop audio processing
  if (audioWorklet) {
    audioWorklet.source.disconnect();
    audioWorklet.processor.disconnect();
    audioWorklet = null;
  }

  // Close WebSocket
  if (realtimeWs) {
    realtimeWs.close();
    realtimeWs = null;
  }

  // Update UI
  audioRecordBtn.classList.remove('recording');
  audioRecordBtn.innerHTML = '<span class="record-icon">&#x23FA;</span> Start Recording';
  audioIndicator.classList.remove('active');
  audioStatusText.textContent = 'Click to start recording';

  // Hide transcription preview after a delay
  setTimeout(() => {
    if (!isRecording && transcriptionPreview) {
      transcriptionPreview.classList.add('hidden');
    }
  }, 3000);
}

// ============================================================
// Question Submission
// ============================================================

/**
 * Submit a typed question
 */
// Available models for question answering (fetched from backend)
const QUESTION_MODELS = [
  'openai/gpt-5.2',
  'google/gemini-3-flash-preview',
  'anthropic/claude-sonnet-4.5'
];

async function submitQuestion(questionText = null) {
  const text = (questionText || questionInput?.value || '').trim();

  if (!text) return;

  if (questionInput) questionInput.value = '';

  if (questionsWon) {
    showStatus('Game already won! Start a new game.', 'info');
    return;
  }

  // Start timer on first question
  if (!timerStarted) {
    startTimer();
  }

  showStatus('Parsing...', 'info');

  try {
    // Parse text into individual questions
    const parseResponse = await fetch(`${API_BASE}/parse-questions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!parseResponse.ok) {
      throw new Error('Failed to parse questions');
    }

    const parseData = await parseResponse.json();

    if (parseData.rateLimited) {
      showStatus('Rate limited - try again later', 'info');
      return;
    }

    const questions = parseData.questions || [];
    if (questions.length === 0) {
      showStatus('No valid questions found', 'info');
      return;
    }

    showStatus('Asking...', 'info');

    // Process each parsed question
    for (const question of questions) {
      if (questionsWon) break;

      const results = await askSingleQuestion(question);

      // Check for hint
      if (results.some(r => r.isHint)) {
        showStatus('Getting hint...', 'info');
        getQuestionsHint();
        return;
      }

      // Check for win
      const winResult = results.find(r => r.won);
      if (winResult) {
        handleQuestionsWin(winResult.secretWord);
        break;
      }

      // Check for rate limiting
      const rateLimited = results.some(r => r.rateLimited);
      if (rateLimited) {
        showStatus('Rate limited - try again later', 'info');
        break;
      }
    }

    showStatus('', '');
    saveGameState();

  } catch (error) {
    console.error('Error submitting question:', error);
    showStatus('Failed to get answer', 'error');
  }
}

/**
 * Update a specific model's answer in an existing QA entry
 */
function updateModelAnswer(qaIndex, model, answer, won = false, secretWord = null) {
  if (qaIndex >= qaHistory.length) return;

  const qa = qaHistory[qaIndex];
  const modelIdx = qa.modelAnswers.findIndex(ma => ma.model === model);

  if (modelIdx !== -1) {
    qa.modelAnswers[modelIdx].answer = answer;
  }

  // Update the primary answer to the first non-loading answer
  const firstAnswer = qa.modelAnswers.find(ma => ma.answer !== 'loading');
  if (firstAnswer && qa.answer === 'loading') {
    qa.answer = firstAnswer.answer;
  }

  // Update won status if any model detected a win
  if (won) {
    qa.won = true;
  }

  renderQAHistory();
}

/**
 * Add a Q&A pair to history
 */
function addQAToHistory(question, answer, won = false, modelAnswers = []) {
  qaHistory.push({
    question,
    answer,
    modelAnswers,
    timestamp: Date.now(),
    won
  });

  renderQAHistory();
  saveGameState();
}

/**
 * Get gravity weight for answer type
 * Higher weight = more important, stays near top longer
 */
function getAnswerGravityWeight(answer) {
  switch (answer.toLowerCase()) {
    case 'so close': return 5;
    case 'yes': return 4;
    case 'hard no': return 3.5; // Strong signal, slightly more important than regular no
    case 'no': return 3;
    case 'maybe': return 2;
    default: return 1; // N/A
  }
}

/**
 * Calculate gravity-weighted sort score for Q&A item
 * Combines recency with answer importance
 */
function calculateGravityScore(qa, index, totalItems, now) {
  // Recency score: newer items get higher base score (0 to 1)
  const recencyScore = index / totalItems;

  // Gravity weight based on answer type
  const weight = getAnswerGravityWeight(qa.answer);

  // Time decay factor: how long ago was this question asked
  // More recent = higher decay multiplier
  const ageMs = now - qa.timestamp;
  const ageMinutes = ageMs / 60000;

  // Decay function: importance decays slower for higher-weighted answers
  // Base decay per minute, reduced by weight
  // Higher weight = slower decay
  const decayRate = 0.15 / weight; // N/A decays 5x faster than "so close"
  const decayMultiplier = Math.exp(-decayRate * ageMinutes);

  // Combined score: recency + (weight bonus * decay)
  // Weight bonus keeps important answers buoyant (reduced for more recency bias)
  const weightBonus = (weight - 1) * 0.15 * decayMultiplier;

  return recencyScore + weightBonus;
}

/**
 * Render Q&A history with gravity-weighted sorting
 */
function renderQAHistory() {
  if (!qaHistoryEl || !noQuestionsEl) return;

  if (qaHistory.length === 0) {
    noQuestionsEl.classList.remove('hidden');
    questionCountEl.textContent = '0';
    qaHistoryEl.innerHTML = '<p id="no-questions" class="no-questions">No questions yet. Start asking!</p>';
    return;
  }

  noQuestionsEl.classList.add('hidden');
  questionCountEl.textContent = qaHistory.length;

  // Create array with original indices for gravity calculation
  const now = Date.now();
  const itemsWithScores = qaHistory.map((qa, index) => ({
    qa,
    originalIndex: index,
    gravityScore: calculateGravityScore(qa, index, qaHistory.length, now)
  }));

  // Sort by gravity score (highest first = most recent/important)
  itemsWithScores.sort((a, b) => b.gravityScore - a.gravityScore);

  qaHistoryEl.innerHTML = itemsWithScores.map((item) => {
    const qa = item.qa;
    const answerClass = getAnswerClass(qa.answer);
    const wonClass = qa.won ? ' won' : '';
    const questionNum = item.originalIndex + 1;

    // Render multi-model answers if available, otherwise fall back to single answer
    const modelAnswersHtml = qa.modelAnswers?.length > 0
      ? qa.modelAnswers.map(ma => `
          <div class="model-answer ${getAnswerClass(ma.answer)}">
            <span class="model-name">${getModelShortName(ma.model)}</span>
            <span class="model-result">${ma.answer.toUpperCase()}</span>
          </div>
        `).join('')
      : `<div class="qa-answer ${answerClass}">${qa.answer.toUpperCase()}</div>`;

    return `
      <div class="qa-item${wonClass}">
        <div class="qa-number">#${questionNum}</div>
        <div class="qa-content">
          <div class="qa-question">${escapeHtml(qa.question)}</div>
          <div class="qa-answers-row">${modelAnswersHtml}</div>
        </div>
      </div>
    `;
  }).join('');
}

/**
 * Get CSS class for answer
 */
function getAnswerClass(answer) {
  switch (answer.toLowerCase()) {
    case 'yes': return 'answer-yes';
    case 'no': return 'answer-no';
    case 'hard no': return 'answer-hardno';
    case 'maybe': return 'answer-maybe';
    case 'so close': return 'answer-close';
    case 'hint': return 'answer-hint';
    case 'loading': return 'answer-loading';
    default: return 'answer-na';
  }
}

/**
 * Get short display name for model
 */
function getModelShortName(model) {
  if (model.includes('gpt')) return 'GPT';
  if (model.includes('gemini')) return 'Gemini';
  if (model.includes('claude')) return 'Claude';
  return model.split('/')[1] || model;
}

/**
 * Request an LLM-generated hint for Questions mode (2 minute penalty)
 */
async function getQuestionsHint() {
  if (questionsWon) {
    showStatus('Game already won!', 'info');
    return;
  }

  if (qaHistory.length < 1) {
    showStatus('Ask at least one question first!', 'info');
    return;
  }

  // Start timer if not started (penalty still applies)
  if (!timerStarted) {
    startTimer();
  }

  // Add 2 minute (120 second) penalty for LLM hint
  addTimerPenalty(120);

  // Get recent Q&A (last 10 for context)
  const recentQA = qaHistory.slice(-10).map(qa => ({
    question: qa.question,
    answer: qa.answer
  }));

  try {
    if (questionsHintBtn) questionsHintBtn.disabled = true;
    showStatus('Getting hint from AI...', 'info');

    const response = await fetch(`${API_BASE}/questions-hint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recentQA, game: currentGameIndex }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to get hint');
    }

    const data = await response.json();

    if (data.rateLimited) {
      showStatus('LLM hint rate limited, try again later', 'info');
    } else {
      // Show hint in the Questions mode hint display
      if (questionsHintDisplay) {
        questionsHintDisplay.textContent = data.hint;
        questionsHintDisplay.classList.remove('hidden');
      }
      showStatus('', '');
    }
  } catch (error) {
    console.error('Error getting Questions hint:', error);
    showStatus(`Failed to get hint: ${error.message}`, 'error');
  } finally {
    if (questionsHintBtn) questionsHintBtn.disabled = false;
  }
}

/**
 * Handle winning in Questions mode
 */
function handleQuestionsWin(secretWord) {
  questionsWon = true;
  questionsSecretWord = secretWord;

  // Stop recording
  stopRecording();

  // Stop timer
  stopTimer();
  const elapsed = getElapsedSeconds();
  const finalTier = getCurrentTier(elapsed);

  // Show win banner
  winWord.textContent = secretWord.toUpperCase();
  winGuesses.textContent = qaHistory.length + ' questions';
  winTime.textContent = formatTime(elapsed);
  winTierEl.textContent = `${finalTier.emoji} ${finalTier.name}`;
  winTierEl.style.color = finalTier.color;
  winBanner.classList.remove('hidden');

  showStatus('You got it!', 'success');
  saveGameState();
}

/**
 * Reset Questions mode state
 */
function resetQuestionsMode() {
  qaHistory = [];
  questionsWon = false;
  questionsSecretWord = null;
  stopRecording();
  renderQAHistory();
}

// ============================================================
// Updated Save/Load for Questions Mode
// ============================================================

// Override saveGameState to include Questions mode
const originalSaveGameState = saveGameState;
saveGameState = function() {
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
    // Questions mode state
    currentMode,
    qaHistory,
    questionsWon,
    questionsSecretWord,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.error('Failed to save game state:', e);
  }
};

// Override loadGameState to restore Questions mode
const originalLoadGameState = loadGameState;
loadGameState = function() {
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

    // Questions mode state
    currentMode = state.currentMode || 'similarity';
    qaHistory = state.qaHistory || [];
    questionsWon = state.questionsWon || false;
    questionsSecretWord = state.questionsSecretWord || null;

    return true;
  } catch (e) {
    console.error('Failed to load game state:', e);
    return false;
  }
};

// Override resetGame to also reset Questions mode
const originalResetGame = resetGame;
resetGame = function() {
  guesses = [];
  recentAttempts = [];
  gameWon = false;
  guessQueue = [];
  winBanner.classList.add('hidden');
  hintDisplay.classList.add('hidden');
  hintDisplay.textContent = '';
  resetTimer();
  resetRerankState();
  resetQuestionsMode();
  // Stop Similarity mode transcribe recording if active
  if (typeof stopSimRecording === 'function') {
    stopSimRecording();
  }
  setInputDisabled(false);
  showStatus('Game reset!', 'success');
  renderGuesses();
  renderQAHistory();
  saveGameState();

  if (currentMode === 'similarity') {
    guessInput.focus();
  } else if (questionInput) {
    questionInput.focus();
  }

  setTimeout(() => {
    if (!gameWon && !questionsWon) showStatus('', '');
  }, 2000);
};

// Override newRandomGame to also reset Questions mode
const originalNewRandomGame = newRandomGame;
newRandomGame = function() {
  currentGameIndex = Math.floor(Math.random() * 1_000_000_000) + 1;
  guesses = [];
  recentAttempts = [];
  gameWon = false;
  guessQueue = [];
  winBanner.classList.add('hidden');
  hintDisplay.classList.add('hidden');
  hintDisplay.textContent = '';
  resetTimer();
  resetRerankState();
  resetQuestionsMode();
  // Stop Similarity mode transcribe recording if active
  if (typeof stopSimRecording === 'function') {
    stopSimRecording();
  }
  setInputDisabled(false);
  showStatus('New random word!', 'success');
  renderGuesses();
  renderQAHistory();
  saveGameState();

  if (currentMode === 'similarity') {
    guessInput.focus();
  } else if (questionInput) {
    questionInput.focus();
  }

  setTimeout(() => {
    if (!gameWon && !questionsWon) showStatus('', '');
  }, 2000);
};

// Initialize Questions mode on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  initQuestionsModeDOM();

  // Restore mode from saved state
  if (currentMode === 'questions') {
    switchMode('questions');

    // Restore Questions win state
    if (questionsWon && questionsSecretWord) {
      const elapsed = getElapsedSeconds();
      const finalTier = getCurrentTier(elapsed);
      winWord.textContent = questionsSecretWord.toUpperCase();
      winGuesses.textContent = qaHistory.length + ' questions';
      winTime.textContent = formatTime(elapsed);
      winTierEl.textContent = `${finalTier.emoji} ${finalTier.name}`;
      winTierEl.style.color = finalTier.color;
      winBanner.classList.remove('hidden');
      timerDisplay.classList.remove('hidden');
      updateTimerDisplay();
    }
  }
});

// Expose functions to window for button onclick
window.resetGame = resetGame;
window.newRandomGame = newRandomGame;
window.submitGuess = submitGuess;
window.startMicGuess = startMicGuess;
window.switchMode = switchMode;
window.toggleRecording = toggleRecording;
window.submitQuestion = submitQuestion;
