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
let gameWon = false;
let isProcessingGuess = false;
let guessQueue = []; // Queue of words waiting to be submitted
let currentGameIndex = 0; // 0 = word of the day, 1+ = random games

// LocalStorage key (includes date so it resets daily)
const STORAGE_KEY = 'semanticTrail_' + new Date().toISOString().split('T')[0];

// LLM Re-ranking state
let rerankInterval = null;
let lastGuessTime = null;
const RERANK_INTERVAL_MS = 30000; // 30 seconds
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
const revealBtn = document.getElementById('reveal-btn');
const hintDisplay = document.getElementById('hint-display');
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
// LLM Re-ranking Functions (Smart Anchor Algorithm)
// ============================================================

/**
 * Start the re-ranking interval
 */
function startRerankInterval() {
  if (rerankInterval || DEMO_MODE) return;
  console.log('Starting Smart LLM re-rank interval');
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
 * Reset re-ranking state
 */
function resetRerankState() {
  stopRerankInterval();
  lastGuessTime = null;
  rerankGroupId = 0;
  previousTop20Hash = '';
  // Clear LLM metadata from guesses
  for (const g of guesses) {
    delete g.llmRank;
    delete g.llmScore;
    delete g.rerankGroupId;
  }
}

/**
 * Main Loop: Check if we need to re-rank
 */
async function checkAndRerank() {
  if (gameWon || guesses.length < 3) return;

  // Inactivity check
  if (lastGuessTime && Date.now() - lastGuessTime > INACTIVITY_TIMEOUT_MS) {
    stopRerankInterval();
    return;
  }

  // Build the "Smart Batch" - skips stable blocks, explores new words
  const batch = buildSmartBatch(20);

  if (batch.length < 2) {
    return;
  }

  // Hash check to prevent spamming the exact same request
  const batchHash = batch.map(g => g.word).join(',');
  if (batchHash === previousTop20Hash) {
    return;
  }

  console.log('Smart Rerank batch:', batch.map(b => `${b.word}(${b.rerankGroupId || 'new'})`).join(', '));
  await performSmartRerank(batch, batchHash);
}

/**
 * ALGORITHM: Build a batch of words to send to LLM
 *
 * Logic: Iterate down the leaderboard. If we find a contiguous block of words
 * that were previously sorted together (same rerankGroupId), keep only the
 * HEAD and TAIL as anchors, skip the body. Use saved slots for lower words.
 */
function buildSmartBatch(maxSize) {
  // Get current leaderboard sorted by effective score
  const sortedGuesses = [...guesses]
    .filter(g => !g.isCorrect)
    .sort((a, b) => {
      const scoreA = a.llmScore ?? a.score;
      const scoreB = b.llmScore ?? b.score;
      return scoreB - scoreA;
    });

  const batch = [];
  let i = 0;

  while (batch.length < maxSize && i < sortedGuesses.length) {
    const current = sortedGuesses[i];

    // If never ranked by LLM, always include it
    if (!current.rerankGroupId) {
      batch.push(current);
      i++;
      continue;
    }

    // Look ahead: find the extent of this "Stable Block"
    // A stable block is adjacent words sharing the same rerankGroupId
    let blockEnd = i;
    while (
      blockEnd + 1 < sortedGuesses.length &&
      sortedGuesses[blockEnd + 1].rerankGroupId === current.rerankGroupId
    ) {
      blockEnd++;
    }

    const blockSize = blockEnd - i + 1;

    // Optimization: If block is large (>=3), skip the middle
    if (blockSize >= 3) {
      // Add Head anchor (top of block)
      batch.push(sortedGuesses[i]);

      // Add Tail anchor (bottom of block) if we have room
      if (batch.length < maxSize) {
        batch.push(sortedGuesses[blockEnd]);
      }

      // Skip past the whole block
      i = blockEnd + 1;
    } else {
      // Small block, just add it
      batch.push(current);
      i++;
    }
  }

  return batch;
}

/**
 * Execute the Rerank and Apply Min-Max Normalization
 */
async function performSmartRerank(batch, batchHash) {
  // Determine Ceiling and Floor from the batch's embedding scores
  const embeddingScores = batch.map(g => g.score); // Original embedding scores
  const ceiling = Math.max(...embeddingScores);
  const floor = Math.min(...embeddingScores);

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

    // Increment Group ID for this new stable set
    rerankGroupId++;

    const resultCount = data.rankings.length;

    // Apply Min-Max Normalization to LLM rankings
    for (let index = 0; index < data.rankings.length; index++) {
      const r = data.rankings[index];
      const guessObj = guesses.find(g => g.word.toLowerCase() === r.word.toLowerCase());
      if (!guessObj) continue;

      // Calculate normalized score
      // Rank 1 (index 0) gets Ceiling, Last Rank gets Floor
      let newScore;
      if (resultCount === 1) {
        newScore = ceiling;
      } else {
        const ratio = (resultCount - 1 - index) / (resultCount - 1); // 1.0 down to 0.0
        newScore = floor + (ratio * (ceiling - floor));
      }

      // Update the guess with LLM-derived score
      guessObj.llmScore = Math.round(newScore * 100) / 100;
      guessObj.llmRank = index + 1;
      guessObj.rerankGroupId = rerankGroupId; // Mark as part of this stable group
    }

    previousTop20Hash = batchHash;
    renderGuesses();
    saveGameState();
    console.log(`Smart rerank complete: ${resultCount} words, range ${ceiling}->${floor}`);

  } catch (error) {
    console.error('Error in performSmartRerank:', error);
  }
}

// ============================================================
// Core Game Logic
// ============================================================

/**
 * Submit a guess to the backend (queues if already processing)
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

  // Check if already guessed or already in queue
  if (guesses.some(g => g.word === guess) || guessQueue.includes(guess)) {
    showStatus(`Already guessed "${guess}"`, 'info');
    return;
  }

  // If currently processing, add to queue
  if (isProcessingGuess) {
    guessQueue.push(guess);
    showStatus(`Queued "${guess}" (${guessQueue.length} waiting)...`, 'info');
    return;
  }

  // Process this guess
  await processGuess(guess);

  // Process any queued guesses
  while (guessQueue.length > 0 && !gameWon) {
    const nextGuess = guessQueue.shift();
    // Skip if it was guessed while in queue
    if (!guesses.some(g => g.word === nextGuess)) {
      await processGuess(nextGuess);
    }
  }
}

/**
 * Process a single guess (internal - called by submitGuess)
 */
async function processGuess(guess) {
  isProcessingGuess = true;
  showStatus(`Thinking... "${guess}"`, 'info');

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

    // Add to guesses (compute bucket on frontend)
    guesses.push({
      word: result.guess,
      similarity: result.similarity,
      score: result.score,
      bucket: getBucket(result.score),
      isCorrect: result.isCorrect || false,
    });

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
      const bucketLabel = result.bucket || getBucket(result.score);
      const queueInfo = guessQueue.length > 0 ? ` (${guessQueue.length} queued)` : '';
      showStatus(`"${result.guess}" → ${result.score}/100 (${bucketLabel}) — Rank #${rank}/${guesses.length}${queueInfo}`, 'info');
    }

    // Update UI
    renderGuesses();
    saveGameState();

  } catch (error) {
    console.error('Error submitting guess:', error);
    showStatus(`Error: ${error.message}`, 'error');
  } finally {
    isProcessingGuess = false;
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
  gameWon = false;
  guessQueue = [];
  isProcessingGuess = false;
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
  gameWon = false;
  guessQueue = [];
  isProcessingGuess = false;
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
 * Render the guesses table
 */
function renderGuesses() {
  if (guesses.length === 0) {
    noGuessesEl.classList.remove('hidden');
    guessTable.classList.add('hidden');
    return;
  }

  noGuessesEl.classList.add('hidden');
  guessTable.classList.remove('hidden');

  // Sort by best available score (llmScore if exists, else embedding score)
  // This allows new words to slot into the appropriate position
  const sorted = [...guesses].sort((a, b) => {
    const scoreA = a.llmScore ?? a.score;
    const scoreB = b.llmScore ?? b.score;
    return scoreB - scoreA; // Descending (highest first)
  });

  // Build table rows
  guessTbody.innerHTML = sorted.map((g, index) => {
    const isLatest = g.word === guesses[guesses.length - 1].word;
    const displayScore = g.llmScore ?? g.score;
    const bucketLabel = g.isCorrect ? 'CORRECT!' : (g.bucket ?? getBucket(displayScore));
    const bucketClass = getBucketClass(displayScore, g.isCorrect);
    const barClass = getBarClass(displayScore, g.isCorrect);

    // Show LLM score if available (yellow, rounded)
    const llmDisplay = g.llmScore !== undefined
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

// Expose resetGame to window for the button onclick
window.resetGame = resetGame;
window.submitGuess = submitGuess;
window.startMicGuess = startMicGuess;
