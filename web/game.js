/**
 * Semantic Trail Web Game - Frontend JavaScript
 *
 * Handles:
 * - Submitting guesses to the Worker backend
 * - Updating the guess table UI
 * - Web Speech API microphone input
 */

// Configuration
// Change this URL to your deployed Cloudflare Worker URL
const API_URL = 'https://semantic-trail-backend.yourname.workers.dev/score';

// For demo/testing without backend, set this to true
const DEMO_MODE = false;

// Game state
let guesses = []; // Array of { word, similarity, score, bucket, isCorrect }
let gameWon = false;

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

// ============================================================
// Core Game Logic
// ============================================================

/**
 * Submit a guess to the backend
 */
async function submitGuess(word = null) {
  const guess = (word || guessInput.value).trim().toLowerCase();

  if (!guess) {
    showStatus('Please enter a word', 'error');
    return;
  }

  if (!/^[a-z]+$/.test(guess)) {
    showStatus('Please enter letters only', 'error');
    return;
  }

  // Check if already guessed
  if (guesses.some(g => g.word === guess)) {
    showStatus(`Already guessed "${guess}"`, 'info');
    guessInput.value = '';
    return;
  }

  // Disable input while processing
  setInputDisabled(true);
  showStatus('Thinking...', 'info');

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
        body: JSON.stringify({ guess }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Request failed');
      }

      result = await response.json();
    }

    // Add to guesses
    guesses.push({
      word: result.guess,
      similarity: result.similarity,
      score: result.score,
      bucket: result.bucket,
      isCorrect: result.isCorrect || false,
    });

    // Check for win
    if (result.isCorrect) {
      handleWin(result.guess);
    } else {
      // Show word, score, bucket, and rank in status
      const rank = getRank(result.score);
      const bucketLabel = result.bucket || getBucket(result.score);
      showStatus(`"${result.guess}" → ${result.score}/100 (${bucketLabel}) — Rank #${rank}/${guesses.length}`, 'info');
    }

    // Update UI
    renderGuesses();
    guessInput.value = '';

  } catch (error) {
    console.error('Error submitting guess:', error);
    showStatus(`Error: ${error.message}`, 'error');
  } finally {
    setInputDisabled(false);
    guessInput.focus();
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
  winWord.textContent = word.toUpperCase();
  winGuesses.textContent = guesses.length;
  winBanner.classList.remove('hidden');
  setInputDisabled(true);
  showStatus('', '');
}

/**
 * Reset the game
 */
function resetGame() {
  guesses = [];
  gameWon = false;
  winBanner.classList.add('hidden');
  setInputDisabled(false);
  showStatus('New game started!', 'success');
  renderGuesses();
  guessInput.focus();

  // Clear status after a moment
  setTimeout(() => {
    if (!gameWon) showStatus('', '');
  }, 2000);
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

  // Sort by score descending
  const sorted = [...guesses].sort((a, b) => b.score - a.score);

  // Build table rows
  guessTbody.innerHTML = sorted.map((g, index) => {
    const isLatest = g.word === guesses[guesses.length - 1].word;
    const bucketClass = getBucketClass(g.score, g.isCorrect);
    const barClass = getBarClass(g.score, g.isCorrect);

    return `
      <tr class="${isLatest ? 'latest' : ''}">
        <td>${index + 1}</td>
        <td class="word-cell">${escapeHtml(g.word)}</td>
        <td class="score-cell">${g.score}</td>
        <td class="similarity-cell">${g.similarity.toFixed(3)}</td>
        <td>
          <div class="score-bar">
            <div class="score-bar-fill ${barClass}" style="width: ${g.score}%"></div>
          </div>
          <span class="bucket-cell ${bucketClass}">${g.isCorrect ? 'CORRECT!' : g.bucket}</span>
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

// Focus input on page load
document.addEventListener('DOMContentLoaded', () => {
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
