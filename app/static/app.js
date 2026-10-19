/* ============================================================================
   Alter-Ego · Ultra-Minimalist Gemini Argon Frontend Logic
   - 100% Live execution on local Ollama (no instant/staged toggle)
   - No scenario pills in conversation view
   - Discreet collapsible tool trace with rich step cards on expand
   - Structured Légifrance source inspector below the grey hierarchy box
   ============================================================================ */

const SESSION_HISTORY_KEY = 'alter_ego_session_history_v1';

const state = {
  mode: 'agent',          // 'agent' | 'rag'
  selectedModel: '',
  localModels: [],
  apiModels: [],
  apiCatalog: {},
  hasApiKey: false,
  configuredProviders: [], // ['openrouter', 'anthropic']
  apiPopoverProvider: 'openrouter', // 'openrouter' | 'anthropic'
  scenarios: [],
  activeScenarioId: null,
  currentTrace: null,
  activeTurns: [],        // Array of completed turn traces in the currently open conversation
  timerInterval: null,
  chatHistory: [],        // Session-persisted completed conversations only (never empty chats)
  activeChatId: null,     // null when on empty Home stage
};

const SUGGESTION_ICONS = [
  // Sparkle / star icon
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2L12 3z" stroke-linejoin="round"/></svg>`,
  // Curved arrow icon
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 10H5a4 4 0 0 0 0 8h11m0 0-3-3m3 3-3 3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  // Clock / historical icon
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2" stroke-linecap="round"/></svg>`,
  // Shield / guardrail icon
  `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6l7-3z" stroke-linejoin="round"/></svg>`,
];

const SHORT_LABELS = {
  sarah_multi_tool: "Salaire minimum à 21 ans (1ère année) et jours de congés examens",
  leo_trial_period: "Rupture pendant les 45 premiers jours de formation en entreprise",
  ines_historical_gap: "Que prévoyait l'article D6222-33 au 15 juin 2019 ? (vide juridique)",
  idcc_out_of_scope: "Salaire minimum conventionnel Syntec (hors Code du travail)",
};

document.addEventListener('DOMContentLoaded', async () => {
  loadSessionHistory();
  updateNewChatButtonState();
  await fetchSystemStatus();
  document.addEventListener('click', (e) => {
    const apiAnchor = document.querySelector('.api-popover-anchor');
    const apiPop = document.getElementById('api-popover');
    if (apiPop && !apiPop.classList.contains('hidden') && apiAnchor && !apiAnchor.contains(e.target)) {
      apiPop.classList.add('hidden');
    }
    const histAnchor = document.querySelector('.history-popover-anchor');
    const histPop = document.getElementById('history-popover');
    if (histPop && !histPop.classList.contains('hidden') && histAnchor && !histAnchor.contains(e.target)) {
      histPop.classList.add('hidden');
    }
  });
});

async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    applyOllamaStatus(data.ollama || {});
    if (Array.isArray(data.scenarios)) {
      state.scenarios = data.scenarios;
      renderSuggestions();
    }
  } catch (err) {
    const dot = document.getElementById('ollama-dot');
    if (dot) dot.classList.add('offline');
  }
}

function getModelProvider(modelId) {
  if (!modelId) return 'local';
  const entry = state.apiCatalog && state.apiCatalog[modelId];
  if (entry && typeof entry === 'object' && entry.provider) {
    return entry.provider;
  }
  if (modelId.startsWith('claude-') || modelId.startsWith('anthropic/')) {
    return 'anthropic';
  }
  if (modelId.includes('/')) {
    return 'openrouter';
  }
  return 'local';
}

function isProviderConfigured(provider) {
  return Array.isArray(state.configuredProviders) && state.configuredProviders.includes(provider);
}

function formatModelLabel(modelId) {
  if (!modelId) return '';
  const entry = state.apiCatalog && state.apiCatalog[modelId];
  if (entry && typeof entry === 'object' && entry.label) {
    return entry.label;
  }
  const base = modelId.includes('/') ? modelId.split('/').pop() : modelId.split('-it')[0];
  return base.replace(/-instruct$/i, '').replace(/-latest$/i, '').replace(/-\d{8}$/, '');
}

function fitSelectWidth(sel) {
  if (!sel || !sel.options || sel.selectedIndex < 0) return;
  const text = sel.options[sel.selectedIndex].text || '';
  const canvas = fitSelectWidth._canvas || (fitSelectWidth._canvas = document.createElement('canvas'));
  const ctx = canvas.getContext('2d');
  ctx.font = "500 12px 'JetBrains Mono', monospace";
  const textWidth = Math.ceil(ctx.measureText(text).width);
  sel.style.width = `${textWidth + 36}px`;
}

function applyOllamaStatus(ollama) {
  const dot = document.getElementById('ollama-dot');
  const statusText = document.getElementById('ollama-status-text');
  const versionTag = document.getElementById('model-version-tag');

  state.localModels = Array.isArray(ollama.local_models) ? ollama.local_models : (ollama.available_models || []);
  state.apiModels = Array.isArray(ollama.api_models) ? ollama.api_models : [];
  state.apiCatalog = ollama.api_catalog || {};
  state.hasApiKey = Boolean(ollama.has_api_key);
  state.configuredProviders = Array.isArray(ollama.configured_providers) ? ollama.configured_providers : [];
  state.selectedModel = ollama.model || state.localModels[0] || 'gemma4:e4b-it-qat';

  renderModelSelectors();

  const provider = getModelProvider(state.selectedModel);
  const shortModel = formatModelLabel(state.selectedModel);
  if (versionTag) versionTag.textContent = shortModel;

  // Automatically sync popover provider with the currently selected model's provider when cloud model is chosen
  if (provider === 'anthropic' || provider === 'openrouter') {
    state.apiPopoverProvider = provider;
  }

  const modelReady =
    provider === 'local' ? Boolean(ollama.online) : isProviderConfigured(provider);

  if (dot) dot.classList.toggle('offline', !modelReady);
  if (statusText) {
    if (provider === 'anthropic') {
      statusText.textContent = isProviderConfigured('anthropic')
        ? `${shortModel} · Anthropic`
        : `${shortModel} · Clé Anthropic requise`;
    } else if (provider === 'openrouter') {
      statusText.textContent = isProviderConfigured('openrouter')
        ? `${shortModel} · OpenRouter`
        : `${shortModel} · Clé OpenRouter requise`;
    } else {
      statusText.textContent = ollama.online ? `${shortModel} · Local` : 'Ollama offline';
    }
  }

  renderApiPopover();
}

function selectApiPopoverProvider(provider) {
  if (provider !== 'openrouter' && provider !== 'anthropic') return;
  state.apiPopoverProvider = provider;
  renderApiPopover();
  const input = document.getElementById('api-key-input');
  if (input) input.focus();
}

function renderApiPopover() {
  const provider = state.apiPopoverProvider === 'anthropic' ? 'anthropic' : 'openrouter';
  const isAnthropic = provider === 'anthropic';
  const configured = isProviderConfigured(provider);
  const modelProvider = getModelProvider(state.selectedModel);
  const shortModel = formatModelLabel(state.selectedModel);

  const tabOr = document.getElementById('api-tab-openrouter');
  const tabAnt = document.getElementById('api-tab-anthropic');
  const dotOr = document.getElementById('provider-dot-openrouter');
  const dotAnt = document.getElementById('provider-dot-anthropic');
  const titleEl = document.getElementById('api-popover-title');
  const keyBadge = document.getElementById('api-key-badge');
  const descEl = document.getElementById('api-popover-desc');
  const inputEl = document.getElementById('api-key-input');

  if (tabOr) tabOr.classList.toggle('active', !isAnthropic);
  if (tabAnt) tabAnt.classList.toggle('active', isAnthropic);
  if (dotOr) dotOr.classList.toggle('active', isProviderConfigured('openrouter'));
  if (dotAnt) dotAnt.classList.toggle('active', isProviderConfigured('anthropic'));

  if (titleEl) {
    titleEl.textContent = isAnthropic ? 'ANTHROPIC API · .ENV' : 'OPENROUTER API · .ENV';
  }
  if (keyBadge) {
    keyBadge.textContent = configured ? '● Clé active (.env)' : 'Non configurée';
    keyBadge.classList.toggle('active', configured);
  }
  if (descEl) {
    const envVar = isAnthropic ? 'ANTHROPIC_API_KEY' : 'OPENROUTER_API_KEY';
    const modelNote =
      modelProvider === provider
        ? ` pour <span class="mono" style="font-weight:600;color:var(--text-main)">${escapeHtml(shortModel)}</span>`
        : '';
    descEl.innerHTML = `Collez votre clé <span class="mono">${envVar}</span>${modelNote} dans <span class="mono">.env</span> ou directement ici :`;
  }
  if (inputEl) {
    inputEl.placeholder = isAnthropic ? 'sk-ant-api03-...' : 'sk-or-v1-...';
  }
}

function renderModelSelectors() {
  const selects = [
    document.getElementById('home-model-select'),
    document.getElementById('chat-model-select'),
  ].filter(Boolean);

  const localList = state.localModels.length > 0 ? state.localModels : ['gemma4:e4b-it-qat'];
  const localOpts = localList
    .map((m) => {
      const sel = m === state.selectedModel ? 'selected' : '';
      return `<option value="${escapeHtml(m)}" ${sel}>${escapeHtml(formatModelLabel(m))}</option>`;
    })
    .join('');

  const openrouterModels = state.apiModels.filter((m) => getModelProvider(m) === 'openrouter');
  const anthropicModels = state.apiModels.filter((m) => getModelProvider(m) === 'anthropic');

  const openrouterOpts = openrouterModels
    .map((m) => {
      const sel = m === state.selectedModel ? 'selected' : '';
      return `<option value="${escapeHtml(m)}" ${sel}>${escapeHtml(formatModelLabel(m))}</option>`;
    })
    .join('');

  const anthropicOpts = anthropicModels
    .map((m) => {
      const sel = m === state.selectedModel ? 'selected' : '';
      return `<option value="${escapeHtml(m)}" ${sel}>${escapeHtml(formatModelLabel(m))}</option>`;
    })
    .join('');

  const html = `
    <optgroup label="Local · Ollama">${localOpts}</optgroup>
    <optgroup label="OpenRouter API">${openrouterOpts}</optgroup>
    <optgroup label="Anthropic API">${anthropicOpts}</optgroup>
  `;

  selects.forEach((sel) => {
    sel.innerHTML = html;
    sel.value = state.selectedModel;
    fitSelectWidth(sel);
  });
}

function triggerMorphAnimation(el, className = 'morphing') {
  if (!el) return;
  el.classList.remove(className);
  void el.offsetWidth; // Force reflow so the keyframe restarts cleanly
  el.classList.add(className);
  setTimeout(() => {
    el.classList.remove(className);
  }, 550);
}

async function selectOllamaModel(modelName) {
  if (!modelName) return;
  state.selectedModel = modelName;

  ['home-model-select', 'chat-model-select'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      if (el.value !== modelName) el.value = modelName;
      fitSelectWidth(el);
      triggerMorphAnimation(el, 'switching');
    }
  });

  try {
    const res = await fetch('/api/model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelName }),
    });
    const ollama = await res.json();
    applyOllamaStatus(ollama);
    triggerMorphAnimation(document.getElementById('model-version-tag'), 'morphing');
    triggerMorphAnimation(document.getElementById('ollama-status-text'), 'morphing');
    // If user selected a cloud model (OpenRouter or Anthropic) and hasn't put that provider's key in .env yet, open the popover on the right provider
    const provider = getModelProvider(modelName);
    if ((provider === 'openrouter' || provider === 'anthropic') && !isProviderConfigured(provider)) {
      state.apiPopoverProvider = provider;
      renderApiPopover();
      const pop = document.getElementById('api-popover');
      if (pop) pop.classList.remove('hidden');
    }
  } catch (err) {
    console.error(err);
  }
}

async function toggleApiPopover() {
  const pop = document.getElementById('api-popover');
  if (!pop) return;
  const willShow = pop.classList.contains('hidden');
  if (willShow) {
    // Re-check .env on disk first in case the user just saved their .env file!
    await fetchSystemStatus();
    const modelProvider = getModelProvider(state.selectedModel);
    if (modelProvider === 'anthropic' || modelProvider === 'openrouter') {
      state.apiPopoverProvider = modelProvider;
    }
    renderApiPopover();
    pop.classList.remove('hidden');
  } else {
    pop.classList.add('hidden');
  }
}

async function saveApiKeyFromUi() {
  const input = document.getElementById('api-key-input');
  const key = (input ? input.value : '').trim();
  if (!key) return;
  const provider = state.apiPopoverProvider === 'anthropic' ? 'anthropic' : 'openrouter';
  try {
    const res = await fetch('/api/apikey', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key, provider }),
    });
    const ollama = await res.json();
    if (input) input.value = '';
    applyOllamaStatus(ollama);
    const pop = document.getElementById('api-popover');
    if (pop) pop.classList.add('hidden');
  } catch (err) {
    console.error(err);
  }
}

function renderSuggestions() {
  const list = document.getElementById('suggestions-list');
  if (!list) return;
  const targetIds = ['sarah_multi_tool', 'leo_trial_period', 'ines_historical_gap', 'idcc_out_of_scope'];
  const items = targetIds
    .map((id) => state.scenarios.find((s) => s.id === id))
    .filter(Boolean);

  list.innerHTML = items
    .map((sc, i) => {
      const icon = SUGGESTION_ICONS[i % SUGGESTION_ICONS.length];
      const label = SHORT_LABELS[sc.id] || sc.title;
      return `
        <button class="suggestion-row" onclick="selectScenario('${escapeHtml(sc.id)}')">
          <span class="suggestion-icon">${icon}</span>
          <span>${escapeHtml(label)}</span>
        </button>
      `;
    })
    .join('');
}

// ============================================================================
// Controls & Session Chat History (Only Completed Conversations are Stored)
// ============================================================================
function isCurrentlyOnEmptyChat() {
  const homeStage = document.getElementById('home-stage');
  return state.activeChatId === null && homeStage && !homeStage.classList.contains('hidden');
}

function updateNewChatButtonState() {
  const btn = document.getElementById('new-chat-btn');
  if (!btn) return;
  btn.classList.toggle('is-empty-chat', isCurrentlyOnEmptyChat());
}

function handleNewChatClick() {
  const histPop = document.getElementById('history-popover');
  if (histPop) histPop.classList.add('hidden');

  // Anti-spam rule: if we are already on an empty chat (no conversation done yet),
  // clicking + does NOT create any entry in history and simply focuses the prompt input.
  if (isCurrentlyOnEmptyChat()) {
    const homeInput = document.getElementById('home-input');
    if (homeInput) homeInput.focus();
    return;
  }
  resetToHome();
}

function resetToHome() {
  state.activeScenarioId = null;
  state.activeChatId = null;
  state.currentTrace = null;
  state.activeTurns = [];
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
  const turnsList = document.getElementById('chat-turns-list');
  if (turnsList) turnsList.innerHTML = '';
  const homeStage = document.getElementById('home-stage');
  if (homeStage) {
    homeStage.classList.remove('is-launching', 'hidden');
  }
  document.body.classList.remove('is-launching', 'is-thinking');
  document.getElementById('chat-view').classList.add('hidden');
  closeSourcePanel();
  const homeInput = document.getElementById('home-input');
  if (homeInput) homeInput.value = '';
  const chatInput = document.getElementById('chat-input');
  if (chatInput) chatInput.value = '';
  updateNewChatButtonState();
  renderHistoryList();
}

function loadSessionHistory() {
  try {
    const raw = sessionStorage.getItem(SESSION_HISTORY_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        // Filter strictly to valid conversations that actually have a question & answer
        state.chatHistory = parsed
          .filter((item) => item && item.question && item.trace && item.trace.answer)
          .map((item) => ({
            ...item,
            turns: Array.isArray(item.turns) && item.turns.length > 0 ? item.turns : [item.trace],
          }));
      }
    }
  } catch (e) {
    state.chatHistory = [];
  }
  renderHistoryList();
}

function saveSessionHistory() {
  try {
    sessionStorage.setItem(SESSION_HISTORY_KEY, JSON.stringify(state.chatHistory));
  } catch (e) {
    console.warn('Could not persist session chat history:', e);
  }
  renderHistoryList();
}

function recordCompletedConversation(traceData, { isFollowUp = false } = {}) {
  if (!traceData || !traceData.question || !traceData.answer) return;

  const now = new Date();
  const timeStr = now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

  // If this is a follow-up in an existing active conversation, append the turn to the SAME conversation!
  if (isFollowUp && state.activeChatId) {
    const existingIdx = state.chatHistory.findIndex((c) => c.id === state.activeChatId);
    if (existingIdx !== -1) {
      const existing = state.chatHistory[existingIdx];
      const prevTurns =
        Array.isArray(existing.turns) && existing.turns.length > 0
          ? existing.turns
          : existing.trace
          ? [existing.trace]
          : [];
      const updated = {
        ...existing,
        mode: traceData.mode || state.mode,
        model: state.selectedModel || existing.model || 'gemma4:e4b-it-qat',
        time: timeStr,
        trace: traceData,
        turns: [...prevTurns, traceData],
      };
      state.chatHistory.splice(existingIdx, 1);
      state.chatHistory.unshift(updated);
      updateNewChatButtonState();
      saveSessionHistory();
      return;
    }
  }

  const chatId = `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  const entry = {
    id: chatId,
    question: traceData.question,
    mode: traceData.mode || state.mode,
    model: state.selectedModel || 'gemma4:e4b-it-qat',
    time: timeStr,
    trace: traceData,
    turns: [traceData],
  };

  // Prepend most recent conversation at top of session history (cap at 25 per session)
  state.chatHistory.unshift(entry);
  if (state.chatHistory.length > 25) {
    state.chatHistory = state.chatHistory.slice(0, 25);
  }
  state.activeChatId = chatId;
  updateNewChatButtonState();
  saveSessionHistory();
}

function toggleHistoryPopover() {
  const pop = document.getElementById('history-popover');
  if (!pop) return;
  const apiPop = document.getElementById('api-popover');
  if (apiPop) apiPop.classList.add('hidden');
  pop.classList.toggle('hidden');
  if (!pop.classList.contains('hidden')) {
    renderHistoryList();
  }
}

function renderHistoryList() {
  const listEl = document.getElementById('history-list');
  const badgeEl = document.getElementById('history-count-badge');
  const clearBtn = document.getElementById('history-clear-btn');

  const count = state.chatHistory.length;
  if (badgeEl) {
    badgeEl.textContent = String(count);
    badgeEl.classList.toggle('hidden', count === 0);
  }
  if (clearBtn) {
    clearBtn.classList.toggle('hidden', count === 0);
  }
  if (!listEl) return;

  if (count === 0) {
    listEl.innerHTML = `<div class="history-empty">Aucune conversation dans cette session.<br/>Posez une question pour l'enregistrer ici.</div>`;
    return;
  }

  listEl.innerHTML = state.chatHistory
    .map((item) => {
      const isActive = item.id === state.activeChatId;
      const shortModel = formatModelLabel(item.model || '');
      const modeLabel = (item.mode || 'agent').toUpperCase();
      const turnCount = Array.isArray(item.turns) ? item.turns.length : 1;
      const turnsBadge = turnCount > 1 ? `<span>· ${turnCount} msg</span>` : '';
      return `
        <div class="history-item${isActive ? ' active' : ''}" onclick="openHistoryChat('${escapeHtml(item.id)}')">
          <div class="history-item-main">
            <div class="history-item-q">${escapeHtml(item.question)}</div>
            <div class="history-item-meta">
              <span class="history-mode-badge">${escapeHtml(modeLabel)}</span>
              <span>${escapeHtml(shortModel)}</span>
              ${turnsBadge}
              <span>· ${escapeHtml(item.time || '')}</span>
            </div>
          </div>
          <button class="history-delete-btn" onclick="deleteHistoryChat(event, '${escapeHtml(item.id)}')" title="Supprimer">×</button>
        </div>
      `;
    })
    .join('');
}

function openHistoryChat(chatId) {
  const item = state.chatHistory.find((c) => c.id === chatId);
  if (!item || !item.trace) return;

  const turns = Array.isArray(item.turns) && item.turns.length > 0 ? item.turns : [item.trace];
  state.activeChatId = item.id;
  state.activeTurns = [...turns];
  state.currentTrace = turns[turns.length - 1];
  setMode(item.mode || 'agent');

  const histPop = document.getElementById('history-popover');
  if (histPop) histPop.classList.add('hidden');

  document.getElementById('home-stage').classList.add('hidden');
  document.getElementById('chat-view').classList.remove('hidden');

  const turnsList = document.getElementById('chat-turns-list');
  if (turnsList) {
    turnsList.innerHTML = '';
    turns.forEach((turnData, idx) => {
      appendTurnElement(idx, turnData.question || '');
      renderConversationTurn(turnData, idx, { autoOpenPanel: idx === turns.length - 1 });
    });
  }

  updateNewChatButtonState();
  renderHistoryList();
}

function deleteHistoryChat(event, chatId) {
  if (event) event.stopPropagation();
  state.chatHistory = state.chatHistory.filter((c) => c.id !== chatId);
  if (state.activeChatId === chatId) {
    resetToHome();
  }
  saveSessionHistory();
}

function clearSessionHistory() {
  state.chatHistory = [];
  saveSessionHistory();
}

function setMode(mode) {
  const changed = state.mode !== mode;
  state.mode = mode;
  ['seg-agent', 'chat-seg-agent'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', mode === 'agent');
  });
  ['seg-rag', 'chat-seg-rag'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', mode === 'rag');
  });
  // Slide the white pill glider smoothly between Agent and RAG
  document.querySelectorAll('.segmented-pill').forEach((pill) => {
    pill.classList.toggle('mode-rag', mode === 'rag');
  });

  if (changed) {
    document.querySelectorAll('.prompt-card').forEach((card) => {
      card.classList.add('mode-ripple');
      clearTimeout(card._rippleTimer);
      card._rippleTimer = setTimeout(() => {
        card.classList.remove('mode-ripple');
      }, 280);
    });
  }
}

function toggleTraceSteps(turnIdx = 0) {
  const drawer = document.getElementById(`trace-accordion-${turnIdx}`);
  const hint = document.getElementById(`trace-toggle-hint-${turnIdx}`);
  if (!drawer) return;
  const isCollapsed = drawer.classList.toggle('collapsed');
  if (hint) {
    hint.textContent = isCollapsed ? 'Details' : 'Hide';
  }
}

function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function appendTurnElement(turnIdx, questionText) {
  const turnsList = document.getElementById('chat-turns-list');
  if (!turnsList) return null;

  const existing = document.getElementById(`chat-turn-${turnIdx}`);
  if (existing) existing.remove();

  const wrapper = document.createElement('div');
  wrapper.className = 'chat-turn';
  wrapper.id = `chat-turn-${turnIdx}`;
  wrapper.dataset.turnIndex = String(turnIdx);
  wrapper.innerHTML = `
    <div class="user-row">
      <div class="user-pill" id="user-question-text-${turnIdx}">${escapeHtml(questionText)}</div>
    </div>

    <div class="assistant-card">
      <div class="trace-drawer collapsed" id="trace-accordion-${turnIdx}">
        <button class="trace-pill-trigger" onclick="toggleTraceSteps(${turnIdx})" title="Click to inspect tool calls and arguments">
          <div class="trace-trigger-left">
            <span class="trace-dot" id="trace-dot-${turnIdx}"></span>
            <span class="trace-label" id="trace-summary-text-${turnIdx}">Reasoning &amp; tools</span>
            <div class="trace-chips" id="trace-chips-${turnIdx}"></div>
          </div>
          <div class="trace-trigger-right">
            <span class="trace-hint" id="trace-toggle-hint-${turnIdx}">Details</span>
            <svg class="trace-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M6 9l6 6 6-6" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </button>

        <div class="trace-details-panel" id="trace-steps-body-${turnIdx}"></div>
      </div>

      <div class="answer-body" id="answer-body-${turnIdx}"></div>

      <div class="answer-footer" id="answer-footer-${turnIdx}">
        <div class="source-pills-row" id="source-pills-row-${turnIdx}"></div>
        <div class="meta-line" id="meta-line-${turnIdx}"></div>
      </div>
    </div>
  `;
  turnsList.appendChild(wrapper);
  return wrapper;
}

// ============================================================================
// 100% Live Query Execution on Ollama / OpenRouter / Anthropic (Multi-Turn)
// ============================================================================
async function selectScenario(scenarioId) {
  if (document.body.classList.contains('is-thinking')) return;
  const sc = state.scenarios.find((s) => s.id === scenarioId);
  if (!sc) return;
  state.activeScenarioId = scenarioId;
  setMode(sc.mode || 'agent');
  // Starting from Home starts a fresh conversation thread
  await executeQuery({ question: sc.question, mode: state.mode, isFollowUp: false });
}

async function submitFromHome() {
  if (document.body.classList.contains('is-thinking')) return;
  const input = document.getElementById('home-input');
  const q = (input.value || '').trim();
  if (!q) return;
  input.value = '';
  state.activeScenarioId = null;
  await executeQuery({ question: q, mode: state.mode, isFollowUp: false });
}

async function submitFromChat() {
  if (document.body.classList.contains('is-thinking')) return;
  const input = document.getElementById('chat-input');
  const q = (input.value || '').trim();
  if (!q) return;
  input.value = '';
  input.style.height = 'auto';
  state.activeScenarioId = null;
  // Submitting from #chat-input continues the current conversation as a true follow-up turn!
  await executeQuery({ question: q, mode: state.mode, isFollowUp: true });
}

async function executeQuery(payload) {
  const qText = payload.question || '';
  const isFollowUp = Boolean(payload.isFollowUp && state.activeTurns.length > 0);
  const homeStage = document.getElementById('home-stage');
  const chatView = document.getElementById('chat-view');
  const turnsList = document.getElementById('chat-turns-list');

  if (!isFollowUp) {
    state.activeTurns = [];
    state.activeChatId = null;
    if (turnsList) turnsList.innerHTML = '';
    closeSourcePanel();
  }

  // Trigger the home title & blue aura launch animation if starting from the home screen
  if (!isFollowUp && homeStage && !homeStage.classList.contains('hidden')) {
    homeStage.classList.add('is-launching');
    document.body.classList.add('is-launching');
    setTimeout(() => {
      document.body.classList.remove('is-launching');
    }, 1150);
    await new Promise((resolve) => setTimeout(resolve, 680));
    homeStage.classList.add('hidden');
    homeStage.classList.remove('is-launching');
  }
  if (chatView) chatView.classList.remove('hidden');

  const turnIdx = state.activeTurns.length;
  const turnWrapper = appendTurnElement(turnIdx, qText);
  if (isFollowUp && turnWrapper) {
    setTimeout(() => {
      turnWrapper.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 40);
  }

  const drawer = document.getElementById(`trace-accordion-${turnIdx}`);
  const dot = document.getElementById(`trace-dot-${turnIdx}`);
  const summaryText = document.getElementById(`trace-summary-text-${turnIdx}`);
  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  const stepsEl = document.getElementById(`trace-steps-body-${turnIdx}`);
  const hintEl = document.getElementById(`trace-toggle-hint-${turnIdx}`);
  const answerEl = document.getElementById(`answer-body-${turnIdx}`);
  const sourcesEl = document.getElementById(`source-pills-row-${turnIdx}`);
  const metaEl = document.getElementById(`meta-line-${turnIdx}`);

  // Always start discreet (collapsed)
  if (drawer) drawer.classList.add('collapsed');
  if (hintEl) hintEl.textContent = 'Details';

  dot.classList.add('running');
  document.body.classList.add('is-thinking');
  const startTs = performance.now();
  const modelLabel = formatModelLabel(state.selectedModel || 'gemma4:12b');

  // Build conversation history payload for follow-up context resolution on the backend
  const historyPayload = isFollowUp
    ? state.activeTurns.map((t) => ({
        question: t.question,
        answer: t.answer,
        citations: t.citations || [],
        cited_nums: (t.sources || []).filter((s) => s.is_cited).map((s) => s.num),
        citable: Object.fromEntries((t.sources || []).map((s) => [s.num, s.doc_id])),
        tool_calls: (t.tool_calls || []).map((tc) => ({
          name: tc.name,
          arguments: tc.arguments,
        })),
      }))
    : [];

  // Live state tracked during streaming
  const liveTimeline = [];
  const completedToolChips = [];
  let activeToolName = null;
  let liveToolStepCount = 0;
  const timerChipId = `live-timer-chip-${turnIdx}`;

  function renderLiveProgress() {
    const elapsed = ((performance.now() - startTs) / 1000).toFixed(1);
    const activeChipHtml = activeToolName
      ? `<span class="trace-chip active-tool">⚙ ${escapeHtml(activeToolName)}…</span>`
      : '';
    const doneChipsHtml = completedToolChips
      .map((t) => `<span class="trace-chip verified">${escapeHtml(t)} ✓</span>`)
      .join('');
    chipsEl.innerHTML = `${doneChipsHtml}${activeChipHtml}<span class="trace-chip" id="${timerChipId}">${elapsed}s</span>`;

    answerEl.innerHTML = `
      <div class="live-activity-box">
        ${liveTimeline
          .map((item) => {
            const stateClass = item.status === 'running' ? 'running' : 'done';
            const iconHtml =
              item.status === 'running'
                ? `<span class="live-step-spinner"></span>`
                : `<span class="live-step-check">✓</span>`;
            const badgeHtml = item.badge
              ? `<span class="live-step-badge mono">${escapeHtml(item.badge)}</span>`
              : '';
            return `
              <div class="live-activity-row ${stateClass}">
                ${iconHtml}
                <div class="live-activity-text">
                  <div class="live-activity-title">
                    <span>${escapeHtml(item.title)}</span>
                    ${badgeHtml}
                  </div>
                  ${item.detail ? `<div class="live-activity-detail">${escapeHtml(item.detail)}</div>` : ''}
                </div>
              </div>
            `;
          })
          .join('')}
      </div>
    `;
  }

  summaryText.textContent = `Initialisation (${modelLabel})…`;
  liveTimeline.push({
    id: 'init',
    status: 'running',
    title: `Démarrage (${modelLabel})`,
    detail: isFollowUp
      ? `Reprise du contexte (${state.activeTurns.length} échange${state.activeTurns.length > 1 ? 's' : ''} précédent${state.activeTurns.length > 1 ? 's' : ''})…`
      : 'Préparation du contexte juridique…',
  });
  stepsEl.innerHTML = '';
  sourcesEl.innerHTML = '';
  metaEl.textContent = '';
  renderLiveProgress();

  if (state.timerInterval) clearInterval(state.timerInterval);
  state.timerInterval = setInterval(() => {
    const elapsed = ((performance.now() - startTs) / 1000).toFixed(1);
    const chip = document.getElementById(timerChipId);
    if (chip) chip.textContent = `${elapsed}s`;
  }, 100);

  function markAllRunningDone() {
    liveTimeline.forEach((item) => {
      if (item.status === 'running') item.status = 'done';
    });
  }

  function handleStreamEvent(ev) {
    if (!ev || !ev.type) return;

    if (ev.type === 'phase') {
      markAllRunningDone();
      activeToolName = null;
      summaryText.textContent = `${ev.title}…`;
      liveTimeline.push({
        id: `phase_${Date.now()}`,
        status: 'running',
        title: ev.title,
        detail: ev.detail || '',
      });
      renderLiveProgress();
      return;
    }

    if (ev.type === 'llm_decision') {
      const lastRunning = [...liveTimeline].reverse().find((i) => i.status === 'running');
      const plannedNames = (ev.planned_tools || []).map((t) => t.name).join(', ');
      if (lastRunning) {
        lastRunning.status = 'done';
        lastRunning.badge = `${(ev.latency_ms / 1000).toFixed(1)}s`;
        if (plannedNames) {
          lastRunning.detail = `Décision du modèle → appel de : ${plannedNames}`;
        }
      }
      renderLiveProgress();
      return;
    }

    if (ev.type === 'tool_start') {
      markAllRunningDone();
      activeToolName = ev.name;
      summaryText.textContent = `Outil en cours : ${ev.name}…`;
      liveTimeline.push({
        id: `tool_${ev.name}`,
        status: 'running',
        title: `Outil · ${ev.name}`,
        detail: ev.detail || '',
      });
      renderLiveProgress();
      return;
    }

    if (ev.type === 'tool_done') {
      activeToolName = null;
      if (!completedToolChips.includes(ev.name)) {
        completedToolChips.push(ev.name);
      }
      const runningTool = [...liveTimeline].reverse().find((i) => i.status === 'running');
      const foundNums = Array.isArray(ev.citable_nums) && ev.citable_nums.length > 0
        ? ` → Articles : ${ev.citable_nums.join(', ')}`
        : '';
      if (runningTool) {
        runningTool.status = 'done';
        runningTool.badge = `${ev.latency_ms} ms`;
        runningTool.detail = (runningTool.detail || '') + foundNums;
      }
      // Also append the tool card live into the expandable Details drawer!
      liveToolStepCount += 1;
      stepsEl.insertAdjacentHTML(
        'beforeend',
        renderToolStepCard(
          {
            step: liveToolStepCount,
            name: ev.name,
            arguments: ev.arguments,
            latency_ms: ev.latency_ms,
            result: ev.result,
          },
          liveToolStepCount
        )
      );
      renderLiveProgress();
    }
  }

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: qText,
        mode: payload.mode || state.mode,
        model: state.selectedModel,
        history: historyPayload,
        stream: true,
      }),
    });

    if (!res.body) {
      const data = await res.json();
      if (data.error) {
        answerEl.textContent = `Error: ${data.error}`;
        return;
      }
      state.activeTurns[turnIdx] = data;
      state.currentTrace = data;
      recordCompletedConversation(data, { isFollowUp });
      renderConversationTurn(data, turnIdx);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        let ev;
        try {
          ev = JSON.parse(line);
        } catch (e) {
          continue;
        }
        if (ev.type === 'error') {
          answerEl.textContent = `Error: ${ev.error}`;
          return;
        }
        if (ev.type === 'final' && ev.data) {
          state.activeTurns[turnIdx] = ev.data;
          state.currentTrace = ev.data;
          recordCompletedConversation(ev.data, { isFollowUp });
          renderConversationTurn(ev.data, turnIdx);
          if (isFollowUp && turnWrapper) {
            setTimeout(() => {
              turnWrapper.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }, 60);
          }
          return;
        }
        handleStreamEvent(ev);
      }
    }
  } catch (err) {
    answerEl.textContent = `Request failed: ${err}`;
  } finally {
    if (state.timerInterval) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
    dot.classList.remove('running');
    document.body.classList.remove('is-thinking');
  }
}

// ============================================================================
// Render Conversation Turn, Discreet Tool Trace & Right Source Panel
// ============================================================================
function renderConversationTurn(data, turnIdx = 0, { autoOpenPanel = true } = {}) {
  const qEl = document.getElementById(`user-question-text-${turnIdx}`);
  if (qEl) qEl.textContent = data.question;

  const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];
  const extCalls = toolCalls.filter((tc) => tc.name && tc.name !== 'final_answer');

  // 1. Discreet Summary Pill
  const summaryEl = document.getElementById(`trace-summary-text-${turnIdx}`);
  if (summaryEl) {
    if (data.mode === 'rag' && extCalls.length === 0) {
      summaryEl.textContent = 'Hybrid RAG retrieval';
    } else if (extCalls.length > 0) {
      summaryEl.textContent = `${extCalls.length} tool${extCalls.length > 1 ? 's' : ''} called`;
    } else {
      summaryEl.textContent = 'Scope guardrail';
    }
  }

  const uniqueTools = [...new Set(extCalls.map((t) => t.name))];
  const chipsHtml = uniqueTools
    .map((t) => `<span class="trace-chip">${escapeHtml(t)}</span>`)
    .join('');
  const chipsEl = document.getElementById(`trace-chips-${turnIdx}`);
  if (chipsEl) {
    chipsEl.innerHTML = chipsHtml + `<span class="trace-chip verified">check_final ✓</span>`;
  }

  // 2. Detailed Tool Step Cards (Visible when user clicks the trace pill during demo)
  const stepsEl = document.getElementById(`trace-steps-body-${turnIdx}`);
  if (stepsEl) {
    if (toolCalls.length > 0) {
      stepsEl.innerHTML = toolCalls.map((tc, idx) => renderToolStepCard(tc, idx + 1)).join('');
    } else {
      // Single-pass RAG mode trace card
      const retrievedCount = (data.retrieved_ids || []).length;
      stepsEl.innerHTML = `
        <div class="trace-step-card">
          <div class="trace-step-top">
            <div class="trace-step-title">
              <span class="trace-step-num">RAG</span>
              <span class="trace-step-fn">search(BM25 + bge-m3 + RRF + bge-reranker-v2-m3)</span>
            </div>
            <span class="trace-step-latency">${retrievedCount} chunks</span>
          </div>
          <div class="trace-step-obs">Retrieved ${retrievedCount} VIGUEUR articles → Constrained RagAnswer JSON → check_final() verified</div>
        </div>
      `;
    }
  }

  // 3. Answer body with clickable citation pills
  const answerEl = document.getElementById(`answer-body-${turnIdx}`);
  if (answerEl) {
    answerEl.innerHTML = decorateCitations(data.answer || '', turnIdx);
  }

  // 4. Source pills
  const sources = Array.isArray(data.sources) ? data.sources : [];
  const cited = sources.filter((s) => s.is_cited);
  const displaySources = cited.length > 0 ? cited : sources;
  const pillsRow = document.getElementById(`source-pills-row-${turnIdx}`);

  if (pillsRow) {
    if (displaySources.length > 0) {
      pillsRow.innerHTML = displaySources
        .map(
          (s) => `
          <button class="src-pill" data-doc-id="${escapeHtml(s.doc_id)}" onclick="openSourcePanel('${escapeHtml(s.doc_id)}', ${turnIdx})">
            <span class="src-dot"></span>
            <span>Art. ${escapeHtml(s.num)}</span>
          </button>
        `
        )
        .join('');
    } else {
      const histCall = extCalls.find((tc) => tc.name === 'get_article_at_date');
      if (histCall && histCall.arguments && histCall.arguments.num) {
        const num = histCall.arguments.num;
        const dateStr = histCall.arguments.date || '2019-06-15';
        pillsRow.innerHTML = `
          <button class="src-pill" onclick="openGapPanel('${escapeHtml(num)}', '${escapeHtml(dateStr)}')">
            <span class="src-dot" style="background:var(--red)"></span>
            <span>Timeline ${escapeHtml(num)} (${escapeHtml(dateStr)})</span>
          </button>
        `;
      } else {
        pillsRow.innerHTML = '';
      }
    }
  }

  // 5. Minimalist telemetry line
  const cCount = (data.citations || []).length;
  const secs = ((data.latency_ms || 0) / 1000).toFixed(2);
  const tokIn = data.tokens_in || 0;
  const tokOut = data.tokens_out || 0;
  const rawCost = Number(data.cost_eur || 0);
  const costSuffix = rawCost > 0 ? ` · €${rawCost.toFixed(4)}` : '';

  const metaEl = document.getElementById(`meta-line-${turnIdx}`);
  if (metaEl) {
    metaEl.textContent =
      cCount > 0
        ? `${cCount}/${cCount} verified · ${secs}s · ${tokIn + tokOut} tok${costSuffix}`
        : `stop='${data.stop_reason || 'answered'}' · ${secs}s${costSuffix}`;
  }

  // 6. Auto-open right panel for visual impact on the latest turn
  if (autoOpenPanel) {
    if (displaySources.length > 0) {
      openSourcePanel(displaySources[0].doc_id, turnIdx);
    } else {
      const histCall = extCalls.find((tc) => tc.name === 'get_article_at_date');
      if (histCall && histCall.arguments && histCall.arguments.num) {
        openGapPanel(histCall.arguments.num, histCall.arguments.date || '2019-06-15');
      } else if (turnIdx === 0) {
        closeSourcePanel();
      }
    }
  }
}

function renderToolStepCard(tc, index) {
  const isFinal = tc.name === 'final_answer';
  const fnName = tc.name || 'schema_recovery';
  const stepBadgeClass = isFinal ? 'trace-step-num final' : 'trace-step-num';
  const stepLabel = isFinal ? 'GATE' : `STEP ${tc.step || index}`;
  const latencyBadge =
    tc.latency_ms != null
      ? `<span class="trace-step-latency">${tc.latency_ms} ms</span>`
      : isFinal
      ? `<span class="trace-step-latency">${tc.error ? 'check_final ↻' : 'check_final ✓'}</span>`
      : '';

  // Render arguments as clean key=value pills (skip repeating full answer prose inside final_answer args)
  let argsHtml = '';
  if (tc.arguments && typeof tc.arguments === 'object') {
    const entries = Object.entries(tc.arguments).filter(([k]) => k !== 'answer');
    if (entries.length > 0) {
      argsHtml = `<div class="trace-args-row">${entries
        .map(
          ([k, v]) =>
            `<span class="trace-arg-pill"><span class="trace-arg-key">${escapeHtml(k)}:</span> ${escapeHtml(
              JSON.stringify(v)
            )}</span>`
        )
        .join('')}</div>`;
    }
  }

  // Clean up observation preview
  let obsText = String(tc.result || '').trim();
  if (obsText) {
    // Make XML <source article="L6222-35"> tags human-friendly in the preview
    obsText = obsText
      .replace(/<source\s+article="([^"]+)"[^>]*>/g, '[Art. $1] ')
      .replace(/<\/source>/g, '');
  }

  const obsHtml = obsText ? `<div class="trace-step-obs">${escapeHtml(obsText)}</div>` : '';

  return `
    <div class="trace-step-card">
      <div class="trace-step-top">
        <div class="trace-step-title">
          <span class="${stepBadgeClass}">${escapeHtml(stepLabel)}</span>
          <span class="trace-step-fn">${escapeHtml(fnName)}</span>
        </div>
        ${latencyBadge}
      </div>
      ${argsHtml}
      ${obsHtml}
    </div>
  `;
}

function decorateCitations(text, turnIdx = 0) {
  const raw = String(text || '').trim();
  if (!raw) return '';
  // Split into sentences for a smooth staggered reveal
  const sentences = raw.split(/(?<=[.!?])\s+/);
  return sentences
    .map((sentence, idx) => {
      const escaped = escapeHtml(sentence);
      const withCites = escaped.replace(/\[([LRD]\.?\s*\d[\w-]*)\]/gi, (match, rawNum) => {
        const cleanNum = rawNum.replace(/\s+/g, '').replace(/\./g, '').toUpperCase();
        return `<button class="cite-btn" style="animation-delay:${idx * 70 + 120}ms" onclick="openSourceByNum('${cleanNum}', ${turnIdx})" onmouseenter="highlightSourceByNum('${cleanNum}', true, ${turnIdx})" onmouseleave="highlightSourceByNum('${cleanNum}', false, ${turnIdx})">${cleanNum}</button>`;
      });
      return `<span class="sentence-chunk" style="animation-delay:${idx * 70}ms">${withCites}</span>`;
    })
    .join(' ');
}

function _resolveSourcesForTurn(turnIdx) {
  const turnTrace =
    turnIdx != null && state.activeTurns[turnIdx] ? state.activeTurns[turnIdx] : state.currentTrace;
  const turnSources = (turnTrace && Array.isArray(turnTrace.sources) && turnTrace.sources) || [];
  // Merge any sources from earlier turns as fallback in case a follow-up cites an earlier turn's article
  const seenIds = new Set(turnSources.map((s) => s.doc_id));
  const extraSources = [];
  for (const t of state.activeTurns) {
    for (const s of (t && t.sources) || []) {
      if (s && s.doc_id && !seenIds.has(s.doc_id)) {
        seenIds.add(s.doc_id);
        extraSources.push(s);
      }
    }
  }
  return {
    turnTrace: turnTrace || state.currentTrace,
    sources: [...turnSources, ...extraSources],
  };
}

function highlightSourceByNum(num, active, turnIdx = null) {
  const { sources } = _resolveSourcesForTurn(turnIdx);
  const match = sources.find((s) => s.num.toUpperCase() === num.toUpperCase());
  if (!match) return;
  document.querySelectorAll('.src-pill').forEach((p) => {
    if (p.dataset.docId === match.doc_id) {
      p.classList.toggle('pulse-highlight', active);
    }
  });
  const panel = document.getElementById('source-panel');
  if (panel && !panel.classList.contains('hidden')) {
    panel.classList.toggle('pulse-highlight', active);
  }
}

function openSourceByNum(num, turnIdx = null) {
  const { sources } = _resolveSourcesForTurn(turnIdx);
  const match = sources.find((s) => s.num.toUpperCase() === num.toUpperCase());
  if (match) openSourcePanel(match.doc_id, turnIdx);
}

// ============================================================================
// Right-Side Légifrance Source Inspector Card (Structured Legal Formatting)
// ============================================================================
async function openSourcePanel(docId, turnIdx = null) {
  const { turnTrace, sources } = _resolveSourcesForTurn(turnIdx);
  if (turnTrace) state.currentTrace = turnTrace;
  const cited = sources.filter((s) => s.is_cited);
  const tabList = cited.length > 0 ? cited : sources;
  const target = sources.find((s) => s.doc_id === docId) || tabList[0];
  if (!target) return;

  document.querySelectorAll('.src-pill').forEach((p) => {
    p.classList.toggle('active', p.dataset.docId === target.doc_id);
  });

  const panel = document.getElementById('source-panel');
  panel.classList.remove('hidden');
  document.getElementById('panel-title').textContent = `Article ${target.num}`;

  const turnArg = turnIdx != null ? `, ${turnIdx}` : '';
  const tabsEl = document.getElementById('panel-tabs');
  tabsEl.innerHTML =
    tabList.length > 1
      ? tabList
          .map(
            (s) => `
          <button class="panel-tab ${s.doc_id === target.doc_id ? 'active' : ''}" onclick="openSourcePanel('${escapeHtml(s.doc_id)}'${turnArg})">
            Art. ${escapeHtml(s.num)}
          </button>
        `
          )
          .join('')
      : '';

  // Keep the SMIC table widget at the top when viewing D6222-26
  const widgetEl = document.getElementById('panel-widget');
  if (target.num === 'D6222-26') {
    const remCall = ((state.currentTrace && state.currentTrace.tool_calls) || []).find(
      (tc) => tc.name === 'compute_remuneration'
    );
    const age = (remCall && remCall.arguments && remCall.arguments.age) || 21;
    const yr = (remCall && remCall.arguments && remCall.arguments.contract_year) || 1;
    await renderSmicWidget(widgetEl, age, yr);
  } else {
    widgetEl.classList.add('hidden');
    widgetEl.innerHTML = '';
  }

  const isVigueur = (target.etat || '').toUpperCase().includes('VIGUEUR');
  const etatClass = isVigueur ? 'vigueur' : 'historical';
  const etatLabel = isVigueur ? '● EN VIGUEUR' : `● ${target.etat || 'HISTORIQUE'}`;

  // Format hierarchy breadcrumbs cleanly while keeping the grey .panel-hierarchy box
  const hierarchyHtml = target.hierarchy
    ? `<div class="panel-hierarchy">${escapeHtml(target.hierarchy).replace(/\s*&gt;\s*/g, ' › ')}</div>`
    : '';

  // Format legal text below the grey box into structured clauses & paragraphs
  const formattedBodyHtml = formatLegalArticleBody(target.text || '');

  // Optional Nota box
  const notaHtml = target.nota
    ? `
      <div class="panel-nota-card">
        <div class="panel-nota-title">Nota Légifrance</div>
        <div>${escapeHtml(target.nota)}</div>
      </div>
    `
    : '';

  document.getElementById('panel-content').innerHTML = `
    <div class="panel-meta-badges">
      <span class="etat-badge ${etatClass}">${escapeHtml(etatLabel)}</span>
      <span class="docid-chip">${escapeHtml(target.doc_id)}</span>
    </div>
    ${hierarchyHtml}
    <div class="legal-section-label">
      <span>Texte officiel consolidé</span>
      <span>Art. ${escapeHtml(target.num)}</span>
    </div>
    <div class="legal-clauses-list">
      ${formattedBodyHtml}
    </div>
    ${notaHtml}
    <div class="panel-link-row">
      <span class="panel-validity">Validité : ${escapeHtml(target.date_debut)} → ${escapeHtml(target.date_fin)}</span>
      <a class="panel-link" href="${escapeHtml(target.url)}" target="_blank" rel="noopener noreferrer" onclick="openLegifranceUrl(event, '${escapeHtml(target.url)}')">
        Légifrance ↗
      </a>
    </div>
  `;
}

async function openLegifranceUrl(event, url) {
  if (!url) return;
  if (event) event.preventDefault();
  try {
    await fetch('/api/open-external', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
  } catch (err) {
    window.open(url, '_blank', 'noopener,noreferrer');
  }
}

/**
 * Splits raw Légifrance article text into a hierarchical tree:
 * - Main numbered clauses (`1°`, `2°`, `3°`, `I.`, `II.`) become framed parent cards (`.legal-group-card`)
 * - Lettered sub-clauses (`a)`, `b)`, `c)`) are nested inside their parent `1°` / `2°` card (`.legal-subclauses-tree`)
 * - Unnumbered paragraphs become `.legal-paragraph-card` (and can also host nested `a)`, `b)` if preceded by `:`)
 */
function formatLegalArticleBody(rawText) {
  const cleaned = String(rawText || '').trim();
  if (!cleaned) return '<div class="legal-paragraph-card">—</div>';

  // Split by newlines first, then split inline numbered/lettered clauses ("1° ", "2° ", "a) ", "b) ")
  const rawLines = cleaned.split(/\n+/);
  const blocks = [];

  for (const line of rawLines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const subParts = trimmed
      .split(/(?<=[.:;])\s+(?=(?:[1-9]\d*°|[a-z]\)|[I]{1,3}\.\s*[-—])\s)/)
      .map((s) => s.trim())
      .filter(Boolean);
    blocks.push(...subParts);
  }

  // Build hierarchy: group a), b), c) inside their parent 1°, 2°, 3° (or colon-ending paragraph)
  const nodes = [];
  for (const block of blocks) {
    const mainClauseMatch = block.match(/^([1-9]\d*°|[I]{1,3}\.\s*[-—])\s*(.*)$/);
    const subClauseMatch = block.match(/^([a-z]\))\s*(.*)$/);

    if (mainClauseMatch) {
      nodes.push({
        type: 'clause',
        badge: mainClauseMatch[1].replace(/\s+/g, ''),
        body: decorateLegalInline(mainClauseMatch[2]),
        children: [],
      });
    } else if (subClauseMatch) {
      const subItem = {
        badge: subClauseMatch[1].replace(/\s+/g, ''),
        body: decorateLegalInline(subClauseMatch[2]),
      };
      const prev = nodes[nodes.length - 1];
      if (prev && (prev.type === 'clause' || prev.type === 'paragraph')) {
        prev.children.push(subItem);
      } else {
        nodes.push({
          type: 'subclause-standalone',
          badge: subItem.badge,
          body: subItem.body,
        });
      }
    } else {
      nodes.push({
        type: 'paragraph',
        body: decorateLegalInline(block),
        children: [],
      });
    }
  }

  return nodes
    .map((node) => {
      const childrenHtml =
        node.children && node.children.length > 0
          ? `<div class="legal-subclauses-tree">
              ${node.children
                .map(
                  (sub) => `
                <div class="legal-subclause-row">
                  <span class="subclause-badge">${escapeHtml(sub.badge)}</span>
                  <div class="subclause-text">${sub.body}</div>
                </div>
              `
                )
                .join('')}
            </div>`
          : '';

      if (node.type === 'clause') {
        const hasChildrenClass = node.children.length > 0 ? ' has-children' : '';
        return `
          <div class="legal-group-card${hasChildrenClass}">
            <div class="legal-group-header">
              <span class="clause-num-badge">${escapeHtml(node.badge)}</span>
              <div class="legal-group-title">${node.body}</div>
            </div>
            ${childrenHtml}
          </div>
        `;
      }

      if (node.type === 'subclause-standalone') {
        return `
          <div class="legal-subclause-row standalone">
            <span class="subclause-badge">${escapeHtml(node.badge)}</span>
            <div class="subclause-text">${node.body}</div>
          </div>
        `;
      }

      return `
        <div class="legal-paragraph-card">
          <div>${node.body}</div>
          ${childrenHtml}
        </div>
      `;
    })
    .join('');
}

/**
 * Highlights percentages, legal day/age thresholds, and Code du travail article references.
 */
function decorateLegalInline(text) {
  let html = escapeHtml(text);

  // Highlight Code du travail article references (e.g. L. 6222-18, D. 6222-26, R. 6222-21)
  html = html.replace(
    /\b([LRD]\.?\s*\d{4}-\d+(?:-\d+)?)\b/g,
    '<span class="legal-highlight-ref">$1</span>'
  );

  // Highlight percentages, days, and age thresholds (e.g. 61 %, 45 premiers jours, 5 jours ouvrables, 16 ans, 25 ans)
  html = html.replace(
    /\b(\d+\s*%|\d+\s*(?:premiers\s+)?jours(?:\s+ouvrables)?|quarante-cinq\s+premiers\s+jours|cinq\s+jours\s+ouvrables|\d+\s*ans)\b/gi,
    '<span class="legal-highlight-num">$1</span>'
  );

  return html;
}

async function renderSmicWidget(widgetEl, age, contractYear) {
  try {
    const res = await fetch('/api/tools/remuneration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ age, contract_year: contractYear }),
    });
    const d = await res.json();
    const rows = [
      ['16-17 ans', '16–17y', [27, 39, 55]],
      ['18-20 ans', '18–20y', [43, 51, 67]],
      ['21-25 ans', '21–25y', [53, 61, 78]],
      ['26 ans et plus', '26y+', [100, 100, 100]],
    ];
    widgetEl.classList.remove('hidden');
    widgetEl.innerHTML = `
      <div class="mono" style="font-size:0.68rem; font-weight:600; color:var(--purple); margin-bottom:6px;">
        D6222-26 SMIC GRID · ${d.percent_smic}% = €${Number(d.monthly_gross_eur).toFixed(2)}/mo
      </div>
      <table class="smic-mini-grid">
        <thead><tr><th style="text-align:left">Age</th><th>Y1</th><th>Y2</th><th>Y3</th></tr></thead>
        <tbody>
          ${rows
            .map(([key, label, vals]) => {
              const isBand = d.band === key;
              return `<tr><td style="text-align:left; font-weight:600;">${label}</td>${vals
                .map((v, i) => `<td class="${isBand && d.effective_year === i + 1 ? 'active-cell' : ''}">${v}%</td>`)
                .join('')}</tr>`;
            })
            .join('')}
        </tbody>
      </table>
    `;
  } catch (e) {
    widgetEl.classList.add('hidden');
  }
}

async function openGapPanel(num, dateStr) {
  const panel = document.getElementById('source-panel');
  panel.classList.remove('hidden');
  document.getElementById('panel-title').textContent = `Article ${num}`;
  document.getElementById('panel-tabs').innerHTML = '';

  const widgetEl = document.getElementById('panel-widget');
  let hierarchy = '';
  try {
    const res = await fetch('/api/tools/article-at-date', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num, date: dateStr }),
    });
    const d = await res.json();
    hierarchy = d.hierarchy || '';
    widgetEl.classList.remove('hidden');
    widgetEl.innerHTML = `
      <div class="mono" style="font-size:0.68rem; font-weight:600; color:var(--red); margin-bottom:6px;">
        TEMPORAL VERSION TIMELINE (${escapeHtml(dateStr)})
      </div>
      <div style="display:flex; flex-direction:column; gap:5px; font-size:0.72rem;" class="mono">
        ${(d.timeline || [])
          .map((item) => {
            if (item.type === 'gap') {
              return `<div style="padding:5px 8px; border-radius:6px; background:var(--red-pill); color:var(--red); font-weight:600;">${escapeHtml(item.date_debut)} → ${escapeHtml(item.date_fin)} · REPEAL GAP</div>`;
            }
            return `<div style="padding:5px 8px; border-radius:6px; background:#FFFFFF; color:var(--text-secondary);">${escapeHtml(item.date_debut)} → ${escapeHtml(item.date_fin)} · ${escapeHtml(item.etat)}</div>`;
          })
          .join('')}
      </div>
    `;
  } catch (e) {
    widgetEl.classList.add('hidden');
  }

  document.getElementById('panel-content').innerHTML = `
    <div class="panel-meta-badges">
      <span class="etat-badge gap">● VIDE JURIDIQUE (${escapeHtml(dateStr)})</span>
      <span class="docid-chip">citable = {}</span>
    </div>
    ${hierarchy ? `<div class="panel-hierarchy">${escapeHtml(hierarchy).replace(/\s*&gt;\s*/g, ' › ')}</div>` : ''}
    <div class="legal-section-label">
      <span>Analyse temporelle Légifrance</span>
      <span>Art. ${escapeHtml(num)}</span>
    </div>
    <div class="legal-clauses-list">
      <div class="legal-paragraph-card">
        L'article <span class="legal-highlight-ref">${escapeHtml(num)}</span> a été <strong>abrogé le 2019-01-01</strong> puis <strong>recréé le 2020-04-01</strong>.
      </div>
      <div class="legal-paragraph-card">
        À la date demandée (<span class="legal-highlight-num">${escapeHtml(dateStr)}</span>), aucune version n'était en vigueur. L'outil <span class="mono">get_article_at_date</span> renvoie <span class="mono">citable={}</span> afin de bloquer toute hallucination d'un texte abrogé.
      </div>
    </div>
  `;
}

function closeSourcePanel() {
  document.getElementById('source-panel').classList.add('hidden');
}

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
