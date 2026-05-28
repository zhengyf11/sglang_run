const fieldGroups = {
  model: [
    { name: 'model_path', label: 'Model path', wide: true },
    { name: 'served_model_name', label: 'Served model name' },
    { name: 'tool_call_parser', label: 'Tool call parser' },
    { name: 'reasoning_parser', label: 'Reasoning parser' },
  ],
  runtime: [
    { name: 'mem_fraction_static', label: 'Mem fraction static' },
    { name: 'host', label: 'SGLang host' },
    { name: 'port', label: 'SGLang port', type: 'number' },
  ],
  mtp: [
    { name: 'speculative_algorithm', label: 'Speculative algorithm' },
    { name: 'speculative_num_steps', label: 'Speculative steps', type: 'number' },
    { name: 'speculative_eagle_topk', label: 'EAGLE top-k', type: 'number' },
    { name: 'speculative_num_draft_tokens', label: 'Draft tokens', type: 'number' },
  ],
  disaggregation: [
    { name: 'nnodes', label: 'Number of nodes', type: 'number' },
    { name: 'node_rank', label: 'Node rank', type: 'number' },
    { name: 'dist_init_addr', label: 'Dist init address' },
    { name: 'disaggregation_mode', label: 'Mode' },
    { name: 'disaggregation_transfer_backend', label: 'Transfer backend' },
    { name: 'disaggregation_ib_device', label: 'IB devices', wide: true },
  ],
  limits: [
    { name: 'max_running_requests', label: 'Max running requests', type: 'number' },
    { name: 'chunked_prefill_size', label: 'Chunked prefill size', type: 'number' },
    { name: 'max_prefill_tokens', label: 'Max prefill tokens', type: 'number' },
  ],
  env: [
    { name: 'NCCL_IB_GID_INDEX', label: 'NCCL IB GID index' },
    { name: 'NCCL_IB_HCA', label: 'NCCL IB HCA' },
    { name: 'NCCL_SOCKET_IFNAME', label: 'NCCL socket interface' },
    { name: 'NCCL_IB_TC', label: 'NCCL IB TC' },
    { name: 'NCCL_IB_TIMEOUT', label: 'NCCL IB timeout' },
    { name: 'NCCL_IB_RETRY_CNT', label: 'NCCL IB retry count' },
  ],
};

const state = {
  defaults: {},
  refreshTimer: null,
};

const form = document.querySelector('#command-form');
const statusDot = document.querySelector('#status-dot');
const statusText = document.querySelector('#status-text');
const errorBox = document.querySelector('#error-box');
const combinedOutput = document.querySelector('#combined-output');
const resetButton = document.querySelector('#reset-button');
const enableMtpInput = document.querySelector('#enable-mtp');
const mtpFields = document.querySelector('#mtp-fields');
const attentionModeInputs = Array.from(document.querySelectorAll('input[name="attention_parallel_mode"]'));
const contextBackendOptions = document.querySelector('#context-backend-options');
const moeModeInputs = Array.from(document.querySelectorAll('input[name="moe_parallel_mode"]'));
const expertOverlapOptions = document.querySelector('#expert-overlap-options');

function createField({ name, label, type = 'text', wide = false }) {
  const wrapper = document.createElement('label');
  wrapper.className = wide ? 'field wide' : 'field';
  wrapper.textContent = label;

  const input = document.createElement('input');
  input.name = name;
  input.type = type;
  input.autocomplete = 'off';

  const hint = document.createElement('span');
  hint.className = 'hint';
  hint.dataset.hintFor = name;

  wrapper.append(input, hint);
  return wrapper;
}

function renderFields() {
  const targets = {
    model: document.querySelector('#model-fields'),
    runtime: document.querySelector('#runtime-fields'),
    mtp: mtpFields,
    disaggregation: document.querySelector('#disaggregation-fields'),
    limits: document.querySelector('#limit-fields'),
    env: document.querySelector('#env-fields'),
  };

  for (const [group, fields] of Object.entries(fieldGroups)) {
    targets[group].replaceChildren(...fields.map(createField));
  }
}

function updateMtpVisibility() {
  const enabled = Boolean(enableMtpInput?.checked);
  mtpFields.hidden = !enabled;
  for (const element of mtpFields.querySelectorAll('input')) {
    element.disabled = !enabled;
  }
}

function updateContextBackendVisibility() {
  const selectedMode = attentionModeInputs.find((input) => input.checked)?.value;
  const visible = selectedMode === 'context_parallel';
  contextBackendOptions.hidden = !visible;
  for (const element of contextBackendOptions.querySelectorAll('input')) {
    element.disabled = !visible;
  }
}

function updateExpertOverlapVisibility() {
  const selectedMode = moeModeInputs.find((input) => input.checked)?.value;
  const visible = selectedMode === 'expert_parallel';
  expertOverlapOptions.hidden = !visible;
  for (const element of expertOverlapOptions.querySelectorAll('input')) {
    element.disabled = !visible;
    if (!visible) element.checked = false;
  }
}

function setStatus(message, tone = 'loading') {
  statusText.textContent = message;
  statusDot.className = `status-dot ${tone === 'ready' ? 'ready' : tone === 'error' ? 'error' : ''}`.trim();
}

function showError(message) {
  if (!message) {
    errorBox.hidden = true;
    errorBox.textContent = '';
    return;
  }
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function applyDefaults() {
  for (const element of form.elements) {
    if (!element.name) continue;
    const value = state.defaults[element.name];
    if (element.type === 'checkbox') {
      element.checked = Boolean(value);
    } else if (element.type === 'radio') {
      element.checked = element.value === String(value ?? '');
    } else {
      element.value = value ?? '';
      element.placeholder = value ?? '';
    }
  }

  for (const hint of document.querySelectorAll('[data-hint-for]')) {
    const value = state.defaults[hint.dataset.hintFor];
    hint.textContent = value === undefined ? '' : `Default: ${value}`;
  }
  updateMtpVisibility();
  updateContextBackendVisibility();
  updateExpertOverlapVisibility();
}

function collectPayload() {
  const payload = {};
  for (const element of form.elements) {
    if (!element.name || element.disabled) continue;
    if (element.type === 'radio' && !element.checked) continue;
    payload[element.name] = element.type === 'checkbox' ? element.checked : element.value;
  }
  return payload;
}

async function refreshCommand() {
  try {
    const response = await fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectPayload()),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Failed to generate command');

    combinedOutput.textContent = body.combined_shell;
    showError('');
    setStatus('Command preview ready', 'ready');
  } catch (error) {
    setStatus('API error', 'error');
    showError(error instanceof Error ? error.message : String(error));
  }
}

function scheduleRefresh() {
  window.clearTimeout(state.refreshTimer);
  state.refreshTimer = window.setTimeout(refreshCommand, 120);
}

async function loadDefaults() {
  setStatus('Loading defaults…');
  try {
    const response = await fetch('/api/defaults');
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Failed to load defaults');
    state.defaults = body.defaults;
    applyDefaults();
    await refreshCommand();
  } catch (error) {
    setStatus('Defaults unavailable', 'error');
    showError(error instanceof Error ? error.message : String(error));
  }
}

async function copyOutput(targetId, button) {
  const target = document.querySelector(`#${targetId}`);
  const text = target?.textContent ?? '';
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = original; }, 1200);
  } catch {
    showError('Clipboard access failed. Select the output text and copy it manually.');
  }
}

renderFields();
form.addEventListener('input', (event) => {
  if (event.target === enableMtpInput) updateMtpVisibility();
  if (attentionModeInputs.includes(event.target)) updateContextBackendVisibility();
  if (moeModeInputs.includes(event.target)) updateExpertOverlapVisibility();
  scheduleRefresh();
});
resetButton.addEventListener('click', () => {
  applyDefaults();
  refreshCommand();
});

document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-copy-target]');
  if (!button) return;
  copyOutput(button.dataset.copyTarget, button);
});

loadDefaults();
