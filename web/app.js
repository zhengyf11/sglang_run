const sharedModelFields = [
  { name: 'model_path', label: 'Model path', wide: true },
  { name: 'served_model_name', label: 'Served model name' },
  { name: 'tool_call_parser', label: 'Tool call parser' },
  { name: 'reasoning_parser', label: 'Reasoning parser' },
];

const profileConfigs = {
  docker_run: {
    title: 'Docker Run parameters',
    formId: 'docker-run-command-form',
    usesSharedModel: true,
    groups: {
      container: [
        { name: 'image', label: 'Image', wide: true },
        { name: 'container_name', label: 'Container name' },
        { name: 'shm_size', label: 'Shm size' },
      ],
      extra: [
        { name: 'volume', label: 'Volumes', type: 'list', placeholder: '/data/cache', samePathMapping: true, wide: true },
        { name: 'env', label: 'Environment variables', type: 'list', placeholder: 'HF_HOME=/data/cache', wide: true },
      ],
    },
  },
  prefill: {
    title: 'Prefill parameters',
    formId: 'prefill-command-form',
    groups: {
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
        { name: 'dist_init_addr', label: 'Dist init address' },
        { name: 'disaggregation_transfer_backend', label: 'Transfer backend' },
        { name: 'disaggregation_ib_device', label: 'IB devices', wide: true },
      ],
      limits: [
        { name: 'chunked_prefill_size', label: 'Chunked prefill size', type: 'number' },
        { name: 'max_prefill_tokens', label: 'Max prefill tokens', type: 'number' },
      ],
    },
  },
  decode: {
    title: 'Decode parameters',
    formId: 'decode-command-form',
    groups: {
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
        { name: 'dist_init_addr', label: 'Dist init address' },
        { name: 'disaggregation_transfer_backend', label: 'Transfer backend' },
        { name: 'disaggregation_ib_device', label: 'IB devices', wide: true },
      ],
      limits: [
        { name: 'max_running_requests', label: 'Max running requests', type: 'number' },
        { name: 'cuda_graph_max_bs', label: 'Cuda graph max bs', type: 'number' },
      ],
    },
  },
  router: {
    title: 'Router parameters',
    formId: 'router-command-form',
    groups: {
      endpoints: [
        { name: 'prefill', label: 'Prefill endpoint' },
        { name: 'decode', label: 'Decode endpoint' },
      ],
      runtime: [
        { name: 'host', label: 'Router host' },
        { name: 'port', label: 'Router port', type: 'number' },
        { name: 'policy', label: 'Policy' },
        { name: 'retry_max_retries', label: 'Retry max retries', type: 'number' },
      ],
    },
  },
};

const ignoredModelPathSegments = new Set(['mnt', 'mount', 'model', 'models', 'vllm', 'sglang', 'workspace', 'data', 'api']);
const profiles = Object.keys(profileConfigs);

const state = {
  activeProfile: 'docker_run',
  defaultsByProfile: {},
  parserMetadata: {},
  refreshTimer: null,
  lastDockerRunModelVolume: '',
};

const sharedModelForm = document.querySelector('#shared-model-form');
const sharedModelFieldsTarget = document.querySelector('[data-model-fields]');
const forms = Object.fromEntries(profiles.map((profile) => [profile, document.querySelector(`#${profileConfigs[profile].formId}`)]));
const panels = Object.fromEntries(profiles.map((profile) => [profile, document.querySelector(`[data-profile-panel="${profile}"]`)]));
const profileButtons = Array.from(document.querySelectorAll('[data-profile-button]'));
const statusDot = document.querySelector('#status-dot');
const statusText = document.querySelector('#status-text');
const errorBox = document.querySelector('#error-box');
const combinedOutput = document.querySelector('#combined-output');
const resetButton = document.querySelector('#reset-button');

function activeForm() {
  return forms[state.activeProfile];
}

function activeDefaults() {
  return state.defaultsByProfile[state.activeProfile] || {};
}

function createListField({ name, label, placeholder = '', samePathMapping = false, wide = false }) {
  const wrapper = document.createElement('div');
  wrapper.className = wide ? 'field list-field wide' : 'field list-field';
  wrapper.dataset.listField = name;

  const heading = document.createElement('div');
  heading.className = 'list-field-heading';
  const labelText = document.createElement('span');
  labelText.textContent = label;
  const addButton = document.createElement('button');
  addButton.type = 'button';
  addButton.className = 'secondary-button add-list-item-button';
  addButton.dataset.addListItem = name;
  addButton.textContent = '添加';
  heading.append(labelText, addButton);

  const list = document.createElement('div');
  list.className = 'list-items';
  list.dataset.listItems = name;
  const hiddenInput = document.createElement('input');
  hiddenInput.type = 'hidden';
  hiddenInput.name = name;
  hiddenInput.dataset.listValue = name;
  if (samePathMapping) hiddenInput.dataset.samePathMapping = 'true';
  hiddenInput.dataset.placeholder = placeholder;

  const hint = document.createElement('span');
  hint.className = 'hint';
  hint.dataset.hintFor = name;
  wrapper.append(heading, list, hiddenInput, hint);
  return wrapper;
}

function createField({ name, label, type = 'text', wide = false, ...options }, profile) {
  if (type === 'list') return createListField({ name, label, wide, ...options });

  const wrapper = document.createElement('label');
  wrapper.className = wide ? 'field wide' : 'field';
  wrapper.textContent = label;

  const input = type === 'textarea' ? document.createElement('textarea') : document.createElement('input');
  input.name = name;
  if (input instanceof HTMLInputElement) input.type = type;
  input.autocomplete = 'off';

  const hint = document.createElement('span');
  hint.className = 'hint';
  hint.dataset.hintFor = name;

  if (name === 'model_path') {
    const inputRow = document.createElement('div');
    inputRow.className = 'model-path-row';
    const identifyButton = document.createElement('button');
    identifyButton.className = 'secondary-button identify-model-button';
    identifyButton.type = 'button';
    identifyButton.dataset.identifyModel = profile;
    identifyButton.textContent = '识别模型';
    inputRow.append(input, identifyButton);
    wrapper.append(inputRow, hint);
  } else {
    wrapper.append(input, hint);
  }
  return wrapper;
}

function renderFields() {
  sharedModelFieldsTarget?.replaceChildren(...sharedModelFields.map((field) => createField(field, 'shared')));
  for (const profile of profiles) {
    const config = profileConfigs[profile];
    for (const [group, fields] of Object.entries(config.groups)) {
      const target = document.querySelector(`[data-profile-fields="${profile}:${group}"]`);
      target?.replaceChildren(...fields.map((field) => createField(field, profile)));
    }
  }
}

function getProfileControls(profile) {
  const form = forms[profile];
  return {
    form,
    defaults: state.defaultsByProfile[profile] || {},
    enableMtpInput: form?.querySelector('input[name="enable_mtp"]'),
    mtpFields: form?.querySelector('[data-mtp-fields]'),
    attentionModeInputs: Array.from(form?.querySelectorAll('input[name="attention_parallel_mode"]') || []),
    contextBackendOptions: form?.querySelector('[data-context-backend-options]'),
    pipelineOptions: form?.querySelector('[data-pipeline-options]'),
    dpSizeField: form?.querySelector('[data-dp-size-field]'),
    dpSizeInput: form?.querySelector('input[name="dp_size"]'),
    worldSizeInput: form?.querySelector('input[name="parallel_tp_size"]'),
    moeModeInputs: Array.from(form?.querySelectorAll('input[name="moe_parallel_mode"]') || []),
    expertOverlapOptions: form?.querySelector('[data-expert-overlap-options]'),
  };
}

function updateMtpVisibility(profile) {
  const { enableMtpInput, mtpFields } = getProfileControls(profile);
  if (!mtpFields) return;
  const enabled = Boolean(enableMtpInput?.checked);
  mtpFields.hidden = !enabled;
  for (const element of mtpFields.querySelectorAll('input')) element.disabled = !enabled;
}

function updateContextBackendVisibility(profile) {
  const { attentionModeInputs, contextBackendOptions } = getProfileControls(profile);
  if (!contextBackendOptions) return;
  const visible = attentionModeInputs.find((input) => input.checked)?.value === 'context_parallel';
  contextBackendOptions.hidden = !visible;
  for (const element of contextBackendOptions.querySelectorAll('input')) element.disabled = !visible;
}

function updatePipelineOptionsVisibility(profile) {
  const { attentionModeInputs, pipelineOptions } = getProfileControls(profile);
  if (!pipelineOptions) return;
  const visible = attentionModeInputs.find((input) => input.checked)?.value === 'pipeline_parallel';
  pipelineOptions.hidden = !visible;
  for (const element of pipelineOptions.querySelectorAll('input')) element.disabled = !visible;
}

function updateDpSizeVisibility(profile, { syncDefault = false } = {}) {
  const { attentionModeInputs, dpSizeField, dpSizeInput, worldSizeInput, defaults } = getProfileControls(profile);
  if (!dpSizeField || !dpSizeInput) return;
  const visible = attentionModeInputs.find((input) => input.checked)?.value === 'dp_attention';
  dpSizeField.hidden = !visible;
  dpSizeInput.disabled = !visible;
  if (visible && (syncDefault || !dpSizeInput.value)) dpSizeInput.value = worldSizeInput.value || defaults.parallel_tp_size || '';
}

function updateExpertOverlapVisibility(profile) {
  const { moeModeInputs, expertOverlapOptions } = getProfileControls(profile);
  if (!expertOverlapOptions) return;
  const visible = moeModeInputs.find((input) => input.checked)?.value === 'expert_parallel';
  expertOverlapOptions.hidden = !visible;
  for (const element of expertOverlapOptions.querySelectorAll('input')) {
    element.disabled = !visible;
    if (!visible) element.checked = false;
  }
}

function inferServedModelName(modelPath) {
  const defaults = activeDefaults();
  const parts = String(modelPath || '').replaceAll('\\', '/').split('/').filter(Boolean);
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    const part = parts[index];
    const normalized = part.trim().toLowerCase();
    if (!ignoredModelPathSegments.has(normalized) && !/^v\d+$/.test(normalized)) return part;
  }
  return defaults.served_model_name || '';
}

function parserChoice(candidate, choices, fallback) {
  return choices?.includes(candidate) ? candidate : fallback;
}

function inferModelParsers(modelPath) {
  const defaults = activeDefaults();
  const modelName = inferServedModelName(modelPath);
  const haystack = `${modelPath || ''} ${modelName}`.toLowerCase().replaceAll('_', '-');
  const toolChoices = state.parserMetadata.tool_call_parser_choices || [];
  const reasoningChoices = state.parserMetadata.reasoning_parser_choices || [];
  const fallbacks = state.parserMetadata.fallbacks || {};
  const toolFallback = parserChoice(fallbacks.tool_call_parser || defaults.tool_call_parser, toolChoices, defaults.tool_call_parser || '');
  const reasoningFallback = parserChoice(fallbacks.reasoning_parser || defaults.reasoning_parser, reasoningChoices, defaults.reasoning_parser || '');

  for (const rule of state.parserMetadata.rules || []) {
    if ((rule.patterns || []).some((pattern) => haystack.includes(pattern))) {
      return {
        tool_call_parser: parserChoice(rule.tool_call_parser, toolChoices, toolFallback),
        reasoning_parser: parserChoice(rule.reasoning_parser, reasoningChoices, reasoningFallback),
      };
    }
  }
  return { tool_call_parser: toolFallback, reasoning_parser: reasoningFallback };
}

function syncModelDerivedDefaults(modelPath, profile = state.activeProfile) {
  const servedModelInput = sharedModelForm?.querySelector('input[name="served_model_name"]');
  const toolParserInput = sharedModelForm?.querySelector('input[name="tool_call_parser"]');
  const reasoningParserInput = sharedModelForm?.querySelector('input[name="reasoning_parser"]');
  const previousProfile = state.activeProfile;
  state.activeProfile = profile;
  if (servedModelInput) servedModelInput.value = inferServedModelName(modelPath);
  const parsers = inferModelParsers(modelPath);
  const defaults = activeDefaults();
  state.activeProfile = previousProfile;
  if (toolParserInput) toolParserInput.value = profile === 'router' ? (defaults.tool_call_parser || parsers.tool_call_parser) : parsers.tool_call_parser;
  if (reasoningParserInput) reasoningParserInput.value = parsers.reasoning_parser;
}

function setStatus(message, tone = 'loading') {
  statusText.textContent = message;
  statusDot.className = `status-dot ${tone === 'ready' ? 'ready' : tone === 'error' ? 'error' : ''}`.trim();
}

function showError(message) {
  errorBox.hidden = !message;
  errorBox.textContent = message || '';
}

function applyModelDefaults(profile = state.activeProfile) {
  const defaults = state.defaultsByProfile[profile] || {};
  for (const element of sharedModelForm.elements) {
    if (!element.name) continue;
    const value = defaults[element.name];
    element.value = value ?? '';
    element.placeholder = value ?? '';
  }
  syncModelDerivedDefaults(sharedModelForm.querySelector('input[name="model_path"]')?.value || defaults.model_path || '', profile);
  for (const hint of sharedModelForm.querySelectorAll('[data-hint-for]')) {
    const value = defaults[hint.dataset.hintFor];
    hint.textContent = value === undefined ? '' : `Default: ${value}`;
  }
}

function setListFieldItems(field, values) {
  const wrapper = document.querySelector(`[data-list-field="${field}"]`);
  const list = wrapper?.querySelector(`[data-list-items="${field}"]`);
  const hiddenInput = wrapper?.querySelector(`[data-list-value="${field}"]`);
  if (!list || !hiddenInput) return;
  const normalizedValues = values.map((value) => String(value || '').trim());
  list.replaceChildren(...normalizedValues.map((value, index) => createListItem(field, value, index, hiddenInput)));
  syncListFieldValue(field);
}

function createListItem(field, value, index, hiddenInput) {
  const item = document.createElement('div');
  item.className = 'list-item';
  const input = document.createElement('input');
  input.type = 'text';
  input.autocomplete = 'off';
  input.value = value;
  input.placeholder = hiddenInput.dataset.placeholder || '';
  input.dataset.listItemInput = field;
  input.dataset.index = String(index);
  const deleteButton = document.createElement('button');
  deleteButton.type = 'button';
  deleteButton.className = 'delete-list-item-button';
  deleteButton.dataset.deleteListItem = field;
  deleteButton.dataset.index = String(index);
  deleteButton.textContent = '删除';
  item.append(input, deleteButton);
  return item;
}

function syncListFieldValue(field) {
  const wrapper = document.querySelector(`[data-list-field="${field}"]`);
  const hiddenInput = wrapper?.querySelector(`[data-list-value="${field}"]`);
  if (!wrapper || !hiddenInput) return;
  const values = Array.from(wrapper.querySelectorAll(`[data-list-item-input="${field}"]`))
    .map((input) => input.value.trim())
    .filter(Boolean);
  hiddenInput.value = values.join('\n');
}

function getListFieldItems(field) {
  const hiddenInput = document.querySelector(`[data-list-value="${field}"]`);
  return (hiddenInput?.value || '').split('\n').map((value) => value.trim()).filter(Boolean);
}

function addListFieldItem(field) {
  setListFieldItems(field, [...getListFieldItems(field), '']);
  const inputs = document.querySelectorAll(`[data-list-item-input="${field}"]`);
  inputs[inputs.length - 1]?.focus();
  scheduleRefresh();
}

function deleteListFieldItem(field, index) {
  const values = getListFieldItems(field);
  values.splice(Number(index), 1);
  setListFieldItems(field, values);
  scheduleRefresh();
}

function applyDefaults(profile = state.activeProfile) {
  const form = forms[profile];
  const defaults = state.defaultsByProfile[profile] || {};
  for (const element of form.elements) {
    if (!element.name) continue;
    const value = defaults[element.name];
    if (element.dataset.listValue) {
      setListFieldItems(element.name, String(value ?? '').split('\n'));
      if (profile === 'docker_run' && element.name === 'volume') {
        const modelPath = sharedModelForm?.querySelector('input[name="model_path"]')?.value || defaults.model_path || '';
        const modelVolume = String(modelPath || '').trim();
        state.lastDockerRunModelVolume = modelVolume ? `${modelVolume}:${modelVolume}` : '';
      }
    }
    else if (element.type === 'checkbox') element.checked = Boolean(value);
    else if (element.type === 'radio') element.checked = element.value === String(value ?? '');
    else {
      element.value = value ?? '';
      element.placeholder = value ?? '';
    }
  }

  for (const hint of form.querySelectorAll('[data-hint-for]')) {
    const value = defaults[hint.dataset.hintFor];
    hint.textContent = value === undefined ? '' : `Default: ${value}`;
  }
  updateMtpVisibility(profile);
  updateContextBackendVisibility(profile);
  updatePipelineOptionsVisibility(profile);
  updateDpSizeVisibility(profile, { syncDefault: true });
  updateExpertOverlapVisibility(profile);
}

function activeUsesSharedModel() {
  return profileConfigs[state.activeProfile]?.usesSharedModel !== false;
}

function syncDockerRunModelVolume(modelPath) {
  const modelVolume = String(modelPath || '').trim();
  const nextModelVolume = modelVolume ? `${modelVolume}:${modelVolume}` : '';
  const values = getListFieldItems('volume');
  const previousIndex = state.lastDockerRunModelVolume ? values.indexOf(state.lastDockerRunModelVolume) : -1;
  if (previousIndex >= 0) values[previousIndex] = nextModelVolume;
  else if (nextModelVolume && !values.includes(nextModelVolume)) values.push(nextModelVolume);
  state.lastDockerRunModelVolume = nextModelVolume;
  setListFieldItems('volume', values.filter(Boolean));
}

function collectPayload() {
  const payload = { profile: state.activeProfile };
  const payloadForms = activeUsesSharedModel() ? [sharedModelForm, activeForm()] : [activeForm()];
  for (const form of payloadForms) {
    for (const element of form.elements) {
      if (!element.name || element.disabled) continue;
      if (element.type === 'radio' && !element.checked) continue;
      payload[element.name] = element.type === 'checkbox' ? element.checked : element.value;
    }
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

async function loadDefaults(profile) {
  const response = await fetch(`/api/defaults?profile=${encodeURIComponent(profile)}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Failed to load ${profile} defaults`);
  state.defaultsByProfile[profile] = body.defaults;
  state.parserMetadata = body.parser_metadata || state.parserMetadata || {};
  applyDefaults(profile);
  if (profile === state.activeProfile) applyModelDefaults(profile);
}

async function loadAllDefaults() {
  setStatus('Loading defaults…');
  try {
    for (const profile of profiles) await loadDefaults(profile);
    switchProfile('docker_run', { refresh: false });
    await refreshCommand();
  } catch (error) {
    setStatus('Defaults unavailable', 'error');
    showError(error instanceof Error ? error.message : String(error));
  }
}

function switchProfile(profile, { refresh = true } = {}) {
  state.activeProfile = profile;
  for (const current of profiles) {
    panels[current].hidden = current !== profile;
    profileButtons.find((button) => button.dataset.profileButton === current)?.classList.toggle('active', current === profile);
  }
  sharedModelForm.hidden = !activeUsesSharedModel();
  if (activeUsesSharedModel()) applyModelDefaults(profile);
  if (refresh) refreshCommand();
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
sharedModelForm.addEventListener('input', (event) => {
  if (event.target.name === 'model_path') {
    syncModelDerivedDefaults(event.target.value, state.activeProfile);
    if (state.activeProfile === 'docker_run') syncDockerRunModelVolume(event.target.value);
  }
  scheduleRefresh();
});

for (const profile of profiles) {
  forms[profile].addEventListener('input', (event) => {
    const listInput = event.target.closest?.('[data-list-item-input]');
    if (listInput) syncListFieldValue(listInput.dataset.listItemInput);
    if (event.target.name === 'enable_mtp') updateMtpVisibility(profile);
    const controls = getProfileControls(profile);
    if (controls.attentionModeInputs.includes(event.target)) {
      updateContextBackendVisibility(profile);
      updatePipelineOptionsVisibility(profile);
      updateDpSizeVisibility(profile, { syncDefault: event.target.value === 'dp_attention' });
    }
    if (event.target === controls.worldSizeInput && controls.dpSizeInput?.disabled) controls.dpSizeInput.value = controls.worldSizeInput.value;
    if (controls.moeModeInputs.includes(event.target)) updateExpertOverlapVisibility(profile);
    if (profile === state.activeProfile) scheduleRefresh();
  });
}

document.addEventListener('click', (event) => {
  const profileButton = event.target.closest('[data-profile-button]');
  if (profileButton) {
    switchProfile(profileButton.dataset.profileButton);
    return;
  }
  const identifyButton = event.target.closest('[data-identify-model]');
  if (identifyButton) {
    const modelPathInput = sharedModelForm.querySelector('input[name="model_path"]');
    syncModelDerivedDefaults(modelPathInput?.value || '', state.activeProfile);
    refreshCommand();
    return;
  }
  const addListButton = event.target.closest('[data-add-list-item]');
  if (addListButton) {
    addListFieldItem(addListButton.dataset.addListItem);
    return;
  }
  const deleteListButton = event.target.closest('[data-delete-list-item]');
  if (deleteListButton) {
    deleteListFieldItem(deleteListButton.dataset.deleteListItem, deleteListButton.dataset.index);
    return;
  }
  const copyButton = event.target.closest('[data-copy-target]');
  if (copyButton) copyOutput(copyButton.dataset.copyTarget, copyButton);
});

resetButton.addEventListener('click', () => {
  applyModelDefaults(state.activeProfile);
  applyDefaults(state.activeProfile);
  refreshCommand();
});

loadAllDefaults();
