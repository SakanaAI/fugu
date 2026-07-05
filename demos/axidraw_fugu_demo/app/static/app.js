const state = {
  generatedDemos: [],
  axicli: "",
  serverFuguEnabled: false,
  fuguEnabled: false,
  apiKey: "",
  baseUrl: "",
  baseUrlOverride: "",
  allowedBaseUrls: [],
  customBaseUrlAllowed: false,
  plotterEnabled: false,
  fuguSettings: { stream: "false", max_output_tokens: "1536", reasoning_effort: "" },
  selected: null,
  poll: 0,
  cameraStream: null,
  cameraMode: "contour",
  displayModel: "Sakana Fugu",
  apiTestRunning: false,
};

const $ = (id) => document.getElementById(id);
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8776" : "";
const EMPTY_PREVIEW_SVG = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 297 210'%3E%3Crect width='297' height='210' fill='white'/%3E%3Cpath d='M28 168 C62 126 96 144 128 104 S199 78 246 44' fill='none' stroke='%23d5d9d6' stroke-width='1.2' stroke-linecap='round'/%3E%3Ccircle cx='222' cy='58' r='18' fill='none' stroke='%23d5d9d6' stroke-width='1.2'/%3E%3Cpath d='M32 184 H265' stroke='%23e4e7e4' stroke-width='0.8'/%3E%3C/svg%3E";
const API_KEY_STORAGE = "sakana-fugu-demo-api-key-v2";
const BASE_URL_STORAGE = "sakana-fugu-demo-base-url-v1";

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function demoName(demo) {
  return demo.demo.replaceAll("_", " ");
}

function demoDescription(demo) {
  return demo.description || demo.prompt || "";
}

function svgBasename(path) {
  return path.split("/").pop();
}

function generatedOnly(demos) {
  const generated = demos.filter((demo) => (demo.category || "").startsWith("Generated"));
  return generated.length ? generated : demos;
}

function quotaMessageFromText(text = "") {
  if (!/usage_limit_reached|Subscription window is exhausted|HTTP 429/i.test(text)) {
    return "";
  }
  const resetMatch = text.match(/Try again after ([0-9T:+.-]+Z?)/i);
  if (!resetMatch) {
    return "Fugu quota exhausted. Safe fallback preview shown.";
  }
  const resetDate = new Date(resetMatch[1]);
  if (Number.isNaN(resetDate.getTime())) {
    return "Fugu quota exhausted. Safe fallback preview shown.";
  }
  return `Fugu quota exhausted until ${resetDate.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })}. Safe fallback preview shown.`;
}

function fallbackMessageForDemo(demo) {
  const warnings = (demo?.warnings || []).join(" ");
  if (/safe fallback/i.test(`${demo?.description || ""} ${warnings}`)) {
    return "Safe fallback preview shown; Fugu did not generate this plot.";
  }
  return "";
}

function loadApiKey() {
  try {
    state.apiKey = sessionStorage.getItem(API_KEY_STORAGE) || "";
  } catch {
    state.apiKey = "";
  }
  $("apiKeyInput").value = state.apiKey;
}

function persistApiKey() {
  try {
    if (state.apiKey) {
      sessionStorage.setItem(API_KEY_STORAGE, state.apiKey);
    } else {
      sessionStorage.removeItem(API_KEY_STORAGE);
    }
  } catch {
    // Session storage can be unavailable in strict browser modes; keep the key in memory.
  }
}

function syncApiKeyFromInput() {
  const inputValue = $("apiKeyInput").value.trim();
  if (inputValue !== state.apiKey) {
    state.apiKey = inputValue;
    persistApiKey();
  }
  return state.apiKey;
}

function normalizeBaseUrl(value) {
  return (value || "").trim().replace(/\/+$/, "");
}

function loadBaseUrlOverride() {
  try {
    state.baseUrlOverride = sessionStorage.getItem(BASE_URL_STORAGE) || "";
  } catch {
    state.baseUrlOverride = "";
  }
  if (state.baseUrlOverride) {
    $("baseUrlInput").value = state.baseUrlOverride;
  }
}

function persistBaseUrlOverride() {
  try {
    if (state.baseUrlOverride) {
      sessionStorage.setItem(BASE_URL_STORAGE, state.baseUrlOverride);
    } else {
      sessionStorage.removeItem(BASE_URL_STORAGE);
    }
  } catch {
    // Session storage can be unavailable; keep the override in memory.
  }
}

function selectedBaseUrl() {
  return normalizeBaseUrl(state.baseUrlOverride || state.baseUrl);
}

function apiEndpointLabel(value = selectedBaseUrl()) {
  if (!state.baseUrlOverride && normalizeBaseUrl(value) === normalizeBaseUrl(state.baseUrl)) {
    return "default API endpoint";
  }
  return baseUrlLabel(value);
}

function serverKeyUsableForSelectedBaseUrl() {
  return state.serverFuguEnabled && state.allowedBaseUrls.includes(selectedBaseUrl());
}

function syncBaseUrlFromInput() {
  const inputValue = normalizeBaseUrl($("baseUrlInput").value);
  const defaultValue = normalizeBaseUrl(state.baseUrl);
  state.baseUrlOverride = inputValue && inputValue !== defaultValue ? inputValue : "";
  persistBaseUrlOverride();
  $("baseUrlInput").value = state.baseUrlOverride;
  updateBaseUrlHelp();
  return selectedBaseUrl();
}

function baseUrlLabel(value) {
  try {
    const url = new URL(value);
    return `${url.hostname}${url.pathname.replace(/\/$/, "")}`;
  } catch {
    return value || "API";
  }
}

function renderBaseUrlOptions() {
  const datalist = $("baseUrlOptions");
  datalist.textContent = "";
  state.allowedBaseUrls
    .filter((url) => normalizeBaseUrl(url) !== normalizeBaseUrl(state.baseUrl))
    .forEach((url) => {
    const option = document.createElement("option");
    option.value = url;
    datalist.append(option);
  });
}

function updateBaseUrlHelp() {
  const selected = selectedBaseUrl();
  const defaultValue = normalizeBaseUrl(state.baseUrl);
  const known = state.allowedBaseUrls.includes(selected);
  const host = baseUrlLabel(selected);
  if (!state.baseUrlOverride || selected === defaultValue) {
    $("baseUrlHelp").textContent = "Default API endpoint.";
  } else if (known || state.customBaseUrlAllowed) {
    $("baseUrlHelp").textContent = known
      ? `Testing override: ${host}`
      : `Testing custom host: ${host}. Use a browser key.`;
  } else {
    $("baseUrlHelp").textContent = "This host needs SAKANA_ALLOWED_BASE_URLS or SAKANA_ALLOW_CUSTOM_BASE_URLS=true.";
  }
}

function updateFuguAvailability() {
  const settings = state.fuguSettings || {};
  const modeText = settings.stream === "true" ? "streaming" : "fast";
  const reasoning = settings.reasoning_effort ? `, reasoning ${settings.reasoning_effort}` : "";
  const endpoint = state.baseUrlOverride ? ` · ${baseUrlLabel(selectedBaseUrl())}` : "";
  const serverKeyUsable = serverKeyUsableForSelectedBaseUrl();
  updateBaseUrlHelp();
  if (serverKeyUsable) {
    state.apiKey = "";
    $("apiKeyInput").value = "";
    $("apiKeyInput").disabled = true;
    $("apiKeyInput").placeholder = "Using server SAKANA_API_KEY";
    $("apiKeyHelp").textContent = "Server key active; browser key entry is locked.";
    try {
      sessionStorage.removeItem(API_KEY_STORAGE);
    } catch {
      // Session storage can be unavailable in strict browser modes.
    }
  } else {
    $("apiKeyInput").disabled = false;
    $("apiKeyInput").placeholder = "API key";
    $("apiKeyHelp").textContent = state.serverFuguEnabled
      ? "Custom API host: enter a browser/session key; server key is not sent."
      : "Enter a browser key for this session, or use offline fallback.";
    syncApiKeyFromInput();
  }
  state.fuguEnabled = serverKeyUsable || Boolean(state.apiKey) || !state.serverFuguEnabled;
  $("fuguStatus").textContent = serverKeyUsable
    ? `${state.displayModel} ready · server key${endpoint} · ${modeText}${reasoning}`
    : state.apiKey
      ? `${state.displayModel} ready · browser key${endpoint} · ${modeText}${reasoning}`
      : state.serverFuguEnabled
        ? `${state.displayModel} custom API needs browser key`
      : `${state.displayModel} offline fallback · add a key in settings`;
}

function renderEmptyPreview() {
  $("demoTitle").textContent = "No generated plot yet";
  $("demoDescription").textContent = "Use a prompt or camera frame to make the next plot.";
  $("metricStrip").innerHTML = "";
  $("previewImage").src = EMPTY_PREVIEW_SVG;
  $("previewImage").classList.add("empty");
  $("previewImage").alt = "Generated AxiDraw artwork preview";
  $("plotCommand").textContent = "Generate a plot first, then run Plot Preview.";
}

function renderSelected() {
  if (!state.selected) {
    renderEmptyPreview();
    return;
  }
  const demo = state.selected;
  const metrics = demo.metrics || {};
  $("demoTitle").textContent = demoName(demo);
  $("demoDescription").textContent = fallbackMessageForDemo(demo) || demoDescription(demo);
  $("metricStrip").innerHTML = [
    metrics.estimated_print_time && `plot time ${metrics.estimated_print_time}`,
    metrics.path_to_draw_m && `ink ${metrics.path_to_draw_m} m`,
    metrics.pen_up_travel_m && `travel ${metrics.pen_up_travel_m} m`,
    `warnings ${demo.warning_count || 0}`,
  ]
    .filter(Boolean)
    .map((x) => `<span class="metric">${x}</span>`)
    .join("");
  $("previewImage").src = `${API_BASE}/outputs/${svgBasename(demo.svg || demo.preview_svg)}`;
  $("previewImage").classList.remove("empty");
  $("previewImage").alt = `${demoName(demo)} AxiDraw artwork preview`;
  $("plotCommand").textContent = `${state.axicli} ${demo.svg} --mode preview`;
}

function selectLatestGenerated() {
  state.selected = state.generatedDemos[0] || null;
  renderSelected();
}

function setBusy(job) {
  updateFuguAvailability();
  const running = Boolean(job && job.running);
  const kind = job?.kind || "";
  const quotaMessage = quotaMessageFromText(job?.output || "");
  $("generateBtn").disabled = running || !state.fuguEnabled;
  $("generateBtn").textContent = running && kind === "generate" ? "Generating..." : "Generate from Prompt";
  const cameraNeedsFugu = state.cameraMode === "fugu";
  $("cameraGenerateBtn").disabled = running || !state.cameraStream || (cameraNeedsFugu && !state.fuguEnabled);
  $("cameraGenerateBtn").textContent = running && kind === "vectorize"
    ? "Vectorizing..."
    : running && kind === "cleanup"
      ? "Cleaning..."
    : running && kind === "generate"
      ? "Generating..."
      : "Plot Webcam";
  $("previewBtn").disabled = running || !state.selected;
  $("previewBtn").textContent = running && kind === "preview" ? "Previewing..." : "Plot Preview";
  $("plotBtn").disabled = running || !state.selected || !state.plotterEnabled;
  $("plotBtn").textContent = running && kind === "plot" ? "Plotting..." : "Send to AxiDraw";
  $("testApiBtn").disabled = running || state.apiTestRunning || (!state.serverFuguEnabled && !state.apiKey);
  $("testApiBtn").textContent = state.apiTestRunning ? "Testing..." : "Test API";
  $("stopBtn").disabled = !running;
  $("stopBtn").hidden = !running;
  $("stageOverlay").hidden = !running;
  $("stageOverlayText").textContent = kind === "preview"
    ? "Running Plot Preview..."
    : kind === "vectorize"
      ? "Vectorizing webcam frame..."
      : kind === "cleanup"
        ? "Cleaning contour with Fugu..."
      : kind === "plot"
        ? "Sending to AxiDraw..."
        : "Generating with Fugu...";
  $("runStatus").textContent = running
    ? kind === "preview"
      ? "Running AxiDraw Plot Preview..."
      : kind === "vectorize"
        ? "Vectorizing webcam frame..."
        : kind === "cleanup"
          ? "Cleaning webcam contour with Fugu..."
        : kind === "plot"
          ? "Sending plot to AxiDraw..."
          : "Generating with Fugu..."
    : quotaMessage
      ? quotaMessage
      : job?.kind
      ? job.returncode === 0
        ? `Last ${job.kind} finished with exit 0.`
        : `Last ${job.kind} failed with exit ${job.returncode}.`
      : "Ready.";
  if (quotaMessage) {
    $("fuguStatus").textContent = "Fugu quota exhausted · fallback active";
  }
}

function renderJob(job) {
  setBusy(job);
  const label = job.running
    ? `[running ${job.kind} ${job.demo}]`
    : job.kind
      ? `[last ${job.kind} ${job.demo}, exit ${job.returncode}]`
      : "[idle]";
  $("jobLog").textContent = `${label}\n${job.output || ""}`;
  $("jobLog").scrollTop = $("jobLog").scrollHeight;
}

function renderFailure(kind, error) {
  const message = error?.stack || error?.message || String(error);
  renderJob({
    running: false,
    kind,
    demo: "",
    returncode: "error",
    output: message,
  });
  const label = kind === "preview"
    ? "Plot Preview"
    : kind === "vectorize"
      ? "Vectorization"
      : kind === "plot"
        ? "AxiDraw plot"
        : "Generation";
  $("runStatus").textContent = `${label} failed: ${error?.message || error}`;
}

async function refresh() {
  const data = await api("/api/demos");
  state.generatedDemos = generatedOnly(data.demos || []);
  state.axicli = data.axicli || "axicli";
  state.serverFuguEnabled = Boolean(data.fugu_enabled);
  state.baseUrl = data.base_url || state.baseUrl;
  state.allowedBaseUrls = data.allowed_base_urls || state.allowedBaseUrls;
  state.customBaseUrlAllowed = Boolean(data.custom_base_url_allowed);
  state.plotterEnabled = Boolean(data.plotter_enabled);
  state.fuguSettings = data.fugu_settings || state.fuguSettings;
  state.displayModel = data.display_model || "Sakana Fugu";
  renderBaseUrlOptions();
  if (!state.baseUrlOverride) {
    $("baseUrlInput").value = "";
  }
  updateFuguAvailability();
  $("hardwarePath").textContent = state.plotterEnabled
    ? "AxiDraw plotting enabled. Send to AxiDraw asks for confirmation before hardware moves."
    : "AxiDraw plotting unavailable. Install pyaxidraw or check the Python environment.";
  if (state.selected) {
    state.selected = state.generatedDemos.find((demo) => demo.demo === state.selected.demo) || null;
  }
  if (!state.selected) selectLatestGenerated();
  renderSelected();
  renderJob(data.job || {});
}

async function pollJob() {
  const job = await api("/api/job");
  renderJob(job);
  if (!job.running) {
    const previous = state.selected?.demo;
    await refresh();
    if (["generate", "vectorize", "cleanup"].includes(job.kind) && job.returncode === 0) {
      selectLatestGenerated();
    } else if (previous && state.selected?.demo !== previous) {
      renderSelected();
    }
  }
  return job;
}

function startPolling(delay = 1200) {
  clearTimeout(state.poll);
  state.poll = setTimeout(async () => {
    try {
      const job = await pollJob();
      if (job.running) {
        startPolling(1200);
      } else {
        state.poll = 0;
      }
    } catch (error) {
      $("jobLog").textContent = error.message;
      startPolling(5000);
    }
  }, delay);
}

async function startCamera() {
  if (state.cameraStream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera capture is unavailable in this browser.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 720 }, height: { ideal: 540 }, facingMode: "user" },
    audio: false,
  });
  state.cameraStream = stream;
  $("cameraPreview").srcObject = stream;
  $("cameraStatus").textContent = "Camera ready.";
  setBusy({});
}

function cameraErrorMessage(error) {
  const message = error?.message || String(error);
  if (error?.name === "NotAllowedError" || /permission/i.test(message)) {
    return "Camera permission is denied for this browser. Enable camera access for 127.0.0.1, then click Start Camera again.";
  }
  if (error?.name === "NotFoundError" || /not found|no camera|device/i.test(message)) {
    return "No camera device is available to this browser.";
  }
  return message;
}

function captureFrame(options = {}) {
  const video = $("cameraPreview");
  if (!state.cameraStream || video.readyState < 2) {
    throw new Error("camera is not ready yet");
  }
  const width = options.width || 420;
  const quality = options.quality || 0.72;
  const canvas = $("cameraCanvas");
  const height = Math.round(width * (video.videoHeight || 540) / (video.videoWidth || 720));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, width, height);
  context.setTransform(1, 0, 0, 1, 0, 0);
  return canvas.toDataURL("image/jpeg", quality);
}

async function generatePlot() {
  const baseUrl = syncBaseUrlFromInput();
  const apiKey = serverKeyUsableForSelectedBaseUrl() ? "" : syncApiKeyFromInput();
  const prompt = $("promptInput").value.trim();
  const model = $("modelSelect").value;
  $("runStatus").textContent = "Starting Fugu generation...";
  $("generateBtn").textContent = "Generating...";
  $("generateBtn").disabled = true;
  $("stageOverlay").hidden = false;
  $("stageOverlayText").textContent = "Generating with Fugu...";
  renderJob(await api("/api/generate", {
    method: "POST",
    body: JSON.stringify({ prompt, model, api_key: apiKey, base_url: baseUrl }),
  }));
  startPolling(500);
}

async function vectorizeWebcam(imageDataUrl) {
  $("runStatus").textContent = "Starting local webcam vectorization...";
  $("cameraGenerateBtn").disabled = true;
  $("stageOverlay").hidden = false;
  $("stageOverlayText").textContent = "Vectorizing webcam frame...";
  renderJob(await api("/api/vectorize-camera", {
    method: "POST",
    body: JSON.stringify({ image_data_url: imageDataUrl }),
  }));
  startPolling(500);
}

async function cleanupWebcamWithFugu(imageDataUrl) {
  const baseUrl = syncBaseUrlFromInput();
  const apiKey = serverKeyUsableForSelectedBaseUrl() ? "" : syncApiKeyFromInput();
  const model = $("modelSelect").value;
  const prompt = "Clean and simplify the fast webcam contour plan while preserving realistic subject geometry and making it easy to plot.";
  $("runStatus").textContent = "Starting Fugu contour cleanup...";
  $("cameraGenerateBtn").textContent = "Cleaning...";
  $("cameraGenerateBtn").disabled = true;
  $("stageOverlay").hidden = false;
  $("stageOverlayText").textContent = "Cleaning contour with Fugu...";
  renderJob(await api("/api/vectorize-camera", {
    method: "POST",
    body: JSON.stringify({
      image_data_url: imageDataUrl,
      cleanup: true,
      model,
      prompt,
      api_key: apiKey,
      base_url: baseUrl,
    }),
  }));
  startPolling(500);
}

async function plotSelectedDemo() {
  if (!state.selected) return;
  const name = demoName(state.selected);
  const confirmed = window.confirm(
    `Send "${name}" to the AxiDraw now?\n\nConnect USB, set paper and pen height, and keep clear of the machine.`
  );
  if (!confirmed) {
    $("runStatus").textContent = "AxiDraw plot cancelled.";
    return;
  }
  $("runStatus").textContent = "Starting AxiDraw plot...";
  $("plotBtn").textContent = "Plotting...";
  $("plotBtn").disabled = true;
  $("stageOverlay").hidden = false;
  $("stageOverlayText").textContent = "Sending to AxiDraw...";
  renderJob(await api("/api/plot", {
    method: "POST",
    body: JSON.stringify({ demo: state.selected.demo, speed: 50 }),
  }));
  startPolling(500);
}

async function testApiEndpoint() {
  const baseUrl = syncBaseUrlFromInput();
  const apiKey = serverKeyUsableForSelectedBaseUrl() ? "" : syncApiKeyFromInput();
  state.apiTestRunning = true;
  $("apiTestResult").textContent = `Testing ${apiEndpointLabel(baseUrl)}...`;
  setBusy({});
  let statusMessage = "";
  try {
    const result = await api("/api/test-api", {
      method: "POST",
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
    });
    const models = (result.models || []).join(", ") || "no models listed";
    $("apiTestResult").textContent = `OK: ${models}`;
    statusMessage = `API test passed at ${apiEndpointLabel(result.base_url || baseUrl)}.`;
  } catch (error) {
    $("apiTestResult").textContent = error.message;
    statusMessage = `API test failed: ${error.message}`;
  } finally {
    state.apiTestRunning = false;
    setBusy({});
    if (statusMessage) {
      $("runStatus").textContent = statusMessage;
    }
  }
}

window.axidrawTestApiEndpoint = testApiEndpoint;

$("baseUrlInput").addEventListener("input", () => {
  syncBaseUrlFromInput();
  updateFuguAvailability();
  $("apiTestResult").textContent = "";
  setBusy({});
});

$("baseUrlInput").addEventListener("change", () => {
  syncBaseUrlFromInput();
  updateFuguAvailability();
  $("apiTestResult").textContent = "";
  setBusy({});
});

$("apiKeyInput").addEventListener("input", () => {
  syncApiKeyFromInput();
  updateFuguAvailability();
  $("apiTestResult").textContent = "";
  setBusy({});
});

$("apiKeyInput").addEventListener("change", () => {
  syncApiKeyFromInput();
  updateFuguAvailability();
  $("apiTestResult").textContent = "";
  setBusy({});
});

$("testApiBtn").addEventListener("click", async () => {
  await testApiEndpoint();
});

$("previewBtn").addEventListener("click", async () => {
  try {
    renderJob(await api("/api/preview", {
      method: "POST",
      body: JSON.stringify({ demo: state.selected.demo }),
    }));
    startPolling(500);
  } catch (error) {
    renderFailure("preview", error);
  }
});

$("generateBtn").addEventListener("click", async () => {
  try {
    await generatePlot();
  } catch (error) {
    renderFailure("generate", error);
  }
});

$("plotBtn").addEventListener("click", async () => {
  try {
    await plotSelectedDemo();
  } catch (error) {
    renderFailure("plot", error);
  }
});

$("startCameraBtn").addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (error) {
    $("cameraStatus").textContent = cameraErrorMessage(error);
  }
});

$("cameraModeSelect").addEventListener("change", () => {
  state.cameraMode = $("cameraModeSelect").value;
  $("cameraStatus").textContent = state.cameraMode === "contour"
    ? "Contour mode uses the visible camera frame."
    : "Fugu cleanup simplifies the fast contour while preserving the real frame.";
  setBusy({});
});

$("cameraGenerateBtn").addEventListener("click", async () => {
  try {
    const imageDataUrl = captureFrame();
    if (state.cameraMode === "contour") {
      await vectorizeWebcam(imageDataUrl);
    } else {
      await cleanupWebcamWithFugu(imageDataUrl);
    }
  } catch (error) {
    renderFailure(state.cameraMode === "contour" ? "vectorize" : "generate", error);
    $("cameraStatus").textContent = cameraErrorMessage(error);
  }
});

$("stopBtn").addEventListener("click", async () => {
  renderJob(await api("/api/stop", { method: "POST", body: "{}" }));
  startPolling(500);
});

loadApiKey();
loadBaseUrlOverride();
refresh().catch((error) => {
  document.body.innerHTML = `<pre>${error.stack}</pre>`;
});
