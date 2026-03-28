// ============================================================
// SoleCheck — app.js  (full rewrite)
// ============================================================

// ─── Elements ────────────────────────────────────────────────
const form        = document.getElementById("demoForm");
const analyzeBtn  = document.getElementById("analyzeBtn");
const loading     = document.getElementById("loading");
const modeRadios  = Array.from(document.querySelectorAll('input[name="mode"]'));
const singleBlock = document.getElementById("singleBlock");
const pairBlock   = document.getElementById("pairBlock");

// Single
const image1     = document.getElementById("image1");
const fileName1  = document.getElementById("fileName1");
const preview1   = document.getElementById("preview1");
const dropSingle = document.getElementById("dropSingle");

// Pair
const imageLeft  = document.getElementById("imageLeft");
const imageRight = document.getElementById("imageRight");
const fileNameL  = document.getElementById("fileNameL");
const fileNameR  = document.getElementById("fileNameR");
const previewL   = document.getElementById("previewL");
const previewR   = document.getElementById("previewR");
const dropLeft   = document.getElementById("dropLeft");
const dropRight  = document.getElementById("dropRight");

// ─── Helpers ─────────────────────────────────────────────────

function setPreview(file, previewEl, nameEl) {
  if (!file) return;
  nameEl.textContent = file.name;
  previewEl.src = URL.createObjectURL(file);
  previewEl.classList.add("show");
}

function assignFile(file, input, previewEl, nameEl) {
  try {
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
  } catch (_) {}
  setPreview(file, previewEl, nameEl);
  validateReady();
}

function currentMode() {
  return (modeRadios.find(r => r.checked) || {}).value || "single";
}

function updateModeUI() {
  const m = currentMode();
  if (singleBlock) singleBlock.style.display = m === "single" ? "" : "none";
  if (pairBlock)   pairBlock.style.display   = m === "pair"   ? "" : "none";
  validateReady();
}

function validateReady() {
  const m = currentMode();
  const ok = m === "single"
    ? !!(image1?.files?.[0])
    : !!(imageLeft?.files?.[0] && imageRight?.files?.[0]);
  if (analyzeBtn) analyzeBtn.disabled = !ok;
}

// ─── Drag and Drop ───────────────────────────────────────────

function setupDrop(dropEl, input, previewEl, nameEl) {
  if (!dropEl || !input) return;

  dropEl.style.cursor = "pointer";
  dropEl.title = "Drag an image here, or click to browse";

  ["dragenter", "dragover"].forEach(evt =>
    dropEl.addEventListener(evt, e => {
      e.preventDefault();
      dropEl.classList.add("drag-over");
    })
  );

  dropEl.addEventListener("dragleave", e => {
    if (!dropEl.contains(e.relatedTarget)) dropEl.classList.remove("drag-over");
  });

  dropEl.addEventListener("dragend", () => dropEl.classList.remove("drag-over"));

  dropEl.addEventListener("drop", e => {
    e.preventDefault();
    dropEl.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      assignFile(file, input, previewEl, nameEl);
    }
  });

  // Click on drop zone opens file picker
  dropEl.addEventListener("click", e => {
    if (e.target === input) return; // avoid double-trigger
    input.click();
  });
}

setupDrop(dropSingle, image1,    preview1, fileName1);
setupDrop(dropLeft,  imageLeft,  previewL, fileNameL);
setupDrop(dropRight, imageRight, previewR, fileNameR);

// ─── File input change listeners ─────────────────────────────

image1?.addEventListener("change",    e => { setPreview(e.target.files?.[0], preview1, fileName1); validateReady(); });
imageLeft?.addEventListener("change", e => { setPreview(e.target.files?.[0], previewL, fileNameL); validateReady(); });
imageRight?.addEventListener("change",e => { setPreview(e.target.files?.[0], previewR, fileNameR); validateReady(); });

// ─── Mode toggle ─────────────────────────────────────────────

modeRadios.forEach(r => r.addEventListener("change", updateModeUI));

// ─── Loading steps ───────────────────────────────────────────

const LOADING_STEPS = [
  "Finding the sole…",
  "Mapping where it's worn…",
  "Reading the texture…",
  "Pulling it together…",
];
let _stepTimer = null;

function startLoadingSteps() {
  const el = loading?.querySelector(".loadingText");
  if (!el) return;
  let i = 0;
  el.textContent = LOADING_STEPS[0];
  _stepTimer = setInterval(() => {
    i = (i + 1) % LOADING_STEPS.length;
    el.textContent = LOADING_STEPS[i];
  }, 1500);
}

// ─── Form submit ─────────────────────────────────────────────

form?.addEventListener("submit", () => {
  if (analyzeBtn) { analyzeBtn.disabled = true; analyzeBtn.textContent = "Analyzing…"; }
  if (loading) loading.style.display = "flex";
  startLoadingSteps();
});

// ─── Metric bar animation ────────────────────────────────────

function animateBars() {
  document.querySelectorAll(".barFill[data-pct]").forEach(bar => {
    const pct = parseFloat(bar.dataset.pct) || 0;
    bar.style.width = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bar.style.width = pct + "%";
    }));
  });
}

// ─── Chat ────────────────────────────────────────────────────

const chatFormEl   = document.getElementById("chatForm");
const chatInput    = document.getElementById("chatInput");
const chatSend     = document.getElementById("chatSend");
const chatMessages = document.getElementById("chatMessages");

const chatHistory = [];

function appendMsg(role, text) {
  if (!chatMessages) return null;
  const d = document.createElement("div");
  d.className = `chatMsg chatMsg-${role}`;
  d.textContent = text;
  chatMessages.appendChild(d);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return d;
}

async function sendChat(text) {
  if (!text.trim()) return;
  // hide suggestion pills after first send
  document.querySelectorAll(".chatSuggestion").forEach(b => b.style.display = "none");

  chatHistory.push({ role: "user", content: text });
  appendMsg("user", text);
  if (chatInput) chatInput.value = "";
  if (chatSend)  chatSend.disabled = true;

  const thinking = appendMsg("ai", "Thinking…");
  thinking?.classList.add("chatMsg-thinking");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        context: window.SOLECHECK_CONTEXT || null,
        history: chatHistory.slice(0, -1),
      }),
    });
    const data = await res.json();
    thinking?.remove();
    const reply = data.response || data.error || "No response.";
    chatHistory.push({ role: "assistant", content: reply });
    appendMsg("ai", reply);
  } catch {
    thinking?.remove();
    appendMsg("ai", "Couldn't reach the AI. Check your connection and try again.");
  } finally {
    if (chatSend) chatSend.disabled = false;
  }
}

chatFormEl?.addEventListener("submit", e => {
  e.preventDefault();
  sendChat(chatInput?.value?.trim() || "");
});

document.querySelectorAll(".chatSuggestion").forEach(btn => {
  btn.addEventListener("click", () => sendChat(btn.dataset.q || ""));
});

// ─── Init ────────────────────────────────────────────────────

updateModeUI();
validateReady();
animateBars();
