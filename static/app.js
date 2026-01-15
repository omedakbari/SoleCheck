const form = document.getElementById("demoForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");

const modeRadios = Array.from(document.querySelectorAll('input[name="mode"]'));
const singleBlock = document.getElementById("singleBlock");
const pairBlock = document.getElementById("pairBlock");

// Single inputs
const image1 = document.getElementById("image1");
const fileName1 = document.getElementById("fileName1");
const preview1 = document.getElementById("preview1");

// Pair inputs
const imageLeft = document.getElementById("imageLeft");
const imageRight = document.getElementById("imageRight");
const fileNameL = document.getElementById("fileNameL");
const fileNameR = document.getElementById("fileNameR");
const previewL = document.getElementById("previewL");
const previewR = document.getElementById("previewR");

function setPreview(file, previewEl, nameEl){
  if(!file) return;
  nameEl.textContent = file.name;
  const url = URL.createObjectURL(file);
  previewEl.src = url;
  previewEl.classList.add("show");
}

function currentMode(){
  const selected = modeRadios.find(r => r.checked);
  return selected ? selected.value : "single";
}

function updateModeUI(){
  const mode = currentMode();
  if(mode === "single"){
    singleBlock.style.display = "";
    pairBlock.style.display = "none";
  } else {
    singleBlock.style.display = "none";
    pairBlock.style.display = "";
  }
  validateReady();
}

function validateReady(){
  const mode = currentMode();
  let ok = false;

  if(mode === "single"){
    ok = !!(image1 && image1.files && image1.files[0]);
  } else {
    ok = !!(imageLeft && imageLeft.files && imageLeft.files[0] && imageRight && imageRight.files && imageRight.files[0]);
  }
  analyzeBtn.disabled = !ok;
}

modeRadios.forEach(r => r.addEventListener("change", updateModeUI));

image1?.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if(f){
    setPreview(f, preview1, fileName1);
  }
  validateReady();
});

imageLeft?.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if(f){
    setPreview(f, previewL, fileNameL);
  }
  validateReady();
});

imageRight?.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if(f){
    setPreview(f, previewR, fileNameR);
  }
  validateReady();
});

form?.addEventListener("submit", () => {
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";
  if(loading) loading.style.display = "flex";
});

// init
updateModeUI();
validateReady();
