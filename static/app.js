const fileInput = document.getElementById("image");
const fileName = document.getElementById("fileName");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const drop = document.getElementById("dropzone");

function setFileUI(file){
  if(!file) return;
  fileName.textContent = file.name;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.classList.add("show");
  analyzeBtn.disabled = false;
}

fileInput?.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if(file) setFileUI(file);
});

drop?.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.style.borderColor = "rgba(124,92,255,0.65)";
});

drop?.addEventListener("dragleave", () => {
  drop.style.borderColor = "rgba(255,255,255,0.25)";
});

drop?.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.style.borderColor = "rgba(255,255,255,0.25)";
  const file = e.dataTransfer.files?.[0];
  if(!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  setFileUI(file);
});
