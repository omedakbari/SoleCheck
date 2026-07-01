# SoleCheck

Upload a photo of your shoe sole and get a gait analysis — wear pattern, pronation tendency, and shoe recommendations — powered by a custom computer vision pipeline.

**No account. No cloud storage. Images never leave your machine.**

---

## What it does

SoleCheck analyzes the outsole of a worn shoe to detect where material has worn smooth. It maps that wear spatially across heel, midfoot, and forefoot zones, then classifies pronation tendency (neutral, overpronation, supination) and gives personalized shoe recommendations based on your activity and surface.

You can upload a single shoe or a left/right pair for an asymmetry comparison.

---

## Demo

> Drop a photo of a worn shoe sole. Results appear in a few seconds.

Results include:
- A wear heatmap overlaid on the aligned sole image
- Classified wear pattern with confidence score
- Strike and pronation notes written in plain English
- Shoe recommendations matched to your activity
- Downloadable overlay PNG, JSON metrics, and ZIP bundle

---

## How the CV pipeline works

The analysis runs entirely on the server with no ML model — just classical computer vision.

**1. Photo quality gate**
Laplacian variance flags blurry images. Mean brightness catches photos that are too dark or overexposed.

**2. Sole segmentation**
Otsu thresholding runs on both the grayscale channel and the HSV saturation channel; the higher-confidence mask wins. If segmentation confidence is below 0.45, GrabCut refines it using the bounding box as a seed rectangle. The largest contour is kept.

**3. Rotation normalization**
`minAreaRect` finds the minimum bounding rectangle of the sole contour. The image rotates so the sole sits vertically, giving consistent top/bottom/left/right orientation for every photo.

**4. Wear scoring**
Three signals combine to estimate wear:
- Laplacian magnitude (45% weight) — low texture = smoother = more worn
- Sobel gradient magnitude (30% weight) — low gradient = flatter surface
- HSV saturation (25% weight) — worn rubber loses colour

All three are normalized within the sole mask and blended. The result is a per-pixel wear map rendered as an INFERNO colormap overlay.

**5. Zone classification**
The sole splits into three horizontal bands: forefoot (top 30%), midfoot (30–70%), heel (bottom 70–100%), and two lateral halves. Regional texture means convert to wear scores. Ratios between zones determine heel/forefoot strike tendency and medial/lateral pronation class.

**6. Pair asymmetry**
For left/right pairs, heel-fore and medial-lateral ratios compare across both shoes. Differences above threshold flag asymmetric loading.

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask |
| Computer vision | OpenCV (Otsu, GrabCut, Laplacian, Sobel, morphology) |
| Image handling | Pillow, NumPy |
| Frontend | Vanilla JS, HTML/CSS |
| Serving | Gunicorn |

No PyTorch. No TensorFlow. No pretrained model. The pipeline is hand-built using classical CV techniques.

---

## Run it locally

```bash
git clone https://github.com/omedakbari/SoleCheck
cd SoleCheck

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python3 app.py
```

Open http://127.0.0.1:5000

No environment variables needed to run locally. The AI chat feature requires an `ANTHROPIC_API_KEY` but everything else works without it.

---

## Project structure

```
solecheck/
├── app.py          # Flask routes, LRU result cache, download endpoints
├── analyzer.py     # Full CV pipeline: segmentation, rotation, wear scoring, classification
├── templates/
│   └── index.html  # Single-page UI with drag-and-drop upload and AI chat
├── static/
│   ├── styles.css
│   ├── app.js      # Drag-and-drop, loading steps, async chat
│   └── samples/    # Sample shoe images for the demo buttons
└── requirements.txt
```

---

## Privacy

- Uploaded images are processed in memory and never written to disk
- The optional request log stores metrics only (wear pattern, activity, surface) — no image data, no hashes linked to personal information
- No login, no cookies, no tracking

---

*Informational guidance only. Not a medical diagnosis.*
