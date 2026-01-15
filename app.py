# app.py  (COPY/PASTE WHOLE FILE)

import os
import io
import hashlib
from collections import OrderedDict
from flask import Flask, request, render_template
from PIL import Image, ImageFile

from analyzer import analyze

ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

# Safer PIL handling
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 25_000_000  # prevents extreme decompression bombs

app = Flask(__name__)

# ----------------------------
# Simple LRU cache for demo speed
# ----------------------------
CACHE_MAX = 64
_cache = OrderedDict()

def _cache_get(key):
    if key not in _cache:
        return None
    _cache.move_to_end(key)
    return _cache[key]

def _cache_set(key, value):
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    overlay_data = None
    result = None

    if request.method == "POST":
        file = request.files.get("image")
        shoe_side = (request.form.get("shoe_side") or "unknown").lower().strip()

        if shoe_side not in {"left", "right", "unknown"}:
            shoe_side = "unknown"

        if not file or file.mimetype not in ALLOWED:
            error = "Please upload a PNG/JPEG/WEBP image."
        else:
            # Read bytes once (enables hashing + size check + stable decoding)
            raw = file.read()
            size = len(raw)

            if size > MAX_BYTES:
                error = "Image too large (>10MB)."
            elif size == 0:
                error = "Empty upload. Please try again."
            else:
                # Cache key: image hash + shoe_side (since labeling differs)
                h = hashlib.sha256(raw).hexdigest()
                cache_key = f"{h}:{shoe_side}"

                cached = _cache_get(cache_key)
                if cached is not None:
                    overlay_data, result = cached
                else:
                    try:
                        img = Image.open(io.BytesIO(raw))
                        overlay_data, result = analyze(img, shoe_side=shoe_side)
                        _cache_set(cache_key, (overlay_data, result))
                    except Exception as e:
                        error = f"Failed to process image: {e}"

    return render_template(
        "index.html",
        error=error,
        overlay_data=overlay_data,
        result=result
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
