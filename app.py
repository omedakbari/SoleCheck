# app.py
import os
from flask import Flask, request, render_template
from PIL import Image, ImageFile
from analyzer import analyze

ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

# Helps avoid failures on slightly corrupted uploads
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

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
        if not file or file.mimetype not in ALLOWED:
            error = "Please upload a PNG/JPEG/WEBP image."
        else:
            file.stream.seek(0, 2)
            size = file.stream.tell()
            file.stream.seek(0)

            if size > MAX_BYTES:
                error = "Image too large (>10MB)."
            else:
                try:
                    image = Image.open(file.stream)
                    overlay_data, result = analyze(image)
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
