from flask import Flask, request, render_template
from PIL import Image
from analyzer import analyze

ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None; overlay_data = None; analysis_text = ""
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.mimetype not in ALLOWED:
            error = "Please upload a PNG/JPEG/WEBP image."
        else:
            file.stream.seek(0, 2); size = file.stream.tell(); file.stream.seek(0)
            if size > MAX_BYTES:
                error = "Image too large (>10MB)."
            else:
                try:
                    image = Image.open(file.stream)
                    overlay_data, analysis_text = analyze(image)
                except Exception as e:
                    error = f"Failed to process image: {e}"
    return render_template("index.html", error=error, overlay_data=overlay_data, analysis_text=analysis_text)

if __name__ == "__main__":
    # Bind only to loopback, port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
