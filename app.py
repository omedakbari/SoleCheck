import os
import io
import json
import csv
import time
import hashlib
import zipfile
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from flask import Flask, request, render_template, send_file, abort
from PIL import Image, ImageFile

from analyzer import analyze_single, analyze_pair

ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 25_000_000

app = Flask(__name__)

# ----------------------------
# LRU cache: stores results for fast repeat demos + downloads
# ----------------------------
CACHE_MAX = 96
_cache = OrderedDict()  # key -> dict payload

def _cache_get(key: str):
    if key not in _cache:
        return None
    _cache.move_to_end(key)
    return _cache[key]

def _cache_set(key: str, value):
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)


# ----------------------------
# Logging (privacy-friendly: no images stored)
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_PATH = os.path.join(DATA_DIR, "requests.csv")

def _ensure_log_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc",
                "mode",
                "activity",
                "weekly_miles",
                "surface",
                "left_hash",
                "right_hash",
                "pattern_left",
                "pattern_right",
                "pair_ok",
                "pair_asymmetry",
            ])

def _log_request(row: list):
    try:
        _ensure_log_header()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass


# ----------------------------
# Sample images support
# Put sample images under: static/samples/
#   static/samples/sample_single.jpg
#   static/samples/sample_left.jpg
#   static/samples/sample_right.jpg
# ----------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")

def _load_sample(name: str) -> Optional[Image.Image]:
    path = os.path.join(SAMPLES_DIR, name)
    if not os.path.exists(path):
        return None
    return Image.open(path)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_upload(file_storage) -> Tuple[Optional[bytes], Optional[str]]:
    if not file_storage:
        return None, "No file uploaded."
    if file_storage.mimetype not in ALLOWED:
        return None, "Please upload a PNG/JPEG/WEBP image."
    raw = file_storage.read()
    if not raw:
        return None, "Empty upload. Please try again."
    if len(raw) > MAX_BYTES:
        return None, "Image too large (>10MB)."
    return raw, None


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def home():
    return render_template("index.html", error=None, view=None)


@app.get("/sample")
def sample():
    """
    Single-shoe sample demo.
    Requires: static/samples/sample_single.jpg
    """
    activity = request.args.get("activity", "walking")
    weekly_miles = request.args.get("weekly_miles", "unknown")
    surface = request.args.get("surface", "mixed")

    img = _load_sample("sample_single.jpg")
    if img is None:
        return render_template("index.html", error="Sample not found. Add static/samples/sample_single.jpg", view=None)

    overlay_png, result = analyze_single(img, shoe_side="unknown", activity=activity, weekly_miles=weekly_miles, surface=surface)

    rid = f"sample-single:{int(time.time())}"
    payload = {
        "mode": "single",
        "result": result,
        "overlay_left_png": overlay_png,
        "overlay_right_png": None,
        "created": datetime.utcnow().isoformat() + "Z",
    }
    _cache_set(rid, payload)

    view = _build_view_from_cache(rid)
    return render_template("index.html", error=None, view=view)


@app.get("/sample_pair")
def sample_pair():
    """
    Two-shoe sample demo.
    Requires:
      static/samples/sample_left.jpg
      static/samples/sample_right.jpg
    """
    activity = request.args.get("activity", "walking")
    weekly_miles = request.args.get("weekly_miles", "unknown")
    surface = request.args.get("surface", "mixed")

    L = _load_sample("sample_left.jpg")
    R = _load_sample("sample_right.jpg")
    if L is None or R is None:
        return render_template(
            "index.html",
            error="Sample pair not found. Add static/samples/sample_left.jpg and static/samples/sample_right.jpg",
            view=None
        )

    out = analyze_pair(L, R, activity=activity, weekly_miles=weekly_miles, surface=surface)

    rid = f"sample-pair:{int(time.time())}"
    payload = {
        "mode": "pair",
        "pair": out["pair"],
        "overlay_left_png": out["left_overlay_png"],
        "overlay_right_png": out["right_overlay_png"],
        "created": datetime.utcnow().isoformat() + "Z",
    }
    _cache_set(rid, payload)

    view = _build_view_from_cache(rid)
    return render_template("index.html", error=None, view=view)


@app.post("/analyze")
def analyze():
    mode = (request.form.get("mode") or "single").lower().strip()
    activity = (request.form.get("activity") or "walking").strip()
    weekly_miles = (request.form.get("weekly_miles") or "unknown").strip()
    surface = (request.form.get("surface") or "mixed").strip()

    if mode not in {"single", "pair"}:
        mode = "single"

    if mode == "single":
        file1 = request.files.get("image1")
        shoe_side = (request.form.get("shoe_side") or "unknown").lower().strip()
        if shoe_side not in {"left", "right", "unknown"}:
            shoe_side = "unknown"

        raw, err = _read_upload(file1)
        if err:
            return render_template("index.html", error=err, view=None)

        h = _sha(raw)
        rid = f"{h}:{mode}:{shoe_side}:{activity}:{weekly_miles}:{surface}"

        cached = _cache_get(rid)
        if cached is None:
            try:
                img = Image.open(io.BytesIO(raw))
                overlay_png, result = analyze_single(
                    img,
                    shoe_side=shoe_side,
                    activity=activity,
                    weekly_miles=weekly_miles,
                    surface=surface,
                )
                payload = {
                    "mode": "single",
                    "result": result,
                    "overlay_left_png": overlay_png,
                    "overlay_right_png": None,
                    "created": datetime.utcnow().isoformat() + "Z",
                    "hash_left": h,
                    "hash_right": "",
                }
                _cache_set(rid, payload)
            except Exception as e:
                return render_template("index.html", error=f"Failed to process image: {e}", view=None)

        view = _build_view_from_cache(rid)

        res = view["single"]
        _log_request([
            datetime.utcnow().isoformat() + "Z",
            "single",
            activity,
            weekly_miles,
            surface,
            h,
            "",
            res.get("pattern", ""),
            "",
            res.get("ok", False),
            "",
        ])
        return render_template("index.html", error=None, view=view)

    # pair
    fileL = request.files.get("image_left")
    fileR = request.files.get("image_right")

    rawL, errL = _read_upload(fileL)
    if errL:
        return render_template("index.html", error=f"Left shoe: {errL}", view=None)
    rawR, errR = _read_upload(fileR)
    if errR:
        return render_template("index.html", error=f"Right shoe: {errR}", view=None)

    hL = _sha(rawL)
    hR = _sha(rawR)
    rid = f"{hL}:{hR}:{mode}:{activity}:{weekly_miles}:{surface}"

    cached = _cache_get(rid)
    if cached is None:
        try:
            imgL = Image.open(io.BytesIO(rawL))
            imgR = Image.open(io.BytesIO(rawR))
            out = analyze_pair(imgL, imgR, activity=activity, weekly_miles=weekly_miles, surface=surface)
            payload = {
                "mode": "pair",
                "pair": out["pair"],
                "overlay_left_png": out["left_overlay_png"],
                "overlay_right_png": out["right_overlay_png"],
                "created": datetime.utcnow().isoformat() + "Z",
                "hash_left": hL,
                "hash_right": hR,
            }
            _cache_set(rid, payload)
        except Exception as e:
            return render_template("index.html", error=f"Failed to process images: {e}", view=None)

    view = _build_view_from_cache(rid)

    pair = view["pair"]
    pair_metrics = pair.get("pair_metrics", {}) or {}
    _log_request([
        datetime.utcnow().isoformat() + "Z",
        "pair",
        activity,
        weekly_miles,
        surface,
        hL,
        hR,
        pair.get("left", {}).get("pattern", ""),
        pair.get("right", {}).get("pattern", ""),
        pair.get("ok", False),
        pair_metrics.get("asymmetry_score_0_1", ""),
    ])
    return render_template("index.html", error=None, view=view)


def _b64_png(png_bytes: Optional[bytes]) -> Optional[str]:
    if not png_bytes:
        return None
    import base64
    return base64.b64encode(png_bytes).decode("ascii")


def _build_view_from_cache(rid: str) -> Dict[str, Any]:
    payload = _cache_get(rid)
    if payload is None:
        abort(404)

    view = {
        "rid": rid,
        "mode": payload.get("mode"),
        "created": payload.get("created"),
        "download": {
            "overlay_left": f"/download/overlay_left/{rid}.png",
            "overlay_right": f"/download/overlay_right/{rid}.png",
            "result_json": f"/download/result/{rid}.json",
            "bundle_zip": f"/download/bundle/{rid}.zip",
        }
    }

    if payload["mode"] == "single":
        view["single"] = payload["result"]
        view["overlay_left_b64"] = _b64_png(payload.get("overlay_left_png"))
        view["overlay_right_b64"] = None
    else:
        view["pair"] = payload["pair"]
        view["overlay_left_b64"] = _b64_png(payload.get("overlay_left_png"))
        view["overlay_right_b64"] = _b64_png(payload.get("overlay_right_png"))

    return view


@app.get("/download/result/<path:rid>.json")
def download_result(rid: str):
    payload = _cache_get(rid)
    if payload is None:
        abort(404)
    out: Dict[str, Any] = {
        "rid": rid,
        "created": payload.get("created"),
        "mode": payload.get("mode"),
    }
    if payload["mode"] == "single":
        out["result"] = payload.get("result")
    else:
        out["pair"] = payload.get("pair")

    data = json.dumps(out, indent=2).encode("utf-8")
    return send_file(io.BytesIO(data), mimetype="application/json", as_attachment=True, download_name="solecheck_result.json")


@app.get("/download/overlay_left/<path:rid>.png")
def download_overlay_left(rid: str):
    payload = _cache_get(rid)
    if payload is None:
        abort(404)
    png = payload.get("overlay_left_png")
    if not png:
        abort(404)
    return send_file(io.BytesIO(png), mimetype="image/png", as_attachment=True, download_name="solecheck_left_overlay.png")


@app.get("/download/overlay_right/<path:rid>.png")
def download_overlay_right(rid: str):
    payload = _cache_get(rid)
    if payload is None:
        abort(404)
    png = payload.get("overlay_right_png")
    if not png:
        abort(404)
    return send_file(io.BytesIO(png), mimetype="image/png", as_attachment=True, download_name="solecheck_right_overlay.png")


@app.get("/download/bundle/<path:rid>.zip")
def download_bundle(rid: str):
    payload = _cache_get(rid)
    if payload is None:
        abort(404)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        out: Dict[str, Any] = {
            "rid": rid,
            "created": payload.get("created"),
            "mode": payload.get("mode"),
        }
        if payload["mode"] == "single":
            out["result"] = payload.get("result")
        else:
            out["pair"] = payload.get("pair")
        z.writestr("result.json", json.dumps(out, indent=2))

        if payload.get("overlay_left_png"):
            z.writestr("overlay_left.png", payload["overlay_left_png"])
        if payload.get("overlay_right_png"):
            z.writestr("overlay_right.png", payload["overlay_right_png"])

    mem.seek(0)
    return send_file(mem, mimetype="application/zip", as_attachment=True, download_name="solecheck_bundle.zip")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
