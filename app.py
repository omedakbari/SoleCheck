import os
import io
import json
import csv
import base64
import time
import hashlib
import zipfile
import urllib.request
import urllib.error
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

from flask import Flask, request, render_template, send_file, abort
from PIL import Image, ImageFile

from analyzer import analyze_single, analyze_pair

ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 25_000_000

app = Flask(__name__)

@app.context_processor
def inject_globals():
    return {"chat_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))}

# ----------------------------
# LRU cache
# ----------------------------
CACHE_MAX = 96
_cache = OrderedDict()

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
# Logging (privacy-friendly)
# ----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_PATH = os.path.join(DATA_DIR, "requests.csv")

def _ensure_log_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_utc", "mode", "activity", "weekly_miles", "surface",
                "left_hash", "right_hash", "pattern_left", "pattern_right",
                "pair_ok", "pair_asymmetry",
            ])

def _log_request(row: list):
    try:
        _ensure_log_header()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass


# ----------------------------
# Sample images
# ----------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")

def _load_sample(name: str) -> Optional[Image.Image]:
    path = os.path.join(SAMPLES_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


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


# ----------------------------
# Routes
# ----------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def home():
    return render_template("index.html", error=None, view=None)


@app.get("/sample")
def sample():
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
        "created": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
    }
    _cache_set(rid, payload)

    view = _build_view_from_cache(rid)
    return render_template("index.html", error=None, view=view)


@app.get("/sample_pair")
def sample_pair():
    activity = request.args.get("activity", "walking")
    weekly_miles = request.args.get("weekly_miles", "unknown")
    surface = request.args.get("surface", "mixed")

    L = _load_sample("sample_left.jpg")
    R = _load_sample("sample_right.jpg")
    if L is None or R is None:
        return render_template(
            "index.html",
            error="Sample pair not found. Add static/samples/sample_left.jpg and static/samples/sample_right.jpg",
            view=None,
        )

    out = analyze_pair(L, R, activity=activity, weekly_miles=weekly_miles, surface=surface)

    rid = f"sample-pair:{int(time.time())}"
    payload = {
        "mode": "pair",
        "pair": out["pair"],
        "overlay_left_png": out["left_overlay_png"],
        "overlay_right_png": out["right_overlay_png"],
        "created": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
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

        if _cache_get(rid) is None:
            try:
                img = Image.open(io.BytesIO(raw))
                overlay_png, result = analyze_single(img, shoe_side=shoe_side, activity=activity, weekly_miles=weekly_miles, surface=surface)
                _cache_set(rid, {
                    "mode": "single",
                    "result": result,
                    "overlay_left_png": overlay_png,
                    "overlay_right_png": None,
                    "created": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
                    "hash_left": h,
                    "hash_right": "",
                })
            except Exception as e:
                return render_template("index.html", error=f"Failed to process image: {e}", view=None)

        view = _build_view_from_cache(rid)
        res = view["single"]
        _log_request([datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z', "single", activity, weekly_miles, surface, h, "", res.get("pattern", ""), "", res.get("ok", False), ""])
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

    hL, hR = _sha(rawL), _sha(rawR)
    rid = f"{hL}:{hR}:{mode}:{activity}:{weekly_miles}:{surface}"

    if _cache_get(rid) is None:
        try:
            imgL = Image.open(io.BytesIO(rawL))
            imgR = Image.open(io.BytesIO(rawR))
            out = analyze_pair(imgL, imgR, activity=activity, weekly_miles=weekly_miles, surface=surface)
            _cache_set(rid, {
                "mode": "pair",
                "pair": out["pair"],
                "overlay_left_png": out["left_overlay_png"],
                "overlay_right_png": out["right_overlay_png"],
                "created": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z',
                "hash_left": hL,
                "hash_right": hR,
            })
        except Exception as e:
            return render_template("index.html", error=f"Failed to process images: {e}", view=None)

    view = _build_view_from_cache(rid)
    pair = view["pair"]
    pair_metrics = pair.get("pair_metrics", {}) or {}
    _log_request([datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z', "pair", activity, weekly_miles, surface, hL, hR, pair.get("left", {}).get("pattern", ""), pair.get("right", {}).get("pattern", ""), pair.get("ok", False), pair_metrics.get("asymmetry_score_0_1", "")])
    return render_template("index.html", error=None, view=view)


# ----------------------------
# AI Chat endpoint
# ----------------------------

def _readable_context(context: dict) -> str:
    if not context:
        return "No analysis context available."

    lines = []
    if "pattern" in context:
        lines.append(f"Wear pattern: {context.get('pattern', 'unknown')}")
        lines.append(f"Confidence: {int(context.get('confidence', 0) * 100)}%")
        if context.get("summary"):
            lines.append(f"Observations: {context['summary']}")
        if context.get("recommendation"):
            lines.append(f"Recommendation: {context['recommendation']}")
        if context.get("shoes"):
            lines.append(f"Shoe suggestions: {context['shoes']}")
        m = context.get("metrics", {})
        if m:
            lines.append(f"Activity: {m.get('activity', 'unknown')}, surface: {m.get('surface', 'unknown')}, weekly miles: {m.get('weekly_miles', 'unknown')}")
            lines.append(f"Strike pattern: {m.get('strike_hint', 'unknown')}, heel/fore ratio: {m.get('heel_fore_wear_ratio', '?')}")
            if "medial_lateral_wear_ratio" in m:
                lines.append(f"Medial/lateral ratio: {m['medial_lateral_wear_ratio']}")
    else:
        for side in ("left", "right"):
            s = context.get(side, {})
            if s:
                lines.append(f"{side.capitalize()} shoe — pattern: {s.get('pattern', 'unknown')} ({int(s.get('confidence', 0)*100)}% confidence)")
                if s.get("summary"):
                    lines.append(f"  Observations: {s['summary']}")
                if s.get("recommendation"):
                    lines.append(f"  Recommendation: {s['recommendation']}")
        if context.get("pair_summary"):
            lines.append(f"Pair comparison: {context['pair_summary']}")
        pm = context.get("pair_metrics", {})
        if pm.get("asymmetry_score_0_1") is not None:
            lines.append(f"Asymmetry score: {int(float(pm['asymmetry_score_0_1']) * 100)}%")

    return "\n".join(lines)


@app.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    context = body.get("context") or {}
    history = body.get("history") or []

    if not message:
        return {"error": "No message provided."}, 400
    if len(message) > 500:
        return {"error": "Message too long (max 500 characters)."}, 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "AI chat is not enabled on this deployment."}, 503

    ctx_str = _readable_context(context)

    system = (
        "You are SoleCheck AI. You know biomechanics, gait, and footwear well.\n"
        "The user just scanned their shoe sole. Here are the results:\n\n"
        f"{ctx_str}\n\n"
        "Answer in 2 to 3 sentences. Be direct. Use their specific results. "
        "Skip filler phrases like 'great question' or 'it is important to note'. "
        "Speak like you are talking to a friend who wants a straight answer. "
        "Never use em dashes or semicolons. You are not a doctor."
    )

    claude_msgs = []
    for h in history[-8:]:
        r, c = h.get("role"), h.get("content", "")
        if r in ("user", "assistant") and c:
            claude_msgs.append({"role": r, "content": c})
    claude_msgs.append({"role": "user", "content": message})

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 350,
        "system": system,
        "messages": claude_msgs,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = (data.get("content") or [{}])[0].get("text", "No response.")
            return {"response": text}
    except urllib.error.HTTPError as e:
        return {"error": f"AI service error {e.code}."}, 502
    except Exception as e:
        return {"error": f"Request failed: {e}"}, 500


# ----------------------------
# Download helpers
# ----------------------------

def _b64_png(png_bytes: Optional[bytes]) -> Optional[str]:
    if not png_bytes:
        return None
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
    out: Dict[str, Any] = {"rid": rid, "created": payload.get("created"), "mode": payload.get("mode")}
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
        out: Dict[str, Any] = {"rid": rid, "created": payload.get("created"), "mode": payload.get("mode")}
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
