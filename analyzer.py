# analyzer.py  (COPY/PASTE WHOLE FILE)
# Adds a photo-quality gate: blur + lighting + "sole found" confidence.
# Keeps the same analyze() signature: returns (overlay_b64, result_dict)

from PIL import Image
import numpy as np
import cv2
import base64

MAX_SIDE = 1100  # speed/latency knob

# --- Quality gate thresholds (tweak if needed) ---
# Blur: variance of Laplacian. Lower = blurrier.
BLUR_VAR_MIN = 60.0

# Lighting: grayscale mean.
DARK_MEAN_MIN = 45.0
BRIGHT_MEAN_MAX = 210.0

# Sole detection confidence gate (based on mask area fraction)
SOLE_CONF_MIN = 0.15


def _resize_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _sole_mask(gray: np.ndarray) -> np.ndarray:
    # blur + Otsu + morph close + largest contour
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure the sole is foreground (white). If thresholded image is mostly white, invert.
    if float(np.mean(bw)) > 127.0:
        bw = cv2.bitwise_not(bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(gray, dtype=np.uint8)

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


def _quality_checks(gray: np.ndarray) -> dict:
    """
    Returns:
      dict with keys:
        ok (bool)
        reasons (list[str])
        metrics (dict)
    """
    reasons = []

    # Blur check (variance of Laplacian)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    if blur_var < BLUR_VAR_MIN:
        reasons.append("Image looks blurry (low detail).")

    # Lighting check
    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_MEAN_MIN:
        reasons.append("Image looks too dark.")
    elif mean_brightness > BRIGHT_MEAN_MAX:
        reasons.append("Image looks too bright / washed out.")

    ok = (len(reasons) == 0)
    return {
        "ok": ok,
        "reasons": reasons,
        "metrics": {
            "blur_var": round(blur_var, 2),
            "mean_brightness": round(mean_brightness, 2),
        }
    }


def analyze(image_pil: Image.Image):
    """
    Returns:
      overlay_b64 (str|None): base64 PNG overlay
      result (dict): structured analysis for template rendering
    """
    # Convert once -> OpenCV BGR
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _resize_bgr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ---- (1) Photo-quality gate: blur + lighting ----
    qc = _quality_checks(gray)
    if not qc["ok"]:
        result = {
            "pattern": "Retake photo",
            "confidence": 0.0,
            "summary": " ".join(qc["reasons"]),
            "recommendation": (
                "Try: bright even lighting, fill the frame with the sole, and hold steady."
            ),
            "shoes": "—",
            "metrics": qc["metrics"],
        }
        return None, result

    # ---- (2) Sole detection + confidence ----
    mask = _sole_mask(gray)

    h, w = gray.shape[:2]
    mask_area = float(cv2.countNonZero(mask))
    img_area = float(h * w)
    area_frac = (mask_area / img_area) if img_area else 0.0

    # Confidence heuristic: ramps up from 5% to 35% mask coverage
    confidence = max(0.0, min(1.0, (area_frac - 0.05) / (0.35 - 0.05)))

    if confidence < SOLE_CONF_MIN:
        result = {
            "pattern": "Retake photo",
            "confidence": round(confidence, 2),
            "summary": "Couldn’t confidently isolate the sole from the background.",
            "recommendation": (
                "Center the sole, move closer so it fills most of the frame, and use a plain background."
            ),
            "shoes": "—",
            "metrics": {
                **qc["metrics"],
                "mask_area_frac": round(area_frac, 3),
            },
        }
        return None, result

    # ---- Main analysis (same logic style, but based on texture proxy) ----
    f_end = int(0.30 * h)
    m_end = int(0.70 * h)

    # Texture proxy: wear tends to reduce texture
    lap_tex = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_tex = cv2.convertScaleAbs(lap_tex)

    def masked_mean(arr, m):
        return cv2.mean(arr, mask=m)[0]

    fore = masked_mean(lap_tex[:f_end, :], mask[:f_end, :])
    mid  = masked_mean(lap_tex[f_end:m_end, :], mask[f_end:m_end, :])
    heel = masked_mean(lap_tex[m_end:, :], mask[m_end:, :])

    left  = masked_mean(lap_tex[:, :w//2], mask[:, :w//2])
    right = masked_mean(lap_tex[:, w//2:], mask[:, w//2:])

    notes = []

    heel_fore = heel - fore
    side_diff = left - right

    if heel_fore < -2:
        notes.append("Forefoot region appears smoother (possible forefoot loading).")
    elif heel_fore > 2:
        notes.append("Heel region appears smoother (possible heel striking).")
    else:
        notes.append("Heel vs forefoot wear looks fairly balanced.")

    if side_diff < -2:
        pattern = "Likely overpronation tendency (medial wear)"
        notes.append("Inner half appears smoother than outer.")
        recommendation = "Consider stability support + foot/ankle strengthening."
        shoes = "Stability: Brooks Adrenaline, ASICS Kayano, HOKA Arahi"
    elif side_diff > 2:
        pattern = "Likely supination tendency (lateral wear)"
        notes.append("Outer half appears smoother than inner.")
        recommendation = "Prioritize cushioning + ankle mobility work."
        shoes = "Neutral cushioned: Brooks Glycerin, ASICS Cumulus, Nike Invincible"
    else:
        pattern = "Neutral or mixed wear"
        notes.append("Inner vs outer wear looks fairly balanced.")
        recommendation = "Rotate pairs; replace once tread flattens in key zones."
        shoes = "Neutral supportive shoes appropriate to activity"

    # ---- Explainable overlay ----
    edges = cv2.Canny(gray, 50, 150)
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.75, heatmap, 0.25, 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, [max(cnts, key=cv2.contourArea)], -1, (255, 255, 255), 2)

    ok, buf = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(buf).decode("ascii") if ok else None

    result = {
        "pattern": pattern,
        "confidence": round(confidence, 2),
        "summary": " | ".join(notes),
        "recommendation": recommendation,
        "shoes": shoes,
        "metrics": {
            **qc["metrics"],
            "heel_texture": round(heel, 1),
            "fore_texture": round(fore, 1),
            "mid_texture": round(mid, 1),
            "left_texture": round(left, 1),
            "right_texture": round(right, 1),
            "mask_area_frac": round(area_frac, 3),
        },
    }

    return overlay_b64, result
