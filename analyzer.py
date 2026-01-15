# analyzer.py
from PIL import Image
import numpy as np
import cv2
import base64

MAX_SIDE = 1100  # speed/latency knob

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

    # Ensure the sole is foreground (white). If the thresholded image is mostly white, invert.
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
    mask = _sole_mask(gray)

    h, w = gray.shape[:2]
    mask_area = float(cv2.countNonZero(mask))
    img_area = float(h * w)
    area_frac = (mask_area / img_area) if img_area else 0.0

    # Confidence heuristic: mask present + decent size
    # ramps up from 5% to 35% of image area
    confidence = 0.0
    if area_frac > 0:
        confidence = max(0.0, min(1.0, (area_frac - 0.05) / (0.35 - 0.05)))

    # If mask too small, return early guidance (better UX + avoids generic output)
    if confidence < 0.15:
        result = {
            "pattern": "Unable to confidently detect sole",
            "confidence": round(confidence, 2),
            "summary": "Try a clearer outsole photo: fill the frame, good lighting, flat angle.",
            "recommendation": "Retake photo with the sole centered and closer.",
            "shoes": "—",
            "metrics": {"mask_area_frac": round(area_frac, 3)},
        }
        return None, result

    # Region splits (on height)
    f_end = int(0.30 * h)
    m_end = int(0.70 * h)

    # Use texture proxy instead of brightness:
    # Wear tends to reduce texture/contrast. Laplacian magnitude is a cheap proxy.
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    def masked_mean(arr, m):
        # cv2.mean returns (mean,0,0,0) for single-channel
        return cv2.mean(arr, mask=m)[0]

    fore = masked_mean(lap[:f_end, :], mask[:f_end, :])
    mid  = masked_mean(lap[f_end:m_end, :], mask[f_end:m_end, :])
    heel = masked_mean(lap[m_end:, :], mask[m_end:, :])

    left  = masked_mean(lap[:, :w//2], mask[:, :w//2])
    right = masked_mean(lap[:, w//2:], mask[:, w//2:])

    # Lower texture => smoother => more wear (heuristic)
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

    # Overlay: edge heatmap + sole contour for explainability
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
            "heel_texture": round(heel, 1),
            "fore_texture": round(fore, 1),
            "mid_texture": round(mid, 1),
            "left_texture": round(left, 1),
            "right_texture": round(right, 1),
            "mask_area_frac": round(area_frac, 3),
        },
    }

    return overlay_b64, result
