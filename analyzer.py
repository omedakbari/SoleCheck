# analyzer.py

from PIL import Image
import numpy as np
import cv2
import base64

# ----------------------------
# Tunables
# ----------------------------
MAX_SIDE = 1100  # resize max dimension for speed

# Photo-quality gate thresholds
BLUR_VAR_MIN = 60.0          # lower = blurrier
DARK_MEAN_MIN = 45.0         # too dark if below
BRIGHT_MEAN_MAX = 210.0      # too bright if above

# Sole detection confidence threshold (based on mask area fraction)
SOLE_CONF_MIN = 0.15

# Wear scoring thresholds (tweak after testing)
SIDE_WEAR_THRESH = 0.12      # how strong medial/lateral imbalance must be
HEEL_FORE_THRESH = 0.10      # how strong heel/forefoot imbalance must be

EPS = 1e-6


def _resize_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _quality_checks(gray: np.ndarray) -> dict:
    """
    Returns:
      dict with keys: ok(bool), reasons(list[str]), metrics(dict)
    """
    reasons = []

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    if blur_var < BLUR_VAR_MIN:
        reasons.append("Image looks blurry (low detail).")

    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_MEAN_MIN:
        reasons.append("Image looks too dark.")
    elif mean_brightness > BRIGHT_MEAN_MAX:
        reasons.append("Image looks too bright / washed out.")

    return {
        "ok": (len(reasons) == 0),
        "reasons": reasons,
        "metrics": {
            "blur_var": round(blur_var, 2),
            "mean_brightness": round(mean_brightness, 2),
        }
    }


def _sole_mask(gray: np.ndarray) -> np.ndarray:
    """
    Simple robust-ish masking:
      blur -> Otsu -> invert if needed -> close -> largest contour filled
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If thresholded image is mostly white, invert (so foreground becomes white)
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


def _mask_confidence(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    mask_area = float(cv2.countNonZero(mask))
    img_area = float(h * w) if h and w else 0.0
    area_frac = (mask_area / img_area) if img_area else 0.0
    # Ramp confidence from 5% to 35% coverage
    conf = max(0.0, min(1.0, (area_frac - 0.05) / (0.35 - 0.05)))
    return conf


def _rotate_bound(img: np.ndarray, angle_deg: float, is_mask: bool = False) -> np.ndarray:
    """
    Rotate without cropping by expanding canvas.
    angle_deg: positive = counter-clockwise
    """
    h, w = img.shape[:2]
    cX, cY = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2.0) - cX
    M[1, 2] += (nH / 2.0) - cY

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (nW, nH), flags=interp, borderValue=0)


def _normalize_rotation(bgr: np.ndarray, mask: np.ndarray):
    """
    Align sole's long axis vertically using minAreaRect on the largest contour.
    Returns rotated (bgr, mask, applied_rotation_deg).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, mask, 0.0

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    rw, rh = rect[1]
    angle = rect[2]

    if rw <= 0 or rh <= 0:
        return bgr, mask, 0.0

    # OpenCV angle in [-90,0). Determine rotation to make long axis vertical.
    if rw < rh:
        rot = angle
    else:
        rot = angle + 90.0

    applied = -rot
    bgr_r = _rotate_bound(bgr, applied, is_mask=False)
    mask_r = _rotate_bound(mask, applied, is_mask=True)

    # Ensure portrait-ish orientation
    h, w = mask_r.shape[:2]
    if w > h:
        bgr_r = _rotate_bound(bgr_r, 90.0, is_mask=False)
        mask_r = _rotate_bound(mask_r, 90.0, is_mask=True)
        applied += 90.0

    return bgr_r, mask_r, float(applied)


def _masked_mean(arr: np.ndarray, m: np.ndarray) -> float:
    return float(cv2.mean(arr, mask=m)[0])


def _wear_map_from_texture(gray: np.ndarray, mask: np.ndarray):
    """
    Compute a wear heatmap from low texture areas:
      texture = |Laplacian|
      wear = inverse texture within mask
    Returns:
      lap_tex (uint8), wear_norm (uint8 0..255)
    """
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_tex = cv2.convertScaleAbs(lap)  # 0..255-ish

    # Invert texture -> higher means smoother (potential wear)
    inv = (255 - lap_tex).astype(np.uint8)

    # Normalize only inside mask for consistent color scaling
    masked_vals = inv[mask > 0]
    if masked_vals.size < 10:
        wear_norm = np.zeros_like(inv, dtype=np.uint8)
        return lap_tex, wear_norm

    p5 = int(np.percentile(masked_vals, 5))
    p95 = int(np.percentile(masked_vals, 95))
    if p95 <= p5:
        wear_norm = np.zeros_like(inv, dtype=np.uint8)
        return lap_tex, wear_norm

    wear = np.clip(inv.astype(np.int32) - p5, 0, (p95 - p5)).astype(np.uint8)
    wear_norm = (wear.astype(np.float32) / float(p95 - p5) * 255.0).astype(np.uint8)

    # Zero out outside mask
    wear_norm[mask == 0] = 0
    return lap_tex, wear_norm


def analyze(image_pil: Image.Image, shoe_side: str = "unknown"):
    """
    shoe_side: "left" | "right" | "unknown"
      Used to correctly label medial/lateral after rotation normalization.

    Returns:
      overlay_b64 (str|None)
      result (dict)
    """
    # Convert once -> OpenCV BGR
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _resize_bgr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # (1) Photo quality gate
    qc = _quality_checks(gray)
    if not qc["ok"]:
        result = {
            "pattern": "Retake photo",
            "confidence": 0.0,
            "summary": " ".join(qc["reasons"]),
            "recommendation": "Try: bright even lighting, fill the frame with the sole, and hold steady.",
            "shoes": "—",
            "metrics": qc["metrics"],
        }
        return None, result

    # (2) Sole detection gate
    mask = _sole_mask(gray)
    conf0 = _mask_confidence(mask)
    if conf0 < SOLE_CONF_MIN:
        result = {
            "pattern": "Retake photo",
            "confidence": round(conf0, 2),
            "summary": "Couldn’t confidently isolate the sole from the background.",
            "recommendation": "Center the sole, move closer so it fills most of the frame, and use a plain background.",
            "shoes": "—",
            "metrics": {**qc["metrics"], "mask_conf": round(conf0, 2)},
        }
        return None, result

    # (3) Rotation normalization
    bgr, mask, rot_deg = _normalize_rotation(bgr, mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    conf = _mask_confidence(mask)
    if conf < SOLE_CONF_MIN:
        # Rare, but if rotation broke the mask badly, ask for retake.
        result = {
            "pattern": "Retake photo",
            "confidence": round(conf, 2),
            "summary": "Sole was detected, but alignment was unstable.",
            "recommendation": "Try again with the sole centered and a plain background.",
            "shoes": "—",
            "metrics": {**qc["metrics"], "mask_conf": round(conf, 2), "rotation_deg": round(rot_deg, 2)},
        }
        return None, result

    # (4) Wear heatmap from texture (evidence)
    lap_tex, wear_norm = _wear_map_from_texture(gray, mask)

    # Region splits (toe top, heel bottom)
    h, w = gray.shape[:2]
    f_end = int(0.30 * h)
    m_end = int(0.70 * h)

    # Compute mean texture in regions (lower texture = smoother)
    fore_tex = _masked_mean(lap_tex[:f_end, :], mask[:f_end, :])
    mid_tex  = _masked_mean(lap_tex[f_end:m_end, :], mask[f_end:m_end, :])
    heel_tex = _masked_mean(lap_tex[m_end:, :], mask[m_end:, :])

    left_tex  = _masked_mean(lap_tex[:, :w//2], mask[:, :w//2])
    right_tex = _masked_mean(lap_tex[:, w//2:], mask[:, w//2:])

    # Convert texture -> wear score (higher wear score means smoother/more worn)
    def wear_score(tex: float) -> float:
        return 1.0 / (tex + 1.0)

    fore_w = wear_score(fore_tex)
    mid_w  = wear_score(mid_tex)
    heel_w = wear_score(heel_tex)
    left_w = wear_score(left_tex)
    right_w = wear_score(right_tex)

    # Heel vs forefoot signal
    heel_fore_ratio = (heel_w / (fore_w + EPS))
    # Side wear: map medial/lateral based on shoe side
    # After normalization: toe up, heel down.
    # Left shoe: medial is RIGHT half. Right shoe: medial is LEFT half.
    if shoe_side == "left":
        medial_w = right_w
        lateral_w = left_w
        medial_tex = right_tex
        lateral_tex = left_tex
    elif shoe_side == "right":
        medial_w = left_w
        lateral_w = right_w
        medial_tex = left_tex
        lateral_tex = right_tex
    else:
        medial_w = None
        lateral_w = None
        medial_tex = None
        lateral_tex = None

    notes = []

    # Heel/forefoot statement
    if heel_fore_ratio > (1.0 + HEEL_FORE_THRESH):
        notes.append("Heel region appears smoother than forefoot (possible heel striking).")
        strike = "heel"
    elif heel_fore_ratio < (1.0 - HEEL_FORE_THRESH):
        notes.append("Forefoot region appears smoother than heel (possible forefoot loading).")
        strike = "forefoot"
    else:
        notes.append("Heel vs forefoot wear looks fairly balanced.")
        strike = "balanced"

    # Medial/lateral statement
    gait_label = "Neutral or mixed wear"
    recommendation = "Rotate pairs; replace once tread flattens in key zones."
    shoes = "Neutral supportive shoes appropriate to activity"

    if shoe_side in ("left", "right"):
        ml_ratio = medial_w / (lateral_w + EPS)
        if ml_ratio > (1.0 + SIDE_WEAR_THRESH):
            gait_label = "Likely overpronation tendency (medial wear)"
            notes.append("Medial side appears smoother than lateral.")
            recommendation = "Consider stability support + foot/ankle strengthening."
            shoes = "Stability: Brooks Adrenaline, ASICS Kayano, HOKA Arahi"
        elif ml_ratio < (1.0 - SIDE_WEAR_THRESH):
            gait_label = "Likely supination tendency (lateral wear)"
            notes.append("Lateral side appears smoother than medial.")
            recommendation = "Prioritize cushioning + ankle mobility work."
            shoes = "Neutral cushioned: Brooks Glycerin, ASICS Cumulus, Nike Invincible"
        else:
            notes.append("Medial vs lateral wear looks fairly balanced.")
    else:
        notes.append("Tip: select Left/Right shoe for more accurate medial/lateral labeling.")
        # fallback based on side imbalance without claiming medial/lateral
        side_ratio = left_w / (right_w + EPS)
        if side_ratio > (1.0 + SIDE_WEAR_THRESH):
            gait_label = "One-side wear detected (select shoe side for medial/lateral)"
        elif side_ratio < (1.0 - SIDE_WEAR_THRESH):
            gait_label = "One-side wear detected (select shoe side for medial/lateral)"
        else:
            gait_label = "Neutral or mixed wear"

    # (5) Build evidence overlay: color wear_norm + contour
    wear_color = cv2.applyColorMap(wear_norm, cv2.COLORMAP_JET)
    # Only show wear colors inside mask; keep outside mostly original
    overlay = cv2.addWeighted(bgr, 0.75, wear_color, 0.25, 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, [max(cnts, key=cv2.contourArea)], -1, (255, 255, 255), 2)

    ok, buf = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(buf).decode("ascii") if ok else None

    # Make output metric-driven
    metrics = {
        **qc["metrics"],
        "rotation_deg": round(rot_deg, 2),
        "mask_conf": round(conf, 2),

        "heel_texture": round(heel_tex, 1),
        "mid_texture": round(mid_tex, 1),
        "fore_texture": round(fore_tex, 1),
        "left_texture": round(left_tex, 1),
        "right_texture": round(right_tex, 1),

        "heel_fore_wear_ratio": round(heel_fore_ratio, 3),
        "strike_hint": strike,
    }

    if shoe_side in ("left", "right"):
        ml_ratio = (medial_w / (lateral_w + EPS))
        metrics.update({
            "shoe_side": shoe_side,
            "medial_texture": round(medial_tex, 1),
            "lateral_texture": round(lateral_tex, 1),
            "medial_lateral_wear_ratio": round(ml_ratio, 3),
        })
    else:
        metrics.update({"shoe_side": "unknown"})

    result = {
        "pattern": gait_label,
        "confidence": round(conf, 2),
        "summary": " | ".join(notes),
        "recommendation": recommendation,
        "shoes": shoes,
        "metrics": metrics,
    }

    return overlay_b64, result
