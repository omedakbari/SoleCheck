from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import cv2

# ----------------------------
# Tunables
# ----------------------------
MAX_SIDE = 1100

# Photo quality thresholds
BLUR_VAR_MIN = 60.0
DARK_MEAN_MIN = 45.0
BRIGHT_MEAN_MAX = 210.0

# Mask confidence threshold
SOLE_CONF_MIN = 0.15

# Wear classification thresholds
SIDE_WEAR_THRESH = 0.12
HEEL_FORE_THRESH = 0.10

EPS = 1e-6


# ----------------------------
# Helpers
# ----------------------------
def _resize_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / m
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _quality_checks(gray: np.ndarray) -> Dict[str, Any]:
    reasons: List[str] = []

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
        },
    }


def _mask_confidence(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    mask_area = float(cv2.countNonZero(mask))
    img_area = float(h * w)
    area_frac = mask_area / img_area
    # ramp up from 5% to 35%
    conf = max(0.0, min(1.0, (area_frac - 0.05) / (0.35 - 0.05)))
    return conf


def _largest_contour_mask(bw: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(bw, dtype=np.uint8)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return mask


def _initial_otsu_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground is white
    if float(np.mean(bw)) > 127.0:
        bw = cv2.bitwise_not(bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_contour_mask(bw)


def _grabcut_refine(bgr: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """
    GrabCut refinement using a rectangle derived from the seed mask's bounding box.
    Returns a binary mask (uint8 0/255).
    """
    h, w = seed_mask.shape[:2]
    if h == 0 or w == 0:
        return seed_mask

    ys, xs = np.where(seed_mask > 0)
    if ys.size < 50:
        # fallback: center rectangle
        x0 = int(0.10 * w)
        y0 = int(0.10 * h)
        x1 = int(0.90 * w)
        y1 = int(0.90 * h)
    else:
        x0 = max(0, int(xs.min()) - 10)
        y0 = max(0, int(ys.min()) - 10)
        x1 = min(w - 1, int(xs.max()) + 10)
        y1 = min(h - 1, int(ys.max()) + 10)

    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    gc_mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        # probable/definite foreground
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = _largest_contour_mask(fg)
        return fg
    except Exception:
        return seed_mask


def _sole_mask(bgr: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Mask pipeline:
      Otsu seed -> if low confidence, refine with GrabCut -> largest contour
    """
    seed = _initial_otsu_mask(gray)
    conf_seed = _mask_confidence(seed)

    refined = seed
    used_grabcut = False
    if conf_seed < 0.25:
        refined = _grabcut_refine(bgr, seed)
        used_grabcut = True

    conf_final = _mask_confidence(refined)
    meta = {
        "mask_conf_seed": round(conf_seed, 3),
        "mask_conf_final": round(conf_final, 3),
        "used_grabcut": used_grabcut,
    }
    return refined, meta


def _rotate_bound(img: np.ndarray, angle_deg: float, is_mask: bool = False) -> np.ndarray:
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


def _normalize_rotation(bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, mask, 0.0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    rw, rh = rect[1]
    angle = rect[2]
    if rw <= 0 or rh <= 0:
        return bgr, mask, 0.0

    if rw < rh:
        rot = angle
    else:
        rot = angle + 90.0

    applied = -rot
    bgr_r = _rotate_bound(bgr, applied, is_mask=False)
    mask_r = _rotate_bound(mask, applied, is_mask=True)

    # Keep portrait-ish
    h, w = mask_r.shape[:2]
    if w > h:
        bgr_r = _rotate_bound(bgr_r, 90.0, is_mask=False)
        mask_r = _rotate_bound(mask_r, 90.0, is_mask=True)
        applied += 90.0

    return bgr_r, mask_r, float(applied)


def _masked_mean(arr: np.ndarray, m: np.ndarray) -> float:
    return float(cv2.mean(arr, mask=m)[0])


def _normalize_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask > 0]
    if vals.size < 10:
        out = np.zeros_like(arr, dtype=np.uint8)
        out[mask > 0] = 0
        return out
    p5 = float(np.percentile(vals, 5))
    p95 = float(np.percentile(vals, 95))
    if p95 <= p5 + 1e-3:
        out = np.zeros_like(arr, dtype=np.uint8)
        return out
    scaled = (np.clip(arr.astype(np.float32), p5, p95) - p5) / (p95 - p5) * 255.0
    scaled = scaled.astype(np.uint8)
    scaled[mask == 0] = 0
    return scaled


def _wear_evidence(gray: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evidence signals:
      - Texture: |Laplacian| (low texture => worn)
      - Gradient magnitude (low gradient => smoother => worn)
    Combine both into a wear score map.
    Returns:
      lap_tex (uint8), grad_mag (uint8), wear_norm (uint8)
    """
    # Texture
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap_tex = cv2.convertScaleAbs(lap)  # high => textured
    inv_tex = (255 - lap_tex).astype(np.uint8)  # high => smooth

    # Gradient
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    grad = np.clip(grad, 0, 255).astype(np.uint8)
    inv_grad = (255 - grad).astype(np.uint8)

    inv_tex_n = _normalize_in_mask(inv_tex, mask)
    inv_grad_n = _normalize_in_mask(inv_grad, mask)

    # Combine (weights can be tuned)
    wear = (0.60 * inv_tex_n.astype(np.float32) + 0.40 * inv_grad_n.astype(np.float32)).astype(np.uint8)
    wear[mask == 0] = 0

    return lap_tex, grad, wear


def _pick_shoes(activity: str, pronation_class: str) -> str:
    """
    Simple recommendation mapping (customize later).
    pronation_class: "overpronation" | "supination" | "neutral"
    """
    activity = (activity or "walking").lower()
    if activity not in {"running", "walking", "basketball", "training"}:
        activity = "walking"

    if activity == "running":
        if pronation_class == "overpronation":
            return "Stability runners: Brooks Adrenaline, ASICS Kayano, HOKA Arahi"
        if pronation_class == "supination":
            return "Cushioned neutral: Brooks Glycerin, ASICS Cumulus, Nike Invincible"
        return "Neutral daily trainers: Nike Pegasus, ASICS Novablast, Saucony Ride"
    if activity == "basketball":
        if pronation_class == "overpronation":
            return "Supportive hoops: Nike LeBron line, KD line (snug fit), orthotic-friendly models"
        if pronation_class == "supination":
            return "Cushioned hoops: Nike GT Jump, Jordan Zion line, plus ankle mobility work"
        return "Balanced hoops: Nike GT Cut, Kobe-style low-tops (if comfortable), good torsional support"
    if activity == "training":
        if pronation_class == "overpronation":
            return "Stable trainers: Nike Metcon, Reebok Nano, plus supportive insole if needed"
        if pronation_class == "supination":
            return "Cushioned trainers + mobility: consider softer midsoles and ankle work"
        return "General trainers: Nike Metcon / Reebok Nano / Adidas Dropset"
    # walking
    if pronation_class == "overpronation":
        return "Stability walkers: ASICS GT-2000, Brooks Adrenaline (walk use), supportive insoles"
    if pronation_class == "supination":
        return "Cushioned walkers: HOKA Bondi, Brooks Glycerin, ASICS Cumulus"
    return "Comfort neutral walkers: Brooks Ghost, Nike Pegasus, ASICS Cumulus"


def analyze_single(
    image_pil,
    shoe_side: str = "unknown",
    activity: str = "walking",
    weekly_miles: str = "unknown",
    surface: str = "mixed",
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Returns:
      overlay_png_bytes (bytes|None)
      result (dict)
    """
    # Convert to OpenCV BGR
    rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _resize_bgr(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Photo quality gate
    qc = _quality_checks(gray)
    if not qc["ok"]:
        return None, {
            "ok": False,
            "pattern": "Retake photo",
            "confidence": 0.0,
            "summary": " ".join(qc["reasons"]),
            "recommendation": "Try: bright even lighting, fill the frame with the sole, and hold steady.",
            "shoes": "—",
            "metrics": qc["metrics"],
        }

    # Mask (Otsu + GrabCut fallback)
    mask, mask_meta = _sole_mask(bgr, gray)
    conf0 = _mask_confidence(mask)
    if conf0 < SOLE_CONF_MIN:
        return None, {
            "ok": False,
            "pattern": "Retake photo",
            "confidence": round(conf0, 2),
            "summary": "Couldn’t confidently isolate the sole from the background.",
            "recommendation": "Center the sole, move closer so it fills most of the frame, and use a plain background.",
            "shoes": "—",
            "metrics": {**qc["metrics"], **mask_meta, "mask_conf": round(conf0, 3)},
        }

    # Normalize rotation
    bgr, mask, rot_deg = _normalize_rotation(bgr, mask)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    conf = _mask_confidence(mask)
    if conf < SOLE_CONF_MIN:
        return None, {
            "ok": False,
            "pattern": "Retake photo",
            "confidence": round(conf, 2),
            "summary": "Sole was detected, but alignment was unstable.",
            "recommendation": "Try again with the sole centered and a plain background.",
            "shoes": "—",
            "metrics": {**qc["metrics"], **mask_meta, "mask_conf": round(conf, 3), "rotation_deg": round(rot_deg, 2)},
        }

    # Wear evidence
    lap_tex, grad_mag, wear_map = _wear_evidence(gray, mask)

    # Region splits (toe top, heel bottom)
    h, w = gray.shape[:2]
    f_end = int(0.30 * h)
    m_end = int(0.70 * h)

    # Region texture means (lower => smoother)
    fore_tex = _masked_mean(lap_tex[:f_end, :], mask[:f_end, :])
    mid_tex  = _masked_mean(lap_tex[f_end:m_end, :], mask[f_end:m_end, :])
    heel_tex = _masked_mean(lap_tex[m_end:, :], mask[m_end:, :])

    left_tex  = _masked_mean(lap_tex[:, :w//2], mask[:, :w//2])
    right_tex = _masked_mean(lap_tex[:, w//2:], mask[:, w//2:])

    # Convert texture to wear score
    def wear_score(tex: float) -> float:
        return 1.0 / (tex + 1.0)

    fore_w = wear_score(fore_tex)
    heel_w = wear_score(heel_tex)
    left_w = wear_score(left_tex)
    right_w = wear_score(right_tex)

    heel_fore_ratio = heel_w / (fore_w + EPS)

    # Medial/lateral mapping depends on shoe side
    shoe_side = (shoe_side or "unknown").lower()
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
        medial_w = lateral_w = None
        medial_tex = lateral_tex = None

    notes = []
    if heel_fore_ratio > (1.0 + HEEL_FORE_THRESH):
        notes.append("Heel region appears smoother than forefoot (possible heel striking).")
        strike_hint = "heel"
    elif heel_fore_ratio < (1.0 - HEEL_FORE_THRESH):
        notes.append("Forefoot region appears smoother than heel (possible forefoot loading).")
        strike_hint = "forefoot"
    else:
        notes.append("Heel vs forefoot wear looks fairly balanced.")
        strike_hint = "balanced"

    pronation_class = "neutral"
    pattern = "Neutral or mixed wear"
    recommendation = "Rotate pairs; replace once tread flattens in key zones."
    if weekly_miles and weekly_miles != "unknown":
        recommendation = "Rotate pairs; track wear; replace once tread flattens in key zones."

    if shoe_side in ("left", "right"):
        ml_ratio = medial_w / (lateral_w + EPS)
        if ml_ratio > (1.0 + SIDE_WEAR_THRESH):
            pronation_class = "overpronation"
            pattern = "Likely overpronation tendency (medial wear)"
            notes.append("Medial side appears smoother than lateral.")
            recommendation = "Consider stability support + foot/ankle strengthening."
        elif ml_ratio < (1.0 - SIDE_WEAR_THRESH):
            pronation_class = "supination"
            pattern = "Likely supination tendency (lateral wear)"
            notes.append("Lateral side appears smoother than medial.")
            recommendation = "Prioritize cushioning + ankle mobility work."
        else:
            notes.append("Medial vs lateral wear looks fairly balanced.")
    else:
        notes.append("Tip: select Left/Right shoe for more accurate medial/lateral labeling.")

    shoes = _pick_shoes(activity, pronation_class)

    # Build overlay: wear heatmap + contour
    wear_color = cv2.applyColorMap(wear_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.78, wear_color, 0.22, 0)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, [max(cnts, key=cv2.contourArea)], -1, (255, 255, 255), 2)

    ok, buf = cv2.imencode(".png", overlay)
    overlay_png = bytes(buf) if ok else None

    metrics: Dict[str, Any] = {
        **qc["metrics"],
        **mask_meta,
        "rotation_deg": round(rot_deg, 2),
        "mask_conf": round(conf, 3),
        "shoe_side": shoe_side,
        "activity": (activity or "walking"),
        "weekly_miles": weekly_miles or "unknown",
        "surface": surface or "mixed",
        "heel_texture": round(heel_tex, 1),
        "mid_texture": round(mid_tex, 1),
        "fore_texture": round(fore_tex, 1),
        "heel_fore_wear_ratio": round(heel_fore_ratio, 3),
        "strike_hint": strike_hint,
        "left_texture": round(left_tex, 1),
        "right_texture": round(right_tex, 1),
    }
    if shoe_side in ("left", "right"):
        ml_ratio = medial_w / (lateral_w + EPS)
        metrics.update({
            "medial_texture": round(medial_tex, 1),
            "lateral_texture": round(lateral_tex, 1),
            "medial_lateral_wear_ratio": round(ml_ratio, 3),
        })

    return overlay_png, {
        "ok": True,
        "pattern": pattern,
        "confidence": round(conf, 2),
        "summary": " | ".join(notes),
        "recommendation": recommendation,
        "shoes": shoes,
        "metrics": metrics,
    }


def analyze_pair(
    left_image_pil,
    right_image_pil,
    activity: str = "walking",
    weekly_miles: str = "unknown",
    surface: str = "mixed",
) -> Dict[str, Any]:
    """
    Runs both shoes and produces an asymmetry report.
    Returns a dict including per-shoe results + pair summary.
    """
    left_png, left_res = analyze_single(left_image_pil, shoe_side="left", activity=activity, weekly_miles=weekly_miles, surface=surface)
    right_png, right_res = analyze_single(right_image_pil, shoe_side="right", activity=activity, weekly_miles=weekly_miles, surface=surface)

    pair = {
        "ok": bool(left_res.get("ok")) and bool(right_res.get("ok")),
        "left": left_res,
        "right": right_res,
        "pair_summary": "",
        "pair_metrics": {},
    }

    if not pair["ok"]:
        reasons = []
        if not left_res.get("ok"):
            reasons.append("Left shoe: " + (left_res.get("summary") or "needs retake"))
        if not right_res.get("ok"):
            reasons.append("Right shoe: " + (right_res.get("summary") or "needs retake"))
        pair["pair_summary"] = " | ".join(reasons)
        return {"pair": pair, "left_overlay_png": left_png, "right_overlay_png": right_png}

    # Asymmetry (compare key ratios)
    L = left_res["metrics"]
    R = right_res["metrics"]

    hf_L = float(L.get("heel_fore_wear_ratio", 1.0))
    hf_R = float(R.get("heel_fore_wear_ratio", 1.0))

    ml_L = float(L.get("medial_lateral_wear_ratio", 1.0))
    ml_R = float(R.get("medial_lateral_wear_ratio", 1.0))

    asym = abs(hf_L - hf_R) + abs(ml_L - ml_R)
    asym_score = float(min(1.0, asym / 1.5))  # heuristic scaling

    bullets = []
    if abs(hf_L - hf_R) > 0.12:
        if hf_L > hf_R:
            bullets.append("Left shoe shows relatively more heel-dominant wear than right.")
        else:
            bullets.append("Right shoe shows relatively more heel-dominant wear than left.")
    if abs(ml_L - ml_R) > 0.12:
        if ml_L > ml_R:
            bullets.append("Left shoe shows relatively more medial wear tendency than right.")
        else:
            bullets.append("Right shoe shows relatively more medial wear tendency than left.")
    if not bullets:
        bullets.append("Left vs right wear looks fairly consistent.")

    pair["pair_summary"] = " ".join(bullets)
    pair["pair_metrics"] = {
        "asymmetry_score_0_1": round(asym_score, 3),
        "heel_fore_ratio_left": round(hf_L, 3),
        "heel_fore_ratio_right": round(hf_R, 3),
        "medial_lateral_ratio_left": round(ml_L, 3),
        "medial_lateral_ratio_right": round(ml_R, 3),
    }

    return {"pair": pair, "left_overlay_png": left_png, "right_overlay_png": right_png}
