from __future__ import annotations

"""
SoleCheck analyzer — CV pipeline
=================================
  1. CLAHE preprocessing — adaptive contrast before every operation
  2. Multi-channel segmentation — Otsu on grayscale AND saturation, best mask wins
  3. GrabCut refinement pass when segmentation confidence is below threshold
  4. Contour-fitted zones — heel/mid/fore split on actual bounding box, not fixed image %
  5. Three-signal wear map — Laplacian (45%) + Sobel (30%) + HSV saturation (25%)
  6. Annotated overlay — zone boundary lines + labels drawn on the heatmap
  7. INFERNO colourmap — more perceptually distinct than JET
"""

from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import cv2

# ─── Tunables ────────────────────────────────────────────────────────────────
MAX_SIDE         = 800          # was 1100 — ~47% fewer pixels, much faster GrabCut
BLUR_VAR_MIN     = 35.0
DARK_MEAN_MIN    = 35.0
BRIGHT_MEAN_MAX  = 220.0
SOLE_CONF_MIN    = 0.13
SIDE_WEAR_THRESH = 0.11
HEEL_FORE_THRESH = 0.09

W_LAPLACIAN  = 0.45
W_SOBEL      = 0.30
W_SATURATION = 0.25

EPS = 1e-6


# ─── Preprocessing ───────────────────────────────────────────────────────────

def _resize_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    s = max_side / m
    return cv2.resize(bgr, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)


def _clahe(gray: np.ndarray) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cl.apply(gray)


# ─── Quality gate ─────────────────────────────────────────────────────────────

def _quality_checks(gray: np.ndarray) -> Dict[str, Any]:
    reasons: List[str] = []
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    if blur_var < BLUR_VAR_MIN:
        reasons.append("The photo looks blurry. Hold the camera steady and tap to focus on the sole.")
    mean_br = float(gray.mean())
    if mean_br < DARK_MEAN_MIN:
        reasons.append("The photo is too dark. Move to a brighter area or turn on a light.")
    elif mean_br > BRIGHT_MEAN_MAX:
        reasons.append("The photo is overexposed. Step away from direct overhead light.")
    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {"blur_var": round(blur_var, 2), "mean_brightness": round(mean_br, 2)},
    }


# ─── Segmentation ─────────────────────────────────────────────────────────────

def _mask_confidence(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    frac = float(cv2.countNonZero(mask)) / float(h * w)
    return float(max(0.0, min(1.0, (frac - 0.05) / 0.35)))


def _largest_contour_mask(bw: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(bw, dtype=np.uint8)
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
    return m


def _otsu_mask(channel: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(channel, (7, 7), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(bw)) > 127.0:
        bw = cv2.bitwise_not(bw)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)
    return _largest_contour_mask(bw)


def _grabcut_refine(bgr: np.ndarray, seed: np.ndarray) -> np.ndarray:
    h, w = seed.shape[:2]
    if h == 0 or w == 0:
        return seed
    ys, xs = np.where(seed > 0)
    if ys.size < 50:
        x0, y0, x1, y1 = int(0.10*w), int(0.10*h), int(0.90*w), int(0.90*h)
    else:
        p = 12
        x0, y0 = max(0, int(xs.min())-p), max(0, int(ys.min())-p)
        x1, y1 = min(w-1, int(xs.max())+p), min(h-1, int(ys.max())+p)
    rect = (x0, y0, max(1, x1-x0), max(1, y1-y0))
    gc   = np.zeros((h, w), np.uint8)
    bgd  = np.zeros((1, 65), np.float64)
    fgd  = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        return _largest_contour_mask(fg)
    except Exception:
        return seed


def _sole_mask(bgr: np.ndarray, gray_c: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1]

    mg   = _otsu_mask(gray_c)
    ms   = _otsu_mask(sat)
    cg   = _mask_confidence(mg)
    cs   = _mask_confidence(ms)

    if cg >= cs:
        seed, conf_seed, ch = mg, cg, "grayscale"
    else:
        seed, conf_seed, ch = ms, cs, "saturation"

    # Only run GrabCut when the initial mask is uncertain — saves 0.5-1s on clear photos
    refined    = seed
    conf_final = conf_seed
    if conf_seed < 0.45:
        gc_result  = _grabcut_refine(bgr, seed)
        gc_conf    = _mask_confidence(gc_result)
        if gc_conf >= conf_seed * 0.85:
            refined    = gc_result
            conf_final = gc_conf

    return refined, {
        "mask_conf_gray":  round(cg,         3),
        "mask_conf_sat":   round(cs,         3),
        "channel_used":    ch,
        "mask_conf_final": round(conf_final, 3),
    }


# ─── Rotation ─────────────────────────────────────────────────────────────────

def _rotate_bound(img: np.ndarray, angle: float, is_mask: bool = False) -> np.ndarray:
    h, w   = img.shape[:2]
    cX, cY = w/2.0, h/2.0
    M      = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int(h*sin + w*cos); nH = int(h*cos + w*sin)
    M[0,2] += nW/2.0 - cX; M[1,2] += nH/2.0 - cY
    return cv2.warpAffine(img, M, (nW, nH),
                          flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
                          borderValue=0)


def _normalize_rotation(bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, mask, 0.0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    rw, rh, angle = rect[1][0], rect[1][1], rect[2]
    if rw <= 0 or rh <= 0:
        return bgr, mask, 0.0
    rot     = angle if rw < rh else angle + 90.0
    applied = -rot
    bgr_r   = _rotate_bound(bgr,  applied)
    mask_r  = _rotate_bound(mask, applied, is_mask=True)
    if mask_r.shape[1] > mask_r.shape[0]:
        bgr_r  = _rotate_bound(bgr_r,  90.0)
        mask_r = _rotate_bound(mask_r, 90.0, is_mask=True)
        applied += 90.0
    return bgr_r, mask_r, float(applied)


# ─── Zone fitting ─────────────────────────────────────────────────────────────

def _contour_zones(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, _ = np.where(mask > 0)
    if ys.size == 0:
        h = mask.shape[0]
        return 0, int(0.30*h), int(0.70*h), h
    y_top = int(ys.min())
    y_bot = int(ys.max()) + 1
    span  = max(1, y_bot - y_top)
    return y_top, y_top + int(0.30*span), y_top + int(0.70*span), y_bot


# ─── Wear evidence ────────────────────────────────────────────────────────────

def _norm_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask > 0]
    if vals.size < 10:
        return np.zeros_like(arr, dtype=np.uint8)
    p5, p95 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    if p95 <= p5 + 1e-3:
        return np.zeros_like(arr, dtype=np.uint8)
    out = ((np.clip(arr.astype(np.float32), p5, p95) - p5) / (p95 - p5) * 255).astype(np.uint8)
    out[mask == 0] = 0
    return out


def _wear_map(bgr: np.ndarray, gray_c: np.ndarray, mask: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Laplacian
    lap_abs = cv2.convertScaleAbs(cv2.Laplacian(gray_c, cv2.CV_16S, ksize=3))
    inv_lap = (255 - lap_abs).astype(np.uint8)

    # Sobel
    gx = cv2.Sobel(gray_c, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_c, cv2.CV_16S, 0, 1, ksize=3)
    grad    = np.clip(cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32)), 0, 255).astype(np.uint8)
    inv_grad = (255 - grad).astype(np.uint8)

    # HSV saturation
    sat     = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    inv_sat = (255 - sat).astype(np.uint8)

    lap_n  = _norm_in_mask(inv_lap,  mask)
    grad_n = _norm_in_mask(inv_grad, mask)
    sat_n  = _norm_in_mask(inv_sat,  mask)

    wear = (W_LAPLACIAN  * lap_n.astype(np.float32) +
            W_SOBEL      * grad_n.astype(np.float32) +
            W_SATURATION * sat_n.astype(np.float32)).astype(np.uint8)
    wear[mask == 0] = 0

    return lap_abs, grad, wear


def _masked_mean(arr: np.ndarray, m: np.ndarray) -> float:
    pix = arr[m > 0]
    return float(pix.mean()) if pix.size > 0 else 0.0


def _wear_score(tex: float) -> float:
    return 1.0 / (tex + 1.0)


# ─── Overlay ──────────────────────────────────────────────────────────────────

def _build_overlay(bgr: np.ndarray, mask: np.ndarray, wear: np.ndarray,
                   y_fore_end: int, y_heel_start: int) -> Optional[bytes]:
    h, w       = bgr.shape[:2]
    wear_color = cv2.applyColorMap(wear, cv2.COLORMAP_INFERNO)
    overlay    = cv2.addWeighted(bgr, 0.72, wear_color, 0.28, 0)

    lc = (255, 255, 255)
    cv2.line(overlay, (0, y_fore_end),   (w, y_fore_end),   lc, 1)
    cv2.line(overlay, (0, y_heel_start), (w, y_heel_start), lc, 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    sc    = max(0.3, min(0.55, w / 700))
    tc    = (210, 210, 210)
    mid_y = y_fore_end + max(1, (y_heel_start - y_fore_end) // 2)
    cv2.putText(overlay, "FORE", (6, max(14, y_fore_end - 6)),   font, sc, tc, 1, cv2.LINE_AA)
    cv2.putText(overlay, "MID",  (6, mid_y),                     font, sc, tc, 1, cv2.LINE_AA)
    cv2.putText(overlay, "HEEL", (6, min(h-4, y_heel_start+16)), font, sc, tc, 1, cv2.LINE_AA)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, [max(cnts, key=cv2.contourArea)], -1, lc, 2)

    ok, buf = cv2.imencode(".png", overlay)
    return bytes(buf) if ok else None


# ─── Shoe picks ───────────────────────────────────────────────────────────────

def _pick_shoes(activity: str, pronation_class: str) -> str:
    activity = (activity or "walking").lower()
    if activity not in {"running", "walking", "basketball", "training"}:
        activity = "walking"

    if activity == "running":
        if pronation_class == "overpronation":
            return ("Look for a stability runner with a medial post — it physically limits the inward roll. "
                    "Brooks Adrenaline GTS, ASICS Gel-Kayano, and HOKA Arahi are the three most proven options. "
                    "All three run on the firmer side, so size up half if you have wide feet.")
        if pronation_class == "supination":
            return ("Your foot rolls outward, so you need max cushioning with a curved last, not a stability shoe. "
                    "Brooks Glycerin, ASICS Gel-Cumulus, and Nike Invincible Run absorb impact without adding structure. "
                    "Calf and ankle mobility work between runs will help long-term.")
        return ("A neutral daily trainer is the right call. Nike Pegasus, ASICS Novablast, and Saucony Ride 17 "
                "all hold up well across different surfaces and hit different price points. "
                "Replace around 400 miles or when the midsole starts feeling flat.")

    if activity == "basketball":
        if pronation_class == "overpronation":
            return ("Court shoes with lateral ankle support and a snug midfoot are your best bet. "
                    "Nike LeBron 21 and KD 17 both have wide bases that limit roll on cuts. "
                    "A supportive insole like Superfeet Green will help more than most shoe upgrades.")
        if pronation_class == "supination":
            return ("You need cushioning underfoot more than ankle support. "
                    "Nike GT Jump 2 and Jordan Zion 3 have softer midsoles that absorb impact on your outside edge. "
                    "Add ankle stability exercises to your warmup — that edge-loading pattern puts you at sprain risk.")
        return ("Any well-cushioned court shoe with torsional rigidity works for your pattern. "
                "Nike GT Cut 3 is the current benchmark for balance of cushion and lateral lockdown.")

    if activity == "training":
        if pronation_class == "overpronation":
            return ("A flat, stable trainer with a firm base works better than cushioned here — "
                    "too much softness under a pronating foot makes lateral movements unpredictable. "
                    "Nike Metcon 9 and Reebok Nano X4 are the two most popular options. Add a supportive insole if needed.")
        if pronation_class == "supination":
            return ("Get a training shoe with a softer, more forgiving midsole than the typical stiff gym shoe. "
                    "New Balance Minimus Trail or Adidas Dropset 3 both have more cushion than the Metcon/Nano category. "
                    "Work ankle circles and single-leg balance into your warmup.")
        return ("A standard cross trainer fits you well. Nike Metcon 9, Reebok Nano X4, and Adidas Dropset 3 "
                "all give you the flat, stable base you need for lifting without sacrificing too much on cardio days.")

    if pronation_class == "overpronation":
        return ("A stability walking shoe with arch support is the right move. "
                "ASICS GT-2000 13, Brooks Adrenaline GTS, and New Balance 860 are all solid picks with medial support built in. "
                "If you spend a lot of time on hard floors, a supportive insole like Superfeet Blue adds meaningful correction.")
    if pronation_class == "supination":
        return ("Prioritize cushioning over structure. HOKA Bondi 9, Brooks Glycerin 22, and ASICS Gel-Cumulus "
                "all have thick midsoles that absorb impact without adding the arch structure you don't need. "
                "Avoid minimalist or stability shoes.")
    return ("Your foot pattern is neutral, so you have a lot of flexibility. "
            "Brooks Ghost 16, Nike Pegasus 41, and ASICS Gel-Cumulus all work well and are available at different price points. "
            "Focus on fit over any specific support category.")


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_single(
    image_pil,
    shoe_side:    str = "unknown",
    activity:     str = "walking",
    weekly_miles: str = "unknown",
    surface:      str = "mixed",
) -> Tuple[Optional[bytes], Dict[str, Any]]:

    rgb    = np.array(image_pil.convert("RGB"))
    bgr    = _resize_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_c = _clahe(gray)

    qc = _quality_checks(gray_c)
    if not qc["ok"]:
        return None, {
            "ok": False, "pattern": "Retake needed", "confidence": 0.0,
            "summary": " ".join(qc["reasons"]),
            "recommendation": "Bright, even lighting works best. Fill the frame with the sole and hold steady.",
            "shoes": "", "metrics": qc["metrics"],
        }

    mask, mask_meta = _sole_mask(bgr, gray_c)
    conf0 = _mask_confidence(mask)
    if conf0 < SOLE_CONF_MIN:
        return None, {
            "ok": False, "pattern": "Retake needed", "confidence": round(conf0, 2),
            "summary": "We could not separate the sole from the background.",
            "recommendation": "Center the sole in frame, move closer, and use a plain background.",
            "shoes": "", "metrics": {**qc["metrics"], **mask_meta},
        }

    bgr, mask, rot_deg = _normalize_rotation(bgr, mask)
    gray_c = _clahe(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    conf = _mask_confidence(mask)
    if conf < SOLE_CONF_MIN:
        return None, {
            "ok": False, "pattern": "Retake needed", "confidence": round(conf, 2),
            "summary": "The sole was found but could not be aligned reliably.",
            "recommendation": "Center the sole and try a plain background.",
            "shoes": "", "metrics": {**qc["metrics"], **mask_meta, "rotation_deg": round(rot_deg, 2)},
        }

    y_top, y_fore_end, y_heel_start, y_bot = _contour_zones(mask)
    h, w = gray_c.shape[:2]

    lap_tex, _, wear = _wear_map(bgr, gray_c, mask)

    fore_tex  = _masked_mean(lap_tex[:y_fore_end,            :], mask[:y_fore_end,            :])
    mid_tex   = _masked_mean(lap_tex[y_fore_end:y_heel_start,:], mask[y_fore_end:y_heel_start,:])
    heel_tex  = _masked_mean(lap_tex[y_heel_start:,          :], mask[y_heel_start:,          :])

    # Split on the contour centroid, not the image midpoint, so an off-center photo
    # doesn't flip the medial/lateral labels.
    _mask_xs = np.where(mask > 0)[1]
    cx = int(_mask_xs.mean()) if _mask_xs.size > 0 else w // 2
    left_tex  = _masked_mean(lap_tex[:, :cx], mask[:, :cx])
    right_tex = _masked_mean(lap_tex[:, cx:], mask[:, cx:])

    fore_w  = _wear_score(fore_tex)
    heel_w  = _wear_score(heel_tex)
    left_w  = _wear_score(left_tex)
    right_w = _wear_score(right_tex)
    heel_fore_ratio = heel_w / (fore_w + EPS)

    shoe_side = (shoe_side or "unknown").lower()
    if shoe_side == "left":
        medial_w, lateral_w, medial_tex, lateral_tex = right_w, left_w, right_tex, left_tex
    elif shoe_side == "right":
        medial_w, lateral_w, medial_tex, lateral_tex = left_w, right_w, left_tex, right_tex
    else:
        medial_w = lateral_w = medial_tex = lateral_tex = None

    notes = []
    if heel_fore_ratio > (1.0 + HEEL_FORE_THRESH):
        notes.append("Your heel wears out faster than your forefoot. You hit heel first when you walk or run.")
        strike_hint = "heel"
    elif heel_fore_ratio < (1.0 - HEEL_FORE_THRESH):
        notes.append("Your forefoot takes more impact than the heel. You load through the front of your foot.")
        strike_hint = "forefoot"
    else:
        notes.append("Heel and forefoot wear are balanced.")
        strike_hint = "balanced"

    pronation_class = "neutral"
    pattern         = "Balanced wear"
    recommendation  = "Rotate your pairs. Replace the shoe once the tread flattens in the worn zones."
    if weekly_miles and weekly_miles != "unknown":
        recommendation = "Rotate your pairs and track mileage. Replace the shoe once the tread flattens in the worn zones."

    if shoe_side in ("left", "right"):
        ml_ratio = medial_w / (lateral_w + EPS)
        if ml_ratio > (1.0 + SIDE_WEAR_THRESH):
            pronation_class = "overpronation"
            pattern         = "Overpronation, inside edge wear"
            notes.append("The inside edge is more worn than the outside. Your foot rolls inward.")
            recommendation  = "Try a stability shoe. Foot and ankle strengthening exercises help correct the inward roll."
        elif ml_ratio < (1.0 - SIDE_WEAR_THRESH):
            pronation_class = "supination"
            pattern         = "Supination, outside edge wear"
            notes.append("The outside edge is more worn than the inside. Your foot rolls outward.")
            recommendation  = "Focus on cushioning and ankle mobility. A neutral shoe with extra padding helps here."
        else:
            notes.append("Inside and outside edge wear look even.")
    else:
        notes.append("Select Left or Right shoe to get medial and lateral wear labels.")

    overlay_png = _build_overlay(bgr, mask, wear, y_fore_end, y_heel_start)

    metrics: Dict[str, Any] = {
        **qc["metrics"], **mask_meta,
        "rotation_deg":         round(rot_deg,        2),
        "mask_conf":            round(conf,           3),
        "shoe_side":            shoe_side,
        "activity":             activity or "walking",
        "weekly_miles":         weekly_miles or "unknown",
        "surface":              surface or "mixed",
        "zone_fore_y":          y_fore_end,
        "zone_heel_y":          y_heel_start,
        "heel_texture":         round(heel_tex,       1),
        "mid_texture":          round(mid_tex,        1),
        "fore_texture":         round(fore_tex,       1),
        "heel_fore_wear_ratio": round(heel_fore_ratio,3),
        "strike_hint":          strike_hint,
        "left_texture":         round(left_tex,       1),
        "right_texture":        round(right_tex,      1),
    }
    if shoe_side in ("left", "right"):
        metrics.update({
            "medial_texture":            round(medial_tex, 1),
            "lateral_texture":           round(lateral_tex,1),
            "medial_lateral_wear_ratio": round(ml_ratio,   3),
        })

    return overlay_png, {
        "ok": True, "pattern": pattern,
        "confidence":     round(conf, 2),
        "summary":        " | ".join(notes),
        "recommendation": recommendation,
        "shoes":          _pick_shoes(activity, pronation_class),
        "metrics":        metrics,
    }


def analyze_pair(
    left_image_pil,
    right_image_pil,
    activity:     str = "walking",
    weekly_miles: str = "unknown",
    surface:      str = "mixed",
) -> Dict[str, Any]:

    left_png,  left_res  = analyze_single(left_image_pil,  "left",  activity, weekly_miles, surface)
    right_png, right_res = analyze_single(right_image_pil, "right", activity, weekly_miles, surface)

    pair: Dict[str, Any] = {
        "ok": bool(left_res.get("ok")) and bool(right_res.get("ok")),
        "left": left_res, "right": right_res,
        "pair_summary": "", "pair_metrics": {},
    }

    if not pair["ok"]:
        reasons = []
        if not left_res.get("ok"):
            reasons.append("Left shoe: " + (left_res.get("summary") or "needs a retake"))
        if not right_res.get("ok"):
            reasons.append("Right shoe: " + (right_res.get("summary") or "needs a retake"))
        pair["pair_summary"] = " | ".join(reasons)
        return {"pair": pair, "left_overlay_png": left_png, "right_overlay_png": right_png}

    L, R = left_res["metrics"], right_res["metrics"]
    hf_L = float(L.get("heel_fore_wear_ratio",      1.0))
    hf_R = float(R.get("heel_fore_wear_ratio",      1.0))
    ml_L = float(L.get("medial_lateral_wear_ratio", 1.0))
    ml_R = float(R.get("medial_lateral_wear_ratio", 1.0))

    asym_score = float(min(1.0, (abs(hf_L-hf_R) + abs(ml_L-ml_R)) / 1.5))

    bullets = []
    if abs(hf_L - hf_R) > 0.12:
        bullets.append("Your left shoe takes more heel impact than the right." if hf_L > hf_R
                       else "Your right shoe takes more heel impact than the left.")
    if abs(ml_L - ml_R) > 0.12:
        bullets.append("Your left foot rolls inward more than the right." if ml_L > ml_R
                       else "Your right foot rolls inward more than the left.")
    if not bullets:
        bullets.append("Left and right wear look consistent. Your gait is fairly symmetrical.")

    pair["pair_summary"] = " ".join(bullets)
    pair["pair_metrics"] = {
        "asymmetry_score_0_1":        round(asym_score, 3),
        "heel_fore_ratio_left":       round(hf_L, 3),
        "heel_fore_ratio_right":      round(hf_R, 3),
        "medial_lateral_ratio_left":  round(ml_L, 3),
        "medial_lateral_ratio_right": round(ml_R, 3),
    }
    return {"pair": pair, "left_overlay_png": left_png, "right_overlay_png": right_png}
