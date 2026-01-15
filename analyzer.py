from PIL import Image
import numpy as np
import cv2
import base64, io

def analyze(image_pil: Image.Image):
    img = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(bgr, 0.7, heatmap, 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    H, W = gray.shape
    f_end = int(0.30 * H); m_end = int(0.70 * H)
    fore = gray[:f_end].mean(); mid = gray[f_end:m_end].mean(); heel = gray[m_end:].mean()
    left = gray[:, :W//2].mean(); right = gray[:, W//2:].mean()

    notes = []
    if heel - fore > 5: notes.append("Heel striking tendency")
    elif fore - heel > 5: notes.append("Forefoot striking tendency")

    if left - right > 3:
        gait = "Likely supination (outer-edge wear)"
        notes.append("Left half appears smoother")
        advice = ["Balance & ankle mobility 3×/week","Avoid very stiff/minimal shoes for long wear"]
        shoes = ["Cushioned neutral: Brooks Glycerin, ASICS Cumulus, Nike Invincible"]
    elif right - left > 3:
        gait = "Likely overpronation (inner-edge wear)"
        notes.append("Right half appears smoother")
        advice = ["Consider arch-support insoles","Strengthen calves and foot intrinsics"]
        shoes = ["Stability: Brooks Adrenaline, ASICS Kayano, HOKA Arahi"]
    else:
        gait = "Neutral or mixed"
        notes.append("Wear appears roughly balanced")
        advice = ["Rotate pairs; replace when tread flattens","Light hips/calf mobility weekly"]
        shoes = ["Neutral supportive shoes appropriate to activity"]

    result_text = (
        f"Gait: {gait}\n\n"
        f"Notes:\n- " + "\n- ".join(notes) + "\n\n"
        f"Advice:\n- " + "\n- ".join(advice) + "\n\n"
        f"Recommended shoes:\n- " + "\n- ".join(shoes)
    )

    out = Image.fromarray(overlay_rgb)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return overlay_b64, result_text
