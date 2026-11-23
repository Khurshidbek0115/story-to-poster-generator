import os
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np
from PIL import Image


_CLIP_CACHE: Dict[str, Any] = {}


def _get_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    if model_name not in _CLIP_CACHE:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP_CACHE[model_name] = (model, processor)
    return _CLIP_CACHE[model_name]


def ensure_sample_backgrounds(bg_dir: str) -> List[str]:
    os.makedirs(bg_dir, exist_ok=True)

    presets = [
        ("bg_cyber_city_night.jpg", "cyber_city"),
        ("bg_golden_sunrise_mountains.jpg", "golden_sunrise"),
        ("bg_dark_mystery_forest.jpg", "mystery_forest"),
        ("bg_crimson_crisis_alert.jpg", "crimson_alert"),
        ("bg_ocean_horizon_calm.jpg", "ocean_calm"),
        ("bg_emerald_tech_grid.jpg", "emerald_tech"),
    ]

    created_paths: List[str] = []
    for filename, preset_kind in presets:
        path = os.path.join(bg_dir, filename)
        if not os.path.exists(path):
            canvas = _render_procedural_background(preset_kind, width=900, height=1200)
            cv2.imwrite(path, canvas)
        created_paths.append(path)

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_files = sorted(
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if f.lower().endswith(valid_exts)
    )
    return all_files if all_files else created_paths


def _render_procedural_background(kind: str, width: int = 900, height: int = 1200) -> np.ndarray:
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    img = np.zeros((height, width, 3), dtype=np.uint8)

    rng = np.random.default_rng(abs(hash(kind)) % (2**32))

    if kind == "cyber_city":
        b = (45 + 115 * (1.0 - y_norm) + 30 * np.sin(x_norm * 6.0)).clip(0, 255)
        g = (15 + 55 * y_norm).clip(0, 255)
        r = (25 + 110 * (y_norm ** 1.3)).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        cv2.circle(img, (int(width * 0.72), int(height * 0.28)), 135, (210, 190, 255), -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (51, 51), 0)
        for x in range(0, width, 52):
            b_h = int(rng.integers(int(height * 0.22), int(height * 0.55)))
            b_w = int(rng.integers(44, 72))
            top_y = height - b_h
            cv2.rectangle(img, (x, top_y), (min(width, x + b_w), height), (24, 14, 18), -1)
            for wy in range(top_y + 18, height - 30, 24):
                for wx in range(x + 8, min(width - 10, x + b_w - 10), 16):
                    if rng.random() > 0.42:
                        win_color = (240, 220, 110) if rng.random() > 0.3 else (180, 110, 255)
                        cv2.rectangle(img, (wx, wy), (wx + 7, wy + 11), win_color, -1)

    elif kind == "golden_sunrise":
        b = (40 + 90 * (1.0 - y_norm)).clip(0, 255)
        g = (85 + 135 * np.sin(y_norm * np.pi * 0.85)).clip(0, 255)
        r = (160 + 90 * (1.0 - y_norm * 0.5)).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        cv2.circle(img, (int(width * 0.5), int(height * 0.36)), 150, (170, 240, 255), -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (45, 45), 0)
        layers = [
            (0.58, (55, 65, 115)),
            (0.68, (40, 45, 85)),
            (0.78, (25, 28, 52)),
        ]
        for base_ratio, color in layers:
            pts = [[0, height]]
            for x in range(0, width + 60, 60):
                peak = int(height * base_ratio + rng.integers(-65, 55))
                pts.append([x, peak])
            pts.append([width, height])
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)

    elif kind == "mystery_forest":
        b = (38 + 55 * (1.0 - y_norm)).clip(0, 255)
        g = (45 + 65 * (1.0 - y_norm)).clip(0, 255)
        r = (20 + 35 * (1.0 - y_norm)).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        cv2.circle(img, (int(width * 0.32), int(height * 0.25)), 95, (215, 230, 220), -1, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (35, 35), 0)
        for _ in range(28):
            tx = int(rng.integers(0, width))
            thick = int(rng.integers(6, 22))
            shade = int(rng.integers(12, 34))
            cv2.line(img, (tx, int(height * 0.15)), (tx, height), (shade, shade + 8, shade), thick)

    elif kind == "crimson_alert":
        b = (20 + 30 * (1.0 - y_norm)).clip(0, 255)
        g = (18 + 35 * (1.0 - y_norm)).clip(0, 255)
        r = (75 + 145 * np.sin(y_norm * np.pi)).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        for i in range(-2, 8):
            x1 = i * 160
            cv2.line(img, (x1, 0), (x1 + 420, height), (45, 55, 195), 18, cv2.LINE_AA)
        img = cv2.GaussianBlur(img, (21, 21), 0)

    elif kind == "ocean_calm":
        b = (110 + 115 * (1.0 - y_norm * 0.6)).clip(0, 255)
        g = (75 + 110 * np.sin(y_norm * np.pi * 0.75)).clip(0, 255)
        r = (25 + 65 * (1.0 - y_norm)).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        horizon_y = int(height * 0.56)
        cv2.line(img, (0, horizon_y), (width, horizon_y), (220, 210, 175), 2, cv2.LINE_AA)
        for wy in range(horizon_y + 15, height, 22):
            span = int(180 * (1.0 - (wy - horizon_y) / max(height - horizon_y, 1)) + 40)
            cx = width // 2 + int(rng.integers(-35, 35))
            cv2.line(img, (cx - span, wy), (cx + span, wy), (195, 185, 130), 2, cv2.LINE_AA)

    else:
        b = (45 + 90 * y_norm).clip(0, 255)
        g = (55 + 115 * (1.0 - y_norm * 0.5)).clip(0, 255)
        r = (15 + 35 * x_norm).clip(0, 255)
        img = np.stack([np.broadcast_to(b, (height, width)),
                        np.broadcast_to(g, (height, width)),
                        np.broadcast_to(r, (height, width))], axis=-1).astype(np.uint8)
        nodes: List[Tuple[int, int]] = [
            (int(rng.integers(60, width - 60)), int(rng.integers(80, height - 80)))
            for _ in range(22)
        ]
        for idx, (nx, ny) in enumerate(nodes):
            for mx, my in nodes[idx + 1: idx + 4]:
                cv2.line(img, (nx, ny), (mx, my), (130, 185, 75), 1, cv2.LINE_AA)
            cv2.circle(img, (nx, ny), 6, (180, 245, 120), -1, cv2.LINE_AA)

    return img


def analyze_image_cv_metrics(image_path: str) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    height, width, channels = img.shape
    mean_b, mean_g, mean_r = [round(float(v), 1) for v in img.mean(axis=(0, 1))]
    brightness = round((mean_b + mean_g + mean_r) / 3.0, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray, 100, 200)
    edge_density = round(float(np.count_nonzero(canny_edges)) / float(height * width) * 100.0, 2)

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "mean_bgr": (mean_b, mean_g, mean_r),
        "brightness": brightness,
        "edge_density_pct": edge_density,
    }


def select_best_background_clip(
    nlp_result: Dict[str, Any],
    bg_dir: str,
) -> Dict[str, Any]:
    candidate_paths = ensure_sample_backgrounds(bg_dir)
    model, processor = _get_clip_model("openai/clip-vit-base-patch32")

    target_query = nlp_result["clip_query"]
    sentiment_label = nlp_result["sentiment"]["label"]

    if sentiment_label == "POSITIVE":
        contrast_query = "a bleak, terrifying, pitch-black horror nightmare"
    else:
        contrast_query = "a cheerful cartoon birthday party with bright balloons"

    text_labels = [target_query, contrast_query]

    pil_images = [Image.open(p).convert("RGB") for p in candidate_paths]

    inputs = processor(
        text=text_labels,
        images=pil_images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().numpy()

    rankings: List[Dict[str, Any]] = []
    for path, prob_row in zip(candidate_paths, probs):
        match_prob = float(prob_row[0])
        cv_stats = analyze_image_cv_metrics(path)
        rankings.append(
            {
                "path": path,
                "filename": os.path.basename(path),
                "clip_score": round(match_prob, 4),
                "cv_metrics": cv_stats,
            }
        )

    rankings.sort(key=lambda item: item["clip_score"], reverse=True)
    best = rankings[0]

    return {
        "selected_path": best["path"],
        "selected_filename": best["filename"],
        "selected_score": best["clip_score"],
        "clip_query": target_query,
        "cv_metrics": best["cv_metrics"],
        "all_rankings": rankings,
    }
