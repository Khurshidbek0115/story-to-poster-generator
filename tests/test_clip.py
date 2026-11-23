import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import clip_selector


def test_background_generation_and_metrics() -> None:
    bg_dir = os.path.join("assets", "backgrounds")
    paths = clip_selector.ensure_sample_backgrounds(bg_dir)
    assert len(paths) >= 6

    metrics = clip_selector.analyze_image_cv_metrics(paths[0])
    assert metrics["height"] > 0 and metrics["width"] > 0
    assert metrics["channels"] == 3
    assert 0.0 <= metrics["edge_density_pct"] <= 100.0
    print("OpenCV metrics test passed:", metrics)


def test_clip_selection() -> None:
    bg_dir = os.path.join("assets", "backgrounds")
    mock_nlp = {
        "clip_query": "a uplifting & heroic visual scene set in Jeju Island representing adventure, hope",
        "sentiment": {"label": "POSITIVE", "score": 0.99, "mood": "Uplifting & Heroic"},
    }
    res = clip_selector.select_best_background_clip(mock_nlp, bg_dir=bg_dir)
    assert os.path.exists(res["selected_path"])
    assert 0.0 <= res["selected_score"] <= 1.0
    assert len(res["all_rankings"]) >= 6
    print("CLIP selector test passed:", res["selected_filename"], res["selected_score"])


if __name__ == "__main__":
    test_background_generation_and_metrics()
    test_clip_selection()
