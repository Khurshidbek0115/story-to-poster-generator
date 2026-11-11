import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import nlp_analyzer


def test_split_sentences() -> None:
    sample = "First sentence here. Second sentence follows! Is there a third?"
    parts = nlp_analyzer.split_sentences(sample)
    assert len(parts) == 3
    assert parts[0] == "First sentence here."


def test_analyze_text_pipeline_book() -> None:
    sample_story = (
        "Detective Minjun walked through the rain-soaked streets of Seoul as neon "
        "holograms flickered above Gachon Tower. Inside CyberCore Corp, Dr. Elena Vance "
        "uncovered a dangerous conspiracy."
    )
    result = nlp_analyzer.analyze_text_pipeline(
        text=sample_story,
        style="book",
        custom_title="Echoes of Neo-Seoul",
    )
    assert result["title"] == "Echoes of Neo-Seoul"
    assert result["sentiment"]["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= result["sentiment"]["score"] <= 1.0
    assert "Seoul" in result["entities"]["LOC"]
    assert len(result["themes"]) > 0
    print("NLP pipeline test passed:", result["sentiment"], result["entities"])


if __name__ == "__main__":
    test_split_sentences()
    test_analyze_text_pipeline_book()
