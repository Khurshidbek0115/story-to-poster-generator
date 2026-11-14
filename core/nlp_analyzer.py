from typing import Dict, List, Any
import os
import re

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_PIPELINES: Dict[str, Any] = {}


def _get_pipeline(task: str, **kwargs):
    key = f"{task}:{sorted(kwargs.items())}"
    if key not in _PIPELINES:
        from transformers import pipeline
        _PIPELINES[key] = pipeline(task, **kwargs)
    return _PIPELINES[key]


def split_sentences(text: str) -> List[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in raw_parts if len(part.strip()) > 2]


def analyze_sentiment(text: str) -> Dict[str, Any]:
    classifier = _get_pipeline("sentiment-analysis")
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text[:512]]

    results = classifier([s[:512] for s in sentences])

    pos_scores = [r["score"] for r in results if r["label"] == "POSITIVE"]
    neg_scores = [r["score"] for r in results if r["label"] == "NEGATIVE"]

    pos_weight = sum(pos_scores)
    neg_weight = sum(neg_scores)

    if pos_weight >= neg_weight:
        overall_label = "POSITIVE"
        confidence = pos_weight / max(len(pos_scores), 1)
    else:
        overall_label = "NEGATIVE"
        confidence = neg_weight / max(len(neg_scores), 1)

    if overall_label == "POSITIVE":
        mood = "Uplifting & Heroic" if confidence >= 0.80 else "Hopeful & Warm"
    else:
        mood = "Dark & Suspenseful" if confidence >= 0.80 else "Tense & Dramatic"

    return {
        "label": overall_label,
        "score": round(float(confidence), 4),
        "mood": mood,
        "sentence_breakdown": results,
    }


def extract_entities(text: str) -> Dict[str, List[str]]:
    try:
        ner_pipe = _get_pipeline("ner", aggregation_strategy="simple")
    except TypeError:
        ner_pipe = _get_pipeline("ner", grouped_entities=True)
    raw_entities = ner_pipe(text)

    grouped: Dict[str, List[str]] = {
        "PER": [],
        "LOC": [],
        "ORG": [],
        "MISC": [],
    }

    for ent in raw_entities:
        group = ent.get("entity_group", "MISC")
        word = ent.get("word", "").replace("##", "").strip()
        score = float(ent.get("score", 0.0))

        if len(word) < 2 or score < 0.60:
            continue

        target_bucket = group if group in grouped else "MISC"
        if word not in grouped[target_bucket]:
            grouped[target_bucket].append(word)

    return grouped


def predict_themes_fill_mask(text: str, style: str = "book") -> List[Dict[str, Any]]:
    unmasker = _get_pipeline("fill-mask")
    mask_token = unmasker.tokenizer.mask_token

    snippet = text.strip()[:280]
    if style == "news":
        masked_prompt = f"{snippet} This news report mainly covers {mask_token}."
    else:
        masked_prompt = f"{snippet} Overall, this story is a tale of {mask_token}."

    predictions = unmasker(masked_prompt, top_k=3)
    themes = []
    for pred in predictions:
        token_str = pred.get("token_str", "").strip().lower()
        clean_token = re.sub(r"[^a-zA-Z0-9_-]", "", token_str)
        if clean_token:
            themes.append(
                {
                    "keyword": clean_token.capitalize(),
                    "score": round(float(pred.get("score", 0.0)), 4),
                }
            )
    return themes


def generate_auto_title(text: str, entities: Dict[str, List[str]], themes: List[Dict[str, Any]], style: str = "book") -> str:
    top_theme = themes[0]["keyword"] if themes else ("Chronicles" if style == "book" else "Report")
    persons = entities.get("PER", [])
    locations = entities.get("LOC", [])
    orgs = entities.get("ORG", [])

    if style == "news":
        if orgs and locations:
            return f"{orgs[0]} in {locations[0]}: {top_theme} Update"
        if locations:
            return f"{locations[0]} Focus: Major {top_theme} Shift"
        if orgs:
            return f"Inside {orgs[0]}: New {top_theme} Breakthrough"
    else:
        if persons and locations:
            return f"The {top_theme} of {locations[0]}"
        if locations:
            return f"Shadows Over {locations[0]}"
        if persons:
            return f"{persons[0]} and the {top_theme}"

    words = text.strip().split()[:5]
    return " ".join(words).rstrip(".,!?;:").title()


def analyze_text_pipeline(text: str, style: str = "book", custom_title: str = "") -> Dict[str, Any]:
    sentiment_info = analyze_sentiment(text)
    entities_info = extract_entities(text)
    themes_info = predict_themes_fill_mask(text, style=style)

    title = custom_title.strip() if custom_title.strip() else generate_auto_title(
        text, entities_info, themes_info, style=style
    )

    theme_words = ", ".join(t["keyword"].lower() for t in themes_info[:2]) or "dramatic atmosphere"
    loc_words = ", ".join(entities_info.get("LOC", [])[:2])
    location_clause = f" set in {loc_words}" if loc_words else ""

    clip_query = (
        f"a {sentiment_info['mood'].lower()} visual scene{location_clause} "
        f"representing {theme_words}"
    )

    return {
        "title": title,
        "raw_text": text.strip(),
        "style": style,
        "sentiment": sentiment_info,
        "entities": entities_info,
        "themes": themes_info,
        "clip_query": clip_query,
    }
