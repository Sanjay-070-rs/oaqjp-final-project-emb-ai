"""
EmotionDetection.emotion_detection
====================================
Detects emotions in a piece of text using a local HuggingFace model
(j-hartmann/emotion-english-distilroberta-base) via the ``transformers``
pipeline.  Works fully offline after the first download — no API keys
or external services required.

The model outputs 7 emotion labels.  We map them to the 5 canonical keys
used by the rest of the application:

    joy      → joy
    anger    → anger
    fear     → fear
    sadness  → sadness
    disgust  → disgust
    surprise → merged into joy  (positive surprise is joy-adjacent)
    neutral  → distributed proportionally (ignored in dominant calc)

If the model is not installed or inference fails the function returns a
safe fallback dictionary with all scores set to None and
dominant_emotion set to None — it never fabricates scores.
"""

from __future__ import annotations

_EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")

_EMPTY_RESULT: dict = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}

# ---------------------------------------------------------------------------
# Lazy-load the pipeline so import is fast and the model is only downloaded
# once on first call.
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    """Return (and cache) the transformers emotion-classification pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        _pipeline = hf_pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,           # return ALL label scores
            truncation=True,
            max_length=512,
        )
    return _pipeline


# Map from HF model labels → our 5 canonical keys
# "surprise" scores are added to "joy"; "neutral" is dropped.
_LABEL_MAP: dict[str, str | None] = {
    "joy":      "joy",
    "anger":    "anger",
    "fear":     "fear",
    "sadness":  "sadness",
    "disgust":  "disgust",
    "surprise": "joy",     # positive surprise → joy
    "neutral":  None,      # ignored
}


def emotion_detector(text_to_analyze: str) -> dict:
    """Analyse *text_to_analyze* and return an emotion dictionary.

    Uses a locally-running HuggingFace DistilRoBERTa model fine-tuned on
    emotion classification.  The model is downloaded automatically on the
    first call and cached in ``~/.cache/huggingface``.

    Args:
        text_to_analyze (str): The text to be analysed.

    Returns:
        dict: A dictionary with keys ``anger``, ``disgust``, ``fear``,
              ``joy``, ``sadness``, and ``dominant_emotion``.
              All values are ``None`` when the input is blank or when
              inference fails.

    Example::

        >>> from EmotionDetection import emotion_detector
        >>> result = emotion_detector("I am so happy today!")
        >>> result['dominant_emotion']
        'joy'
    """
    # ---- guard: blank / whitespace input --------------------------------
    if not text_to_analyze or not text_to_analyze.strip():
        return _EMPTY_RESULT.copy()

    try:
        pipe = _get_pipeline()
        raw: list[dict] = pipe(text_to_analyze)[0]  # list of {label, score}
    except Exception:  # noqa: BLE001
        return _EMPTY_RESULT.copy()

    # Accumulate scores into our 5 canonical buckets
    result: dict[str, float] = {k: 0.0 for k in _EMOTION_KEYS}
    for item in raw:
        label = item["label"].lower()
        target = _LABEL_MAP.get(label)
        if target is not None:
            result[target] = result.get(target, 0.0) + item["score"]

    # Determine dominant emotion
    dominant = max(result, key=result.get)  # type: ignore[arg-type]
    result["dominant_emotion"] = dominant

    return result
