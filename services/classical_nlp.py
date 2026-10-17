"""Classical sentence classifier: TF-IDF (1-2 grams) + Multinomial Naive Bayes, sigmoid-calibrated."""
from __future__ import annotations

import json
import re
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from . import config

HAND_TEST_CSV = config.ROOT / "models" / "data" / "hand_labelled_test.csv"
LABELS = config.LABELS

_model = None


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)),
        ("nb", MultinomialNB(alpha=0.2)),
    ])


def build_model():
    return CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)


def _brier(y_true: list[str], proba: np.ndarray, classes: list[str]) -> float:
    onehot = np.array([[1.0 if c == y else 0.0 for c in classes] for y in y_true])
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _evaluate(model, texts: list[str], labels: list[str]) -> dict:
    proba = model.predict_proba(texts)
    classes = list(model.classes_)
    pred = [classes[i] for i in proba.argmax(1)]
    srt = np.sort(proba, axis=1)
    conf, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    routed = (conf < config.CONF_MIN) | (margin < config.MARGIN_MIN)
    correct = np.array([p == y for p, y in zip(pred, labels)])
    return {
        "n": len(texts),
        "accuracy": round(float(accuracy_score(labels, pred)), 4),
        "macro_f1": round(float(f1_score(labels, pred, average="macro")), 4),
        "brier": round(_brier(labels, proba, classes), 4),
        "fallback_rate_at_thresholds": round(float(routed.mean()), 4),
        "accuracy_when_confident": round(float(correct[~routed].mean()), 4) if (~routed).any() else None,
        "accuracy_when_routed": round(float(correct[routed].mean()), 4) if routed.any() else None,
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(labels, pred, labels=classes).tolist()},
        "report": classification_report(labels, pred, output_dict=True, zero_division=0),
    }


def train(save: bool = True):
    """Train on synthetic seed data; evaluate on (a) synthetic hold-out and (b) hand-written test set."""
    from models import data_gen

    df = pd.DataFrame(data_gen.generate(), columns=["sentence", "label"])
    tr, te = train_test_split(df, test_size=0.2, random_state=1, stratify=df["label"])
    eval_model = build_model().fit(tr["sentence"], tr["label"])
    metrics = {"synthetic_holdout": _evaluate(eval_model, list(te["sentence"]), list(te["label"])),
               "thresholds": {"conf_min": config.CONF_MIN, "margin_min": config.MARGIN_MIN},
               "note": "Trained on template-generated synthetic data; hand_labelled_test is never trained on."}
    if HAND_TEST_CSV.exists():
        hand = pd.read_csv(HAND_TEST_CSV)
        metrics["hand_written_test"] = _evaluate(eval_model, list(hand["sentence"]), list(hand["label"]))
    final = build_model().fit(df["sentence"], df["label"])
    if save:
        config.CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, config.CLASSIFIER_PATH)
        config.CLASSIFIER_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return final


def load_model():
    """Load the saved model; retrain from the seed generator if missing or incompatible."""
    global _model
    if _model is not None:
        return _model
    try:
        _model = joblib.load(config.CLASSIFIER_PATH)
        _model.predict_proba(["warm up"])
    except Exception:
        _model = train(save=True)
    return _model


_REQUEST = re.compile(r"^(?:(?:okay|ok|so|alright|well)[,.]?\s+)?(?:\w+,\s*)?(?:please|(?:can|could|would) you)\b", re.I)


def _rule_override(text: str) -> Optional[str]:
    """Cheap, transparent rules applied on top of the model: questions and backchannel are not commitments."""
    t = text.strip()
    if len(t.split()) <= 3:
        return "DISCUSSION"
    if t.endswith("?") and not _REQUEST.search(t):
        return "QUESTION"
    return None


def predict(sentences: list[str]) -> list[dict]:
    """Return per-sentence {label, probs, conf, margin, rule}."""
    if not sentences:
        return []
    model = load_model()
    proba = model.predict_proba(sentences)
    classes = list(model.classes_)
    out = []
    for text, row in zip(sentences, proba):
        order = np.argsort(row)[::-1]
        item = {
            "label": classes[order[0]],
            "probs": {c: round(float(p), 4) for c, p in zip(classes, row)},
            "conf": float(row[order[0]]),
            "margin": float(row[order[0]] - row[order[1]]),
            "rule": None,
        }
        forced = _rule_override(text)
        if forced:
            item.update(label=forced, rule="override", conf=0.95, margin=0.9,
                        probs={c: (0.95 if c == forced else round(0.05 / (len(classes) - 1), 4)) for c in classes})
        out.append(item)
    return out


def load_metrics() -> Optional[dict]:
    try:
        return json.loads(config.CLASSIFIER_METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None
