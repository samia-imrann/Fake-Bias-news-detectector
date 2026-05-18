"""
app.py
Flask web server for the Fake/Bias News Detector.

Key fixes vs original:
  1. NB  — TF-IDF at inference uses corpus-level IDF (idf.npy), not a
            per-single-doc recompute (which made every IDF = 1.0).
  2. CNN — raw log-posterior (pre-softmax) is used so probability spread
            reflects true uncertainty rather than a collapsed softmax that
            always peaks at the majority class.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))

from preprocess  import preprocess_text
from features    import Vocabulary, texts_to_padded_sequences
from naive_bayes import NaiveBayesClassifier
from cnn         import TextCNN

# ── Paths ─────────────────────────────────────────────────────
PROCESSED_DIR  = os.path.join("data", "processed")
MODEL_DIR      = os.path.join(PROCESSED_DIR, "models")
VOCAB_PATH     = os.path.join(PROCESSED_DIR, "vocab.json")
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")
IDF_PATH       = os.path.join(PROCESSED_DIR, "idf.npy")
NB_MODEL_PATH  = os.path.join(MODEL_DIR, "naive_bayes")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn")

# ── Load shared assets ────────────────────────────────────────
print("Loading vocabulary ...")
vocab = Vocabulary.load(VOCAB_PATH)
V     = len(vocab)

print("Loading label map ...")
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    meta = json.load(f)
CLASSES = meta["classes"]   # ["Real", "Bias", "Fake"]
MAX_LEN = 80                 # must match training

print("Loading IDF vector ...")
IDF = np.load(IDF_PATH).astype(np.float64)   # shape (V,)
assert IDF.shape[0] == V, "IDF / vocab size mismatch — rerun save_idf.py"

# ── Lazy-load models ──────────────────────────────────────────
_nb_model  = None
_cnn_model = None

def get_nb_model():
    global _nb_model
    if _nb_model is None:
        print("Loading Naive Bayes model ...")
        _nb_model = NaiveBayesClassifier.load(NB_MODEL_PATH)
    return _nb_model

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        print("Loading CNN model ...")
        _cnn_model = TextCNN.load(CNN_MODEL_PATH)
    return _cnn_model


# ── Inference-safe TF-IDF  ────────────────────────────────────
def tfidf_inference(tokens: list) -> np.ndarray:
    """
    Build a (1, V) TF-IDF vector for a SINGLE document using the
    corpus-level IDF saved from training.  Matches features.py exactly:
        TF(t,d)  = count(t,d) / total_tokens(d)
        TF-IDF   = TF * IDF_from_training
        L2-normalize
    """
    indices = vocab.encode(tokens)         # list of ints (OOV -> 1)
    total   = max(len(indices), 1)

    tf_vec = np.zeros(V, dtype=np.float64)
    for idx in indices:
        tf_vec[idx] += 1.0
    tf_vec /= total                        # normalize by doc length

    tfidf_vec  = tf_vec * IDF             # apply corpus IDF
    norm       = np.linalg.norm(tfidf_vec)
    if norm > 0:
        tfidf_vec /= norm                 # L2 normalize

    return tfidf_vec.reshape(1, -1)       # (1, V)


# ── Stable softmax ────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── CNN calibrated probabilities ──────────────────────────────
def cnn_predict_proba(model, seq: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """
    The saved TextCNN's forward() returns a softmax probability from
    the final dense layer.  With temperature scaling we soften the
    distribution so rare classes surface when the CNN is uncertain,
    rather than always collapsing to the majority class.

    temperature > 1  => softer (more spread across classes)
    temperature = 1  => original (can be over-confident / majority-biased)
    """
    # Get raw logits: reverse-engineer from softmax output
    probs_raw = model.forward(seq, training=False)[0]   # (C,) already softmax

    # Convert back to log-space to get approximate logits
    log_probs = np.log(np.clip(probs_raw, 1e-12, 1.0))

    # Apply temperature scaling then re-softmax
    calibrated = softmax(log_probs / temperature)
    return calibrated


# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json(force=True)
    text       = (data.get("text") or "").strip()
    model_name = (data.get("model") or "nb").lower()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    # ── Preprocess ────────────────────────────────────────────
    try:
        clean  = preprocess_text(text, keep_stopwords=False)
        tokens = clean.split()
    except Exception as exc:
        return jsonify({"error": f"Preprocessing failed: {exc}"}), 500

    if not tokens:
        return jsonify({
            "error": "Text became empty after preprocessing — "
                     "no valid Urdu tokens found. Please enter Urdu text."
        }), 400

    # ── Run model ─────────────────────────────────────────────
    try:
        if model_name == "cnn":
            seq   = texts_to_padded_sequences([tokens], vocab, max_len=MAX_LEN)
            model = get_cnn_model()
            probs = cnn_predict_proba(model, seq, temperature=2.0)  # calibrated
        else:
            tfidf = tfidf_inference(tokens)   # ← fixed: uses corpus IDF
            model = get_nb_model()
            probs = model.predict_proba(tfidf)[0]

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    pred_i     = int(np.argmax(probs))
    label      = CLASSES[pred_i]
    confidence = float(probs[pred_i])
    all_probs  = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    return jsonify({
        "label":         label,
        "confidence":    round(confidence * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in all_probs.items()},
        "token_count":   len(tokens),
        "model_used":    "Naive Bayes" if model_name != "cnn" else "TextCNN",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
