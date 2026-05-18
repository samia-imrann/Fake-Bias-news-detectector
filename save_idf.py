"""
save_idf.py
One-time helper script.

Loads the preprocessed dataset + saved vocab, recomputes the exact IDF
vector that was used when building features.npz (same formula as
features.py::build_tfidf), then saves it as data/processed/idf.npy.

Run once:
    python save_idf.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import Vocabulary

PROCESSED_DIR = os.path.join("data", "processed")
DATASET_PATH  = os.path.join(PROCESSED_DIR, "dataset.npz")
VOCAB_PATH    = os.path.join(PROCESSED_DIR, "vocab.json")
IDF_OUT       = os.path.join(PROCESSED_DIR, "idf.npy")

print("Loading vocab …")
vocab = Vocabulary.load(VOCAB_PATH)
V = len(vocab)

print("Loading dataset …")
data  = np.load(DATASET_PATH, allow_pickle=True)
texts = list(data["texts"])          # already cleaned strings

print(f"  {len(texts)} documents, vocab size {V}")

print("Computing document frequencies …")
N  = len(texts)
df = np.zeros(V, dtype=np.float32)

for text in texts:
    tokens  = text.split()
    indices = set(vocab.encode(tokens))   # unique indices per doc
    for idx in indices:
        df[idx] += 1.0

# Same IDF formula as features.py::build_tfidf
# idf = log( (N+1) / (df+1) ) + 1   (smoothed, sklearn-style)
idf = np.log((N + 1) / (df + 1)) + 1.0

np.save(IDF_OUT, idf)
print(f"IDF saved → {IDF_OUT}  shape={idf.shape}")
print("Done.")
