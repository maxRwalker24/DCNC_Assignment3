import json
import pickle
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

KB = Path("data/kb.jsonl")
texts, ids, meta = [], [], []

with KB.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj["id"])
        texts.append(obj["text"])
        meta.append({
            k: obj.get(k)
            for k in ("source_title", "section_heading", "source_url", "page_start", "page_end", "tags")
        })

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=1,           # permissive for small corpora
    max_df=0.98
)
X = vectorizer.fit_transform(texts)

Path("data").mkdir(exist_ok=True)
sparse.save_npz("data/matrix.npz", X)
np.save("data/ids.npy", np.array(ids, dtype=object))
with open("data/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("data/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Indexed:", X.shape, "chunks")
