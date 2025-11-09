import json
import pickle
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
with open(DATA / "vectorizer.pkl", "rb") as f:
    V = pickle.load(f)
X = sparse.load_npz(DATA / "matrix.npz")
IDS = np.load(DATA / "ids.npy", allow_pickle=True)
META = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))

# lazily load id->text map from kb.jsonl
KBMAP = None
def _load_kbmap():
    global KBMAP
    if KBMAP is None:
        KBMAP = {}
        with open("data/kb.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                KBMAP[o["id"]] = o["text"]
    return KBMAP

ID2META = {IDS[i]: META[i] for i in range(len(IDS))}

def retrieve(query: str, top_k=6, tag_boost=None):
    qv = V.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    order = sims.argsort()[::-1]
    results = []
    for idx in order[: top_k * 3]:  # overfetch for re-rank
        cid = IDS[idx]
        m = ID2META[cid]
        score = float(sims[idx])
        if tag_boost and any(t in (m.get("tags") or []) for t in tag_boost):
            score *= 1.1
        results.append((score, cid, m))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def page_span(m):
    a, b = m.get("page_start"), m.get("page_end")
    if a and b and a == b:
        return f" (p. {a})"
    if a and b:
        return f" (pp. {a}–{b})"
    return ""

def format_context(retrieved):
    kb = _load_kbmap()
    blocks = []
    for score, cid, m in retrieved:
        title = m.get("source_title") or "Source"
        sect = m.get("section_heading") or ""
        pages = page_span(m)
        text = kb.get(cid, "")
        blocks.append(f"### Source: {title}{pages}\n### Section: {sect}\n{text}\n")
    return "\n\n".join(blocks)
