# retrieve.py
"""
Lightweight retrieval helper for the local KB.

What this module does
---------------------
- Loads a TF-IDF vectorizer, sparse document matrix, and metadata saved by your
  build scripts (vectorizer.pkl, matrix.npz, ids.npy, meta.json).
- Computes cosine similarity between the user query and all KB chunks.
- Optionally applies a small re-ranking boost if a chunk has any of the tags
  listed in `tag_boost`.
- Returns the top-k matches, plus a convenience function to format a readable
  context block.

Files expected (under data/):
- vectorizer.pkl : fitted sklearn TfidfVectorizer
- matrix.npz     : scipy.sparse CSR/CSC of TF-IDF features for KB chunks
- ids.npy        : numpy array of KB chunk IDs (strings)
- meta.json      : list of per-chunk metadata dicts aligned with ids.npy
- kb.jsonl       : id -> text mapping (one JSON record per line)

Public API
----------
- retrieve(query: str, top_k=6, tag_boost=None) -> list[(score, id, meta)]
- format_context(retrieved) -> str
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data locations and one-time loads
# ---------------------------------------------------------------------------

DATA = Path("data")

# Load vectorizer, matrix, ids, meta on import. These are small enough and
# avoids reloading them for each query.
with open(DATA / "vectorizer.pkl", "rb") as f:
    V = pickle.load(f)

X = sparse.load_npz(DATA / "matrix.npz")

# ids.npy and meta.json must be aligned (same length, same order)
IDS: np.ndarray = np.load(DATA / "ids.npy", allow_pickle=True)
META: List[Dict] = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))

# Convenience dict to look up metadata by chunk id
ID2META: Dict[str, Dict] = {IDS[i]: META[i] for i in range(len(IDS))}

# Lazily-loaded map from chunk id -> chunk text (from kb.jsonl)
_KBMAP: Optional[Dict[str, str]] = None


def _load_kbmap() -> Dict[str, str]:
    """
    Build an in-memory mapping from chunk ID to cleaned text.
    Loaded once on first use to keep initial import fast.
    """
    global _KBMAP
    if _KBMAP is not None:
        return _KBMAP

    kbmap: Dict[str, str] = {}
    kb_path = DATA / "kb.jsonl"
    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("id")
            txt = rec.get("text", "")
            if cid:
                kbmap[cid] = txt
    _KBMAP = kbmap
    return _KBMAP


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 6,
    tag_boost: Optional[Iterable[str]] = None,
) -> List[Tuple[float, str, Dict]]:
    """
    Return top-k relevant chunks for `query`.

    Parameters
    ----------
    query : str
        User's search query.
    top_k : int
        Number of results to return.
    tag_boost : Optional[Iterable[str]]
        If provided, any chunk whose `meta["tags"]` contains at least one of
        these values will receive a small multiplicative boost (1.1x).

    Returns
    -------
    List[Tuple[float, str, Dict]]
        A list of (score, chunk_id, meta) tuples, highest score first.
        `meta` is the per-chunk metadata dict read from meta.json.
    """
    # 1) Vectorize query and compute cosine similarities over the whole matrix
    qv = V.transform([query])
    sims = cosine_similarity(qv, X).ravel()

    # 2) Initial ranking by raw similarity (we overfetch for a small re-rank step)
    overfetch = max(top_k * 3, top_k)  # small cushion to allow re-ranking/boosts
    order = sims.argsort()[::-1]  # descending

    # 3) Build candidate list and apply optional tag boosts
    results: List[Tuple[float, str, Dict]] = []
    boost_tags = set(tag_boost or [])
    for idx in order[:overfetch]:
        cid = IDS[idx]
        meta = ID2META.get(cid, {})
        score = float(sims[idx])

        # Light re-rank: multiply if the chunk has any boosted tag
        if boost_tags and any(t in (meta.get("tags") or []) for t in boost_tags):
            score *= 1.1

        results.append((score, cid, meta))

    # 4) Final sort by (possibly boosted) score and take top-k
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Pretty formatting helpers
# ---------------------------------------------------------------------------

def page_span(meta: Dict) -> str:
    """
    Render page numbers like ' (p. 3)' or ' (pp. 3–5)' if present in metadata.
    """
    a, b = meta.get("page_start"), meta.get("page_end")
    if a and b and a == b:
        return f" (p. {a})"
    if a and b:
        return f" (pp. {a}–{b})"
    return ""


def format_context(retrieved: List[Tuple[float, str, Dict]]) -> str:
    """
    Build a human-readable markdown block from retrieved tuples.

    Each block contains:
      - Source title (+ page span if any)
      - Section heading (if any)
      - The chunk text

    Parameters
    ----------
    retrieved : list of (score, id, meta)
        Output from `retrieve(...)`.

    Returns
    -------
    str
        Markdown string containing all selected chunks.
    """
    kb = _load_kbmap()
    parts: List[str] = []

    for score, cid, meta in retrieved:
        title = meta.get("source_title") or "Source"
        sect = meta.get("section_heading") or ""
        pages = page_span(meta)
        text = kb.get(cid, "")

        parts.append(
            f"### Source: {title}{pages}\n"
            f"### Section: {sect}\n"
            f"{text}\n"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Quick CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple manual test from the terminal:
    #   python retrieve.py "how do I get an extension"
    import sys
    q = " ".join(sys.argv[1:]).strip() or "extension request process"
    hits = retrieve(q, top_k=5, tag_boost=None)
    print(f"Query: {q!r}\n")
    for rank, (score, cid, meta) in enumerate(hits, start=1):
        print(f"{rank:>2}. score={score:.4f}  id={cid}  title={meta.get('source_title')}")
    print("\n--- Context ---\n")
    print(format_context(hits))
