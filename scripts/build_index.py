# scripts/build_index.py
"""
Build a simple TF-IDF passage index from cleaned .txt files.

Pipeline (kept simple for the assignment):
1) Read all *.txt files from --in (e.g., data/clean).
2) Heuristically assign each file to a category (policy_admin / study_skills / identity_access / general).
3) Split each file into ~900-char overlapping passages (for better retrieval granularity).
4) Save:
   - kb.jsonl             : one JSON record per passage (id, source path, category, passage_index, text)
   - vectorizer.pkl       : fitted sklearn TfidfVectorizer
   - matrix.npz           : TF-IDF sparse matrix (rows = passages)
   - meta.json            : small summary counts

Usage:
  python scripts/build_index.py \
      --in data/clean \
      --kb data/clean/kb.jsonl \
      --vec data/clean/vectorizer.pkl \
      --mat data/clean/matrix.npz \
      --meta data/clean/meta.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_docs(clean_dir: Path) -> List[Tuple[str, str]]:
    """Return a list of (path_str, text) for all *.txt in `clean_dir`."""
    docs: List[Tuple[str, str]] = []
    for p in sorted(clean_dir.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        docs.append((str(p), txt))
    return docs


def guess_category_from_name(name: str) -> str:
    """
    Heuristic category guess based on filename.
    This is only used to help the UI filter/bias retrieval; it’s intentionally simple.
    """
    n = name.lower()
    if n.startswith("policy_") or "policy" in n or "admin" in n:
        return "policy_admin"
    if n.startswith("studyskills_") or "study" in n or "skills" in n:
        return "study_skills"
    if n.startswith("access_") or "accessibility" in n or "lgbt" in n or "neuro" in n or "divers" in n:
        return "identity_access"
    return "general"


# ---------------------------------------------------------------------------
# Normalisation & chunking
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Small, safe cleanup to keep lines tidy before chunking."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk(text: str, target: int = 900, overlap: int = 150) -> List[str]:
    """
    Paragraph-aware chunker:
    - Start with paragraph splits.
    - Merge into ~`target` characters.
    - Keep a small tail overlap to help retrieval across boundaries.
    - Drop very tiny chunks; ensure at least one chunk exists.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    buf = ""

    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= target:
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            tail = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
            buf = f"{tail}\n\n{p}" if tail else p

    if buf:
        out.append(buf)

    # Filter out very tiny chunks; fall back to a single head slice if needed.
    out = [c for c in out if len(c) >= min(250, target // 4)]
    return out or [text[:target]]


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(
    clean_dir: Path,
    out_jsonl: Path,
    out_vec: Path,
    out_mat: Path,
    out_meta: Path,
) -> None:
    """
    Create TF-IDF artefacts and a KB jsonl from cleaned text files.
    """
    docs = read_docs(clean_dir)

    kb: List[Dict] = []     # records for kb.jsonl
    passages: List[str] = []  # raw passages for vectorizer
    pid = 0

    for src, txt in docs:
        txt = normalize(txt)
        category = guess_category_from_name(Path(src).name)
        chs = chunk(txt, target=900, overlap=150)

        for i, ch in enumerate(chs):
            rec = {
                "id": f"p{pid}",
                "source": src,
                "category": category,
                "passage_index": i,
                "text": ch,
            }
            kb.append(rec)
            passages.append(ch)
            pid += 1

    # Write JSONL knowledge base
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in kb:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Fit TF-IDF (keep settings modest and explainable)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.9,     # ignore very common ngrams
        min_df=1,       # include rare ngrams (small corpora)
        max_features=120_000,  # cap vocab size
    )
    X = vectorizer.fit_transform(passages)  # sparse (n_passages x vocab)

    # Persist artefacts
    joblib.dump(vectorizer, out_vec)
    sparse.save_npz(out_mat, X)

    # Small metadata summary (counts only)
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {"count_passages": int(X.shape[0]), "count_docs": len(docs)},
            f,
            indent=2,
        )

    print(f"✅ Indexed {len(passages)} passages from {len(docs)} docs")
    print(f"• KB:     {out_jsonl}")
    print(f"• VEC:    {out_vec}")
    print(f"• MATRIX: {out_mat}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF index over cleaned text passages.")
    parser.add_argument("--in",   dest="inp",  default="data/clean",           help="Folder of cleaned .txt files")
    parser.add_argument("--kb",   dest="kb",   default="data/clean/kb.jsonl",  help="Output KB jsonl")
    parser.add_argument("--vec",  dest="vec",  default="data/clean/vectorizer.pkl", help="Output vectorizer.pkl")
    parser.add_argument("--mat",  dest="mat",  default="data/clean/matrix.npz",    help="Output TF-IDF matrix .npz")
    parser.add_argument("--meta", dest="meta", default="data/clean/meta.json",      help="Output meta.json")
    args = parser.parse_args()

    build_index(Path(args.inp), Path(args.kb), Path(args.vec), Path(args.mat), Path(args.meta))


if __name__ == "__main__":
    main()
