# scripts/build_index.py
import argparse, json, math, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

def read_docs(clean_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (path, text)."""
    docs = []
    for p in sorted(clean_dir.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        docs.append((str(p), txt))
    return docs

def guess_category_from_name(name: str) -> str:
    n = name.lower()
    if n.startswith("policy_") or "policy" in n or "admin" in n:
        return "policy_admin"
    if n.startswith("studyskills_") or "study" in n or "skills" in n:
        return "study_skills"
    if n.startswith("access_") or "accessibility" in n or "lgbt" in n or "neuro" in n or "divers" in n:
        return "identity_access"
    return "general"

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk(text: str, target=900, overlap=150) -> List[str]:
    """
    Simple passage chunker on paragraphs, then merge into ~target chars with overlap.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= target:
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            # create overlap by taking the end of previous buffer
            tail = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
            buf = f"{tail}\n\n{p}" if tail else p
    if buf:
        out.append(buf)
    # ensure not too tiny
    out = [c for c in out if len(c) >= min(250, target // 4)]
    return out or [text[:target]]

def build_index(clean_dir: Path, out_jsonl: Path, out_vec: Path, out_mat: Path, out_meta: Path):
    docs = read_docs(clean_dir)

    kb = []
    passages = []
    meta = []  # (source, category, passage_index)

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
                "text": ch
            }
            kb.append(rec)
            passages.append(ch)
            meta.append((src, category, i))
            pid += 1

    # Write JSONL knowledge base
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in kb:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        max_df=0.9,
        min_df=1,
        max_features=120000
    )
    X = vectorizer.fit_transform(passages)  # sparse (n_passages x vocab)

    # Persist artifacts
    joblib.dump(vectorizer, out_vec)
    sparse.save_npz(out_mat, X)

    # Small meta file for convenience
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump({"count_passages": X.shape[0], "count_docs": len(docs)}, f, indent=2)

    print(f"✅ Indexed {len(passages)} passages from {len(docs)} docs")
    print(f"• KB:     {out_jsonl}")
    print(f"• VEC:    {out_vec}")
    print(f"• MATRIX: {out_mat}")

def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF index over cleaned text passages.")
    ap.add_argument("--in", dest="inp", default="data/clean", help="Folder of cleaned .txt files")
    ap.add_argument("--kb", dest="kb", default="data/clean/kb.jsonl", help="Output KB jsonl")
    ap.add_argument("--vec", dest="vec", default="data/clean/vectorizer.pkl", help="Output vectorizer.pkl")
    ap.add_argument("--mat", dest="mat", default="data/clean/matrix.npz", help="Output TF-IDF matrix .npz")
    ap.add_argument("--meta", dest="meta", default="data/clean/meta.json", help="Output meta.json")
    args = ap.parse_args()

    build_index(Path(args.inp), Path(args.kb), Path(args.vec), Path(args.mat), Path(args.meta))

if __name__ == "__main__":
    main()
