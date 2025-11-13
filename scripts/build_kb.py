# scripts/build_kb.py
"""
Build a lightweight knowledge base (KB) and TF-IDF index from cleaned .txt files.

Pipeline (kept simple for the assignment):
1) Read cleaned .txt files in data/clean.
2) Heuristically split each file into (heading, body) sections.
3) Sub-chunk each body into ~1100-character passages.
4) Infer simple tags from each (heading + text), then map tags → category.
5) Emit kb.jsonl (one record per passage) and TF-IDF artefacts for retrieval.

Outputs (where session_ingest.py looks):
- data/clean/kb.jsonl
- data/clean/vectorizer.pkl
- data/clean/matrix.npz
"""

from pathlib import Path
from typing import List, Dict, Tuple
import re
import json

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Locations (aligned with session_ingest.py)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
KB_PATH   = CLEAN_DIR / "kb.jsonl"
VEC_PATH  = CLEAN_DIR / "vectorizer.pkl"
MAT_PATH  = CLEAN_DIR / "matrix.npz"


# ---------------------------------------------------------------------------
# Tagging & category mapping (very small, explainable heuristics)
# ---------------------------------------------------------------------------

def infer_tags(heading: str, text: str) -> List[str]:
    """
    Return a small set of tags inferred from the heading + first ~600 chars of text.
    Purely keyword-based.
    """
    k = (heading + " " + text[:600]).lower()
    tags: List[str] = []

    tag_map = [
        # Policy/Admin
        ("integrity", "integrity"),
        ("plagiarism", "misconduct"),
        ("collusion", "misconduct"),
        ("assessment", "assessment"),
        ("extension", "extensions"),
        ("special consideration", "special"),
        ("appeal", "appeals"),
        ("census", "census"),
        ("enrol", "enrolment"),
        ("credit", "credit"),
        ("rpl", "credit"),
        ("grade", "assessment"),
        ("misconduct", "misconduct"),

        # Study skills
        ("study", "study"),
        ("time management", "time"),
        ("time-management", "time"),
        ("referenc", "writing"),   # picks up reference/referencing
        ("writing", "writing"),
        ("academic skills", "study"),
        ("learning lab", "study"),
        ("group work", "groupwork"),
        ("team", "groupwork"),

        # Identity & Accessibility
        ("equitable learning", "els"),
        ("equitable learning service", "els"),
        ("equitable learning plan", "elp"),
        ("accessibility", "accessibility"),
        ("disability", "accessibility"),
        ("neurodiverg", "neurodivergent"),
        ("adhd", "neurodivergent"),
        ("autism", "neurodivergent"),
        ("dyslexia", "neurodivergent"),
        ("gender affirmation", "gender-affirmation"),
        ("affirmation", "gender-affirmation"),
        ("pronoun", "gender-affirmation"),
        ("name change", "gender-affirmation"),
        ("lgbtiqa", "lgbtiqa"),
        ("queer", "lgbtiqa"),
        ("pride", "lgbtiqa"),
        ("counselling", "wellbeing"),
        ("wellbeing", "wellbeing"),
        ("safer community", "safer-community"),
        ("placement", "wil"),
        ("work integrated learning", "wil"),
        ("wil", "wil"),
    ]
    for key, tag in tag_map:
        if key in k:
            tags.append(tag)

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def tags_to_category(tags: List[str]) -> str:
    """
    Map tags → the role keys used by session_ingest.py.
   
    """
    t = set(tags)
    if t & {"integrity", "misconduct", "assessment", "extensions", "special", "appeals",
            "census", "enrolment", "credit"}:
        return "policy_admin"
    if t & {"study", "time", "writing", "groupwork"}:
        return "study_skills"
    if t & {"els", "elp", "accessibility", "neurodivergent", "gender-affirmation", "lgbtiqa",
            "wellbeing", "safer-community"}:
        return "identity_access"
    return "general"


# ---------------------------------------------------------------------------
# Sectioning & chunking
# ---------------------------------------------------------------------------

def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Split a document into (heading, body) pairs.
    Heuristics:
      - Markdown headings:   ^#{1,6}\s.*
      - ALL-CAPS headings:   ^[A-Z][A-Z0-9 ,\-&/]{6,}$
    """
    parts = re.split(r'(?m)^(#{1,6}\s.*|[A-Z][A-Z0-9 ,\-&/]{6,})\s*$', text)
    chunks: List[Tuple[str, str]] = []

    # Anything before the first heading
    if parts and parts[0].strip():
        chunks.append(("Introduction", parts[0].strip()))

    # (heading, body) pairs
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunks.append((heading or "Section", body))

    return chunks


def sub_chunk(text: str, limit: int = 1100) -> List[str]:
    """
    Break a section body into ~`limit`-char passages by sentence-ish boundaries.
    """
    sents = re.split(r'(?<=[.?!])\s+', text)
    buf, out = "", []
    for s in sents:
        if buf and (len(buf) + len(s) > limit):
            out.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        out.append(buf.strip())
    return out


# ---------------------------------------------------------------------------
# Build KB + TF-IDF artefacts
# ---------------------------------------------------------------------------

def build_kb() -> None:
    """Create kb.jsonl + vectorizer.pkl + matrix.npz from data/clean/*.txt."""
    KB_PATH.unlink(missing_ok=True)

    records: List[Dict] = []

    # Iterate cleaned .txt files (stable order for reproducibility)
    txt_files = sorted(CLEAN_DIR.glob("*.txt"))
    for f in txt_files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        title = f.stem.replace("_", " ").strip()
        doc_id = f.stem  # stable per source

        pi = 0  # monotonic passage index per document
        for heading, body in split_by_headings(raw):
            for piece in sub_chunk(body, limit=1100):
                if not piece.strip():
                    continue
                tags = infer_tags(heading, piece)
                category = tags_to_category(tags)
                records.append({
                    "doc_id": doc_id,
                    "source": str(f),
                    "title": title,
                    "category": category,
                    "passage_index": pi,
                    "text": piece,
                })
                pi += 1

    # Vectorize texts for retrieval (small, transparent config)
    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)

    # Persist artefacts where session_ingest.py expects them
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KB_PATH.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    joblib.dump(vectorizer, VEC_PATH)
    sparse.save_npz(MAT_PATH, X)

    print(f"✅ Built KB: {len(records)} chunks → {KB_PATH}")
    print(f"✅ Vectorizer → {VEC_PATH}")
    print(f"✅ Matrix {X.shape} → {MAT_PATH}")


if __name__ == "__main__":
    build_kb()
