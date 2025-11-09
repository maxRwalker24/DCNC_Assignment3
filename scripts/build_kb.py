import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]  # project root
CLEAN_DIR = ROOT / "data" / "clean"
# where session_ingest.py expects these:
KB_PATH   = CLEAN_DIR / "kb.jsonl"
VEC_PATH  = CLEAN_DIR / "vectorizer.pkl"
MAT_PATH  = CLEAN_DIR / "matrix.npz"

# --- your helpers carried over / adapted ---

def infer_tags(heading: str, text: str):
    k = (heading + " " + text[:600]).lower()
    tags = []
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
        ("referenc", "writing"),
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
    return sorted(set(tags))

def tags_to_category(tags: List[str]) -> str:
    """
    Map your tags to the role keys used in session_ingest.py.
    """
    t = set(tags)
    if t & {"integrity","misconduct","assessment","extensions","special","appeals",
            "census","enrolment","credit"}:
        return "policy_admin"
    if t & {"study","time","writing","groupwork"}:
        return "study_skills"
    if t & {"els","elp","accessibility","neurodivergent","gender-affirmation","lgbtiqa",
            "wellbeing","safer-community"}:
        return "identity_access"
    return "general"

def split_by_headings(text: str):
    """
    Split by Markdown headings OR ALL-CAPS headings as a heuristic.
    Returns list[(heading, body)].
    """
    parts = re.split(r'(?m)^(#{1,6}\s.*|[A-Z][A-Z0-9 ,\-&/]{6,})\s*$', text)
    chunks = []
    if parts and parts[0].strip():
        chunks.append(("Introduction", parts[0].strip()))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        chunks.append((heading or "Section", body))
    return chunks

def sub_chunk(text: str, limit=1100):
    # Split by sentence-ish boundaries, enforce ~1.1k char chunks
    sents = re.split(r'(?<=[.?!])\s+', text)
    buf, out = "", []
    for s in sents:
        if len(buf) + len(s) > limit and buf:
            out.append(buf.strip())
            buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        out.append(buf)
    return out

def build_kb():
    KB_PATH.unlink(missing_ok=True)
    records = []

    # Use the cleaned .txt files (not data/intermediate)
    txt_files = sorted((CLEAN_DIR).glob("*.txt"))
    for f in txt_files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        title = f.stem.replace("_", " ").strip()
        doc_id = f.stem  # stable per source
        # scripts/build_kb.py (the version aligned to session_ingest)
        # inside the per-file loop:
        pi = 0  # global passage index for this document
        for heading, body in split_by_headings(raw):
            for piece in sub_chunk(body, limit=1100):
                if not piece.strip():
                    continue
                tags = infer_tags(heading, piece)
                category = tags_to_category(tags)
                rec = {
                    "doc_id": doc_id,
                    "source": str(f),
                    "title": title,
                    "category": category,
                    "passage_index": pi,   # <-- global, monotonic
                    "text": piece
                }
                records.append(rec)
                pi += 1




    # Vectorize texts for retrieval
    texts = [r["text"] for r in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    X = vectorizer.fit_transform(texts)

    # Save artifacts where session_ingest expects them
    with KB_PATH.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    joblib.dump(vectorizer, VEC_PATH)
    sparse.save_npz(MAT_PATH, X)

    print(f"Built KB: {len(records)} chunks → {KB_PATH}")
    print(f"Vectorizer → {VEC_PATH}")
    print(f"Matrix {X.shape} → {MAT_PATH}")

if __name__ == "__main__":
    build_kb()
