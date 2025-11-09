# app/session_ingest.py
import os, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ---------- Load artefacts ----------

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CLEAN_DIR = ROOT / "data" / "clean"
KB_PATH   = CLEAN_DIR / "kb.jsonl"
VEC_PATH  = CLEAN_DIR / "vectorizer.pkl"
MAT_PATH  = CLEAN_DIR / "matrix.npz"

# Lazy singletons
_VECTORIZER = None
_MATRIX = None
_KB = None


def _load_vectorizer():
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(VEC_PATH)
    return _VECTORIZER


def _load_matrix():
    global _MATRIX
    if _MATRIX is None:
        _MATRIX = sparse.load_npz(MAT_PATH)
    return _MATRIX


def _load_kb():
    global _KB
    if _KB is None:
        recs: List[Dict] = []
        with KB_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        _KB = recs
    return _KB


# ---------- Retrieval ----------
# app/session_ingest.py
def retrieve(query: str, k: int = 4, category: Optional[str] = None,
             diversify: bool = True, per_source: int = 2) -> List[Dict]:
    kb = _load_kb()
    vec = _load_vectorizer()
    X = _load_matrix()
    qv = vec.transform([query])

    def doc_key(r):
        return r.get("doc_id") or r.get("source") or r.get("title") or r.get("url") or "?"

    if category:
        idxs = [i for i, r in enumerate(kb) if r.get("category") == category]
        if not idxs:
            return []
        sub_X = X[idxs, :]
        sims = cosine_similarity(qv, sub_X)[0]
        order = sims.argsort()[::-1]
        counts = {}
        top = []
        for local_i in order:
            i = idxs[local_i]
            r = kb[i].copy()
            key = doc_key(r)
            if diversify:
                c = counts.get(key, 0)
                if c >= per_source:
                    continue
                counts[key] = c + 1
            r["score"] = float(sims[local_i])
            top.append(r)
            if len(top) >= k:
                break
        for rank, r in enumerate(top, start=1):
            r["rank"] = rank
        return top

    # no category
    sims = cosine_similarity(qv, X)[0]
    order = sims.argsort()[::-1]
    counts = {}
    top = []
    for i in order:
        r = kb[i].copy()
        key = doc_key(r)
        if diversify:
            c = counts.get(key, 0)
            if c >= per_source:
                continue
            counts[key] = c + 1
        r["score"] = float(sims[i])
        top.append(r)
        if len(top) >= k:
            break
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank
    return top



def format_context(snippets: List[Dict], max_chars: int = 2800) -> Tuple[str, List[str]]:
    """
    Build a compact context string with inline source markers like [S#],
    where S# is stable per unique source (filename), not per snippet order.
    Returns (context_text, citations_list).
    """
    # 1) Assign stable IDs per unique source (order of first appearance)
    source_ids: Dict[str, int] = {}
    next_id = 1

    def mark_for(rec: Dict) -> str:
        nonlocal next_id
        src_path = rec.get("source", "")
        src_name = Path(src_path).name
        if src_name not in source_ids:
            source_ids[src_name] = next_id
            next_id += 1
        return f"[S{source_ids[src_name]}]"

    # 2) Build blocks within char budget
    blocks: List[str] = []
    total = 0
    for r in snippets:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        m = mark_for(r)
        remain = max_chars - total
        if remain <= 0:
            break
        piece = text[:remain]
        blocks.append(f"{m} {piece}")
        total += len(piece) + 2  # rough newline/spacing allowance

    ctx = "\n\n".join(blocks)

    # 3) Build a de-duplicated citations list, one per source
    cites: List[str] = []
    for src_name, sid in sorted(source_ids.items(), key=lambda kv: kv[1]):
        any_rec = next((r for r in snippets if Path(r.get("source", "")).name == src_name), None)
        cat = any_rec.get("category") if any_rec else None
        chunk = any_rec.get("passage_index") if any_rec else None

        line = f"[S{sid}] {src_name}"
        details = []
        if cat is not None:
            details.append(f"cat={cat}")
        if chunk is not None:
            details.append(f"chunk={chunk}")
        if details:
            line += " (" + ", ".join(details) + ")"

        cites.append(line)

    return ctx, cites




# ---------- Prompt assembly ----------

ROLE_SYSTEMS = {
    "policy_admin": (
        "You are an RMIT Policy & Admin Guide. Answer using ONLY the provided context.\n"
        "If the question is unrelated to context, say you don’t have information.\n"
        "Be concise, cite sources inline by [S#]. If advice has deadlines or forms, be explicit."
       

    ),
    "study_skills": (
        "You are an RMIT Study Skills Mentor (time management, group work, academic writing).\n"
        "Use ONLY the provided context for RMIT specifics; you may add generic study tips if clearly marked.\n"
        "Be practical, step-by-step, and cite context facts with [S#]."
        
    ),
    "identity_access": (
        "You are an RMIT Identity & Accessibility Support assistant (LGBTQIA+, disability, neurodivergent).\n"
        "Use ONLY the provided context for RMIT services and policies. Be respectful, trauma-informed,\n"
        "and list official contacts/links mentioned in the context. Cite with [S#]."
        
    ),
    "general": (
        "You are an RMIT Student Support assistant. Prefer the provided context; if insufficient, say so.\n"
        "Keep answers short and actionable. Cite context with [S#]."
        
    ),
}

CATEGORY_MAP = {
    "Policy & Admin Guide": "policy_admin",
    "Study Skills Mentor": "study_skills",
    "Identity & Accessibility": "identity_access",
    "General": "general",
}

def build_prompt(user_query: str, role_key: str, context_text: str) -> str:
    sys = ROLE_SYSTEMS.get(role_key, ROLE_SYSTEMS["general"])

    prompt = (
        f"{sys}\n\n"
        f"### Context (snippets with source markers)\n"
        f"{context_text}\n\n"

        f"### User question\n"
        f"{user_query}\n\n"

        "### Instructions\n"
        "- You MUST use the context accurately and ONLY use facts that appear in the snippets above.\n"
        "- Always cite the source marker ([S1], [S2], [S3]...) **immediately after** each factual claim.\n"
        "- If multiple snippets contain relevant facts, distribute citations across all appropriate [S#] sources.\n"
        "- Do NOT repeatedly cite the same marker unless ALL referenced facts come from that same snippet.\n"
        "- Never default to [S1]. If a detail is not in the context, write: (no source).\n"
        "- If the context is insufficient, explicitly state limitations and direct the student to the correct RMIT channel.\n"
        "- Keep the answer concise (~150–250 words) unless the user requests more detail.\n"
        "- Double-check each citation by matching the fact to the exact snippet containing it.\n"
        "- You may NOT invent additional details or interpretive expansions beyond what appears in the snippets.\n"
        "- Your goal is accuracy, clarity, and correct distribution of citations—not guessing.\n"
    )

    return prompt



# ---------- Public entrypoint for UI ----------

def get_context_and_prompt(
    user_query: str,
    ui_category_label: str,
    top_k: int = 4,
    max_ctx_chars: int = 2800
) -> Tuple[str, List[str], str]:
    """
    Returns (context_text, citations_list, final_prompt) for the given query & UI category.
    """
    role_key = CATEGORY_MAP.get(ui_category_label, "general")
    category_filter = role_key if role_key in {"policy_admin", "study_skills", "identity_access"} else None
    # hits = retrieve(user_query, k=top_k, category=category_filter)
    hits = retrieve(user_query, k=8)  # bump k temporarily for diagnostics

    from collections import Counter
    def _src_key(r):
        return (r.get("doc_id")
                or r.get("source")
                or r.get("url")
                or r.get("title")
                or "")

    print("Top hits by source key:")
    print(Counter(_src_key(r) for r in hits))

    print("\nTop hits raw sources (first 8):")
    for r in hits:
        print(_src_key(r), "|", r.get("source"), "|", r.get("title"), "| chunk", r.get("passage_index"))

    ctx, cites = format_context(hits, max_chars=max_ctx_chars)
    print(ctx)
    print("-- cites --")
    for c in cites: print(c)

    prompt = build_prompt(user_query, role_key, ctx)
    return ctx, cites, prompt
