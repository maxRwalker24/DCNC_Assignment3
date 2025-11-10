# app/session_ingest.py
"""
Session ingestion, retrieval, and prompt assembly for the RMIT chatbot.

Responsibilities
---------------
1) Load artefacts created by your build scripts:
   - data/clean/kb.jsonl         : one record per chunk (text + metadata)
   - data/clean/vectorizer.pkl   : fitted TfidfVectorizer
   - data/clean/matrix.npz       : TF-IDF sparse matrix (rows=chunks)
2) Retrieve top-k chunks for a user query (with small, readable boosts).
3) Build a compact context string with stable source markers [S1], [S2], ...
4) Assemble the final LLM prompt with role-specific system text.

Notes
-----
- This file is intentionally simple and well-commented for the assignment.
- “Auto (recommended)” role uses rule-based keyword detection to prefer a category
  but does not hard-filter (it biases scoring instead).
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Role detection (for "Auto (recommended)")
# =============================================================================

AUTO_ROLE = "auto"

# Very small keyword lists to nudge the "Auto" choice toward a category.
ROLE_KEYWORDS = {
    "policy_admin": {
        "assessment", "extension", "extensions", "special consideration", "appeal", "appeals",
        "census", "enrol", "enrolment", "credit", "rpl", "grade", "misconduct", "integrity", "plagiarism",
    },
    "study_skills": {
        "study", "time management", "time-management", "referencing", "reference", "writing",
        "learning lab", "group work", "teamwork", "note-taking", "revision", "exam tips",
    },
    "identity_access": {
        "equitable learning", "elp", "els", "accessibility", "disability", "neurodivergent", "adhd",
        "autism", "dyslexia", "gender affirmation", "pronoun", "name change", "lgbtiqa", "queer",
        "safer community", "wellbeing", "counselling", "counseling", "mental", "anxiety", "depression",
    },
}


def detect_role(user_query: str) -> str:
    """
    Rule-based role detection used by the "Auto" mode.

    Returns
    -------
    "policy_admin" | "study_skills" | "identity_access" | "general"
    """
    q = user_query.lower()
    scores = {k: 0 for k in ROLE_KEYWORDS}
    for role, terms in ROLE_KEYWORDS.items():
        for t in terms:
            if t in q:
                scores[role] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# =============================================================================
# Load artefacts (vectorizer, matrix, KB records)
# =============================================================================

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CLEAN_DIR = ROOT / "data" / "clean"
KB_PATH   = CLEAN_DIR / "kb.jsonl"
VEC_PATH  = CLEAN_DIR / "vectorizer.pkl"
MAT_PATH  = CLEAN_DIR / "matrix.npz"

# Lazy singletons to avoid reloading on every request.
_VECTORIZER = None   # type: ignore[var-annotated]
_MATRIX = None       # type: ignore[var-annotated]
_KB = None           # type: ignore[var-annotated]


def _load_vectorizer():
    """Load and cache the fitted TfidfVectorizer."""
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = joblib.load(VEC_PATH)
    return _VECTORIZER


def _load_matrix():
    """Load and cache the TF-IDF sparse matrix (each row is a chunk)."""
    global _MATRIX
    if _MATRIX is None:
        _MATRIX = sparse.load_npz(MAT_PATH)
    return _MATRIX


def _load_kb() -> List[Dict]:
    """Load and cache the KB JSONL records (text + metadata per chunk)."""
    global _KB
    if _KB is None:
        recs: List[Dict] = []
        with KB_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        _KB = recs
    return _KB


# =============================================================================
# Retrieval helpers (tiny and focused)
# =============================================================================

def _expand_query(q: str) -> str:
    """
    Lightweight normalization + synonym expansion to improve TF-IDF hit rate.
    Keep this tiny so it's easy to explain in the assignment.
    """
    ql = q.lower()

    # Simple normalisations
    replacements = {
        "well-being": "wellbeing",
        "well being": "wellbeing",
        "counseling": "counselling",  # US -> AU/UK
        "depressed": "depression",
        "anxious": "anxiety",
    }
    for a, b in replacements.items():
        ql = ql.replace(a, b)

    # Tiny synonym hints
    synonyms = [
        ("depression", ["depressed", "low mood", "sadness"]),
        ("anxiety", ["anxious", "panic"]),
        ("wellbeing", ["mental health", "well-being", "well being"]),
        ("counselling", ["counseling", "therapy", "support"]),
    ]
    expanded = [ql]
    for head, alts in synonyms:
        if any(term in ql for term in [head] + alts):
            expanded.append(head)
            expanded.extend(alts)

    # De-duplicate tokens while preserving order
    tokens = list(dict.fromkeys(" ".join(expanded).split()))
    return " ".join(tokens)


def _is_mental_intent(query: str) -> bool:
    """Detect mental-health intent (used only for a small scoring nudge)."""
    terms = {
        "mental", "wellbeing", "well-being", "counselling", "counseling",
        "anxiety", "depress", "therapy", "support",
    }
    ql = query.lower()
    return any(t in ql for t in terms)


def _doc_boost(rec: Dict, is_mental: bool) -> float:
    """
    Small additive boost if the query looks mental-health related and the record
    appears to be wellbeing content. Keep values tiny so they just break ties.
    """
    if not is_mental:
        return 0.0

    score = 0.0
    txt   = (rec.get("text")   or "").lower()
    title = (rec.get("title")  or "").lower()
    src   = (rec.get("source") or "").lower()
    tags  = set(rec.get("tags") or [])

    if {"wellbeing", "safer-community"} & tags:
        score += 0.06
    if any(t in txt   for t in ("wellbeing", "counselling", "counseling", "mental", "anxiety", "depression")):
        score += 0.06
    if any(t in title for t in ("wellbeing", "counselling", "mental", "anxiety", "depression")):
        score += 0.04
    if "wellbeing" in src or "mental" in src:
        score += 0.02
    return score


CATEGORY_BOOST = {
    "policy_admin": 0.02,
    "study_skills": 0.02,
    "identity_access": 0.03,
    "general": 0.0,
}


def _cat_boost(rec: Dict, preferred_category: Optional[str]) -> float:
    """
    Tiny additive boost when a record's category == preferred_category
    (used in Auto mode; searching widely while nudging the detected category).
    """
    if not preferred_category:
        return 0.0
    return CATEGORY_BOOST.get(preferred_category, 0.0) if rec.get("category") == preferred_category else 0.0


# =============================================================================
# Retrieval
# =============================================================================

def retrieve(
    query: str,
    k: int = 4,
    category: Optional[str] = None,           # hard filter (None = search all)
    diversify: bool = True,
    per_source: int = 2,
    fetch_k: int = 40,
    preferred_category: Optional[str] = None, # soft target for boosting (Auto mode)
) -> List[Dict]:
    """
    Retrieve top-k chunks for `query`.

    Behaviour (kept intentionally simple):
    - Expand the query a little to improve TF-IDF recall.
    - Compute cosine similarity against all KB chunks.
    - If `category` is set, only consider those chunks (hard filter).
    - If raw similarities are all zero, do a keyword fallback to seed candidates.
    - Apply small mental-health and category boosts.
    - Optionally diversify by capping results per source.
    """
    kb = _load_kb()
    vec = _load_vectorizer()
    X   = _load_matrix()

    # 1) Vectorize expanded query
    expanded_query = _expand_query(query)
    qv = vec.transform([expanded_query])

    # 2) Cosine similarities (1 x N)
    sims_all = cosine_similarity(qv, X)[0]

    # 3) Index set (optional hard category filter)
    if category:
        idxs = [i for i, r in enumerate(kb) if r.get("category") == category]
        if not idxs:
            return []
    else:
        idxs = list(range(len(kb)))

    # 4) Candidate pool: either normal top-N or zero-sim fallback
    if sims_all.max() == 0.0:
        KEY_TERMS = {
            "depression", "depressed", "anxiety", "anxious",
            "wellbeing", "well-being", "counselling", "counseling",
            "mental", "support",
        }
        prelim = []
        for i in idxs:
            txt = (kb[i].get("text") or "").lower()
            if any(term in txt for term in KEY_TERMS):
                prelim.append(i)
        pre = prelim[:max(fetch_k, k)] if prelim else idxs[:max(fetch_k, k)]
    else:
        pre = sorted(idxs, key=lambda i: sims_all[i], reverse=True)[:max(fetch_k, k)]

    # 5) Score with tiny boosts (mental intent + preferred category)
    is_mental = _is_mental_intent(query)
    scored: List[Tuple[int, float, float]] = []
    for i in pre:
        base = float(sims_all[i])
        boosted = base + _doc_boost(kb[i], is_mental) + _cat_boost(kb[i], preferred_category)
        scored.append((i, boosted, base))

    # 6) Re-rank by boosted score
    scored.sort(key=lambda t: t[1], reverse=True)

    # 7) Diversify with per-source cap
    def doc_key(r: Dict) -> str:
        return r.get("doc_id") or r.get("source") or r.get("title") or r.get("url") or "?"

    counts: Dict[str, int] = {}
    top: List[Dict] = []
    for i, boosted, base in scored:
        r = kb[i].copy()
        key = doc_key(r)
        if diversify and counts.get(key, 0) >= per_source:
            continue
        counts[key] = counts.get(key, 0) + 1
        r["score"]   = boosted     # for transparency/debug
        r["sim_raw"] = base        # for transparency/debug
        top.append(r)
        if len(top) >= k:
            break

    # 8) Attach rank numbers
    for rank, r in enumerate(top, start=1):
        r["rank"] = rank

    return top


# =============================================================================
# Context formatting (stable [S#] per source file)
# =============================================================================

def format_context(snippets: List[Dict], max_chars: int = 2800) -> Tuple[str, List[str]]:
    """
    Build a compact context string with inline source markers like [S1].
    S# is stable per unique source file (order of first appearance), not per snippet rank.

    Returns
    -------
    (context_text, citations_list)
    """
    # 1) Assign stable IDs per unique source
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

    # 2) Build context blocks within a character budget
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
        total += len(piece) + 2  # rough spacing allowance

    ctx = "\n\n".join(blocks)

    # 3) Build a unique citations list (one line per source)
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


# =============================================================================
# Prompt assembly
# =============================================================================

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
    """Assemble the final prompt sent to the LLM."""
    sys = ROLE_SYSTEMS.get(role_key, ROLE_SYSTEMS["general"])
    return (
        f"{sys}\n\n"
        f"### Context (snippets with source markers)\n"
        f"{context_text}\n\n"
        f"### User question\n"
        f"{user_query}\n\n"
        "### Instructions\n"
        "- Use ONLY facts present in the Context snippets above.\n"
        "- Cite the correct source marker ([S1], [S2], …) immediately after each factual claim.\n"
        "- If multiple snippets contain relevant facts, distribute citations across those [S#] sources.\n"
        "- Do NOT reuse the same [S#] unless every cited fact truly comes from that one snippet.\n"
        "- Never guess or default to [S1]. If a required detail is missing, write: (no source).\n"
        "- Only use [S#] markers that actually appear in the Context block.\n"
        "- If the Context is insufficient, state the limitation and direct the student to the appropriate RMIT channel.\n"
        "- Keep the answer concise (~150–250 words) unless the user asks for more detail.\n"
        "- Double-check each citation matches the exact snippet containing that fact.\n"
        "- Do not invent extra policy details or interpretations beyond the snippets.\n"
        "- When ≥2 distinct [S#] appear in Context and are relevant, use at least two different markers in your answer.\n"
    )


# =============================================================================
# Public entrypoint used by the UI
# =============================================================================

def get_context_and_prompt(
    user_query: str,
    ui_category_label: str,
    top_k: int = 4,
    max_ctx_chars: int = 2800,
) -> Tuple[str, List[str], str]:
    """
    Main helper for the UI: returns (context_text, citations_list, final_prompt).

    - If the UI selected "Auto", detect a preferred category but search widely.
    - Otherwise, enforce the selected category as a hard filter.
    """
    # 1) Map UI label → role_key (allow "Auto")
    if ui_category_label.lower().startswith("auto"):
        role_key = AUTO_ROLE
    else:
        role_key = CATEGORY_MAP.get(ui_category_label, "general")

    # 2) Decide filtering strategy
    if role_key == AUTO_ROLE:
        detected = detect_role(user_query)  # "policy_admin" | "study_skills" | "identity_access" | "general"
        # If mental intent but detection didn’t pick identity_access, prefer it.
        if _is_mental_intent(user_query) and detected != "identity_access":
            detected = "identity_access"
        category_filter = None            # search all categories
        preferred_category = detected     # nudge scores toward detected
    else:
        preferred_category = None
        category_filter = role_key if role_key in {"policy_admin", "study_skills", "identity_access"} else None

    # 3) Retrieve with the chosen strategy
    hits = retrieve(
        user_query,
        k=top_k,
        category=category_filter,              # hard filter or None
        preferred_category=preferred_category, # soft bias when Auto
    )

    # (Optional) Debug print; helpful while developing
    print(f"[AUTO] role={role_key} preferred={preferred_category} filter={category_filter}")

    # 4) Build context and final prompt
    ctx, cites = format_context(hits, max_chars=max_ctx_chars)
    prompt = build_prompt(user_query, (preferred_category or role_key), ctx)
    return ctx, cites, prompt
