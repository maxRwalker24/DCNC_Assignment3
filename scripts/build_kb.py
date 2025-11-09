import re
import json
from pathlib import Path

INTER = Path("data/intermediate")
KB = Path("data/kb.jsonl")

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

def main():
    KB.unlink(missing_ok=True)
    count = 0
    with KB.open("w", encoding="utf-8") as fout:
        for p in INTER.glob("*.txt"):
            source_title = p.stem.replace("_", " ").title()
            raw = p.read_text(encoding="utf-8")
            for heading, body in split_by_headings(raw):
                for i, piece in enumerate(sub_chunk(body, limit=1100)):
                    rec = {
                        "id": f"{p.stem}:{heading}:{i}",
                        "source_title": source_title,
                        "source_url": "",  # optional: add the public URL if you keep a mapping
                        "source_type": "doc",
                        "page_start": None,
                        "page_end": None,
                        "section_heading": heading,
                        "text": piece,
                        "tags": infer_tags(heading, piece),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
    print(f"Built KB with {count} chunks -> {KB}")

if __name__ == "__main__":
    main()
