# scripts/extract_clean.py
"""
Extract & clean documents into plain text for indexing.

What it does:
1) Reads PDFs or HTML (or passes through plain text).
2) Normalises whitespace and line-break artifacts.
3) Removes repeated page headers/footers (PDFs) and boilerplate (HTML).
4) Drops very short noisy lines but keeps headings/bullets.
5) Writes cleaned .txt files to the output folder.

Usage:
  python scripts/extract_clean.py --in data/raw --out data/clean
  python scripts/extract_clean.py --in data/raw --out data/clean --only "policy_1318"
"""

import argparse
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import html2text

# Optional pdfminer fallback (installed via pdfminer.six)
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None


# -------------------------------------------------------------------
# Normalisation helpers
# -------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """
    Light text normalisation:
    - remove soft hyphen
    - unify line endings
    - fix hyphenated line breaks ("-\n")
    - squeeze runs of spaces/tabs
    - collapse 3+ blank lines to 2
    """
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def drop_link_footnotes(md: str) -> str:
    """
    html2text adds footnote-style links:
        [1]: https://example ...
    Remove those to keep the text compact.
    """
    return re.sub(r"(?m)^\s*\[\d+\]:\s+\S+\s*$", "", md)


def filter_short_noise(lines: List[str], min_len: int = 20) -> List[str]:
    """
    Drop very short lines that are unlikely to be useful (menus, crumbs),
    but keep headings and bullets even if they're short.
    """
    out: List[str] = []
    for ln in lines:
        t = ln.strip()
        if not t:
            out.append("")  # keep paragraph breaks
            continue
        # Keep headings and list bullets (markdown-style) even if short
        if re.match(r"^(#{1,6}\s|[-*•]\s|\d+\.\s)", t):
            out.append(t)
            continue
        if len(t) < min_len:
            continue
        out.append(t)

    # Collapse excessive blank lines again
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.split("\n")


def drop_repeated_headers_footers(pages: List[str], threshold: float = 0.6) -> List[str]:
    """
    Heuristic to remove repeating headers/footers in PDFs:
    - Look at first/last non-empty line on each page.
    - If a line appears on >= `threshold` of pages, strip it everywhere.
    """
    firsts, lasts = [], []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip()]
        if not lines:
            firsts.append("")
            lasts.append("")
            continue
        firsts.append(lines[0].strip())
        lasts.append(lines[-1].strip())

    def frequent(items: List[str]):
        counts = {}
        for x in items:
            if x:
                counts[x] = counts.get(x, 0) + 1
        n = len(items)
        return {k for k, v in counts.items() if v >= max(2, int(threshold * n))}

    rm_firsts = frequent(firsts)
    rm_lasts = frequent(lasts)

    cleaned_pages: List[str] = []
    for p in pages:
        lines = p.splitlines()
        # Remove standalone page numbers (extra safety)
        lines = [re.sub(r"(?m)^\s*\d+\s*$", "", l) for l in lines]
        # Drop repeating header/footer candidates
        if lines and lines[0].strip() in rm_firsts:
            lines = lines[1:]
        if lines and lines[-1].strip() in rm_lasts:
            lines = lines[:-1]
        cleaned_pages.append("\n".join(lines))
    return cleaned_pages


# -------------------------------------------------------------------
# Extractors
# -------------------------------------------------------------------

def pdf_to_text(p: Path, prefer: str = "pypdf2") -> str:
    """
    Extract text from a PDF via PyPDF2; if too short/empty and pdfminer is available,
    fallback to pdfminer. In both cases, try to remove repeated headers/footers.
    """
    def extract_with_pypdf2(pp: Path) -> str:
        try:
            reader = PdfReader(str(pp))
        except Exception:
            return ""
        pages: List[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        pages = drop_repeated_headers_footers(pages)
        text = "\n\n".join(pages)
        # Remove bare page numbers again (safety net)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        return text

    def extract_with_pdfminer(pp: Path) -> str:
        if not pdfminer_extract_text:
            return ""
        try:
            txt = pdfminer_extract_text(str(pp)) or ""
        except Exception:
            txt = ""
        # Split into pseudo-pages by form feed (if present) and clean
        pages = [t for t in txt.split("\f") if t]
        if pages:
            pages = drop_repeated_headers_footers(pages)
            return "\n\n".join(pages)
        return txt

    # Choose extraction order
    order = ["pypdf2", "pdfminer"] if prefer == "pypdf2" else ["pdfminer", "pypdf2"]
    text = ""
    for engine in order:
        text = extract_with_pypdf2(p) if engine == "pypdf2" else extract_with_pdfminer(p)
        if len(text.strip()) >= 300:  # heuristic: consider it "enough content"
            break
    return normalize_text(text)


def html_to_text(p: Path) -> str:
    """
    Convert HTML to markdown-ish text via BeautifulSoup + html2text,
    dropping common boilerplate tags.
    """
    html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove chrome/boilerplate
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript", "aside", "form"]):
        tag.decompose()

    md = html2text.html2text(str(soup))
    md = drop_link_footnotes(md)
    return normalize_text(md)


# -------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------

def process_file(src: Path, out_dir: Path, min_line_len: int) -> Path:
    """
    Read a single file, clean it, and write out a .txt neighbour in `out_dir`.
    """
    out = out_dir / (src.stem + ".txt")

    # Detect type
    suffix = src.suffix.lower()
    if suffix == ".pdf":
        text = pdf_to_text(src)
    elif suffix in {".html", ".htm"}:
        text = html_to_text(src)
    else:
        # Plain text passthrough (still normalised + noise-filtered)
        text = src.read_text(encoding="utf-8", errors="ignore")

    # Normalise + noise filter (keep headings/bullets)
    text = normalize_text(text)
    lines = filter_short_noise(text.split("\n"), min_len=min_line_len)
    final = "\n".join(lines).strip()

    out.write_text(final, encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract & clean PDFs/HTML into plain text.")
    parser.add_argument("--in",  dest="inp",  default="data/raw",  help="Input folder (default: data/raw)")
    parser.add_argument("--out", dest="out",  default="data/clean", help="Output folder (default: data/clean)")
    parser.add_argument("--only",              help="Process a single filename (substring match on name)")
    parser.add_argument("--min-line-len", type=int, default=20, help="Drop short noise lines under this length (default 20)")
    args = parser.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [p for p in inp.iterdir() if p.is_file()]
    if args.only:
        files = [p for p in files if args.only in p.name]

    for p in sorted(files):
        try:
            out_file = process_file(p, out, args.min_line_len)
            print(f"wrote: {out_file}")
        except Exception as e:
            print(f"[extract_clean] error processing {p.name}: {e}")

if __name__ == "__main__":
    main()
