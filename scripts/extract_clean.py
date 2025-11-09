# scripts/extract_clean.py
import argparse
import re
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import html2text

# --- Optional pdfminer fallback (installed via pdfminer.six) ---
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None


def normalize_text(s: str) -> str:
    # common artifacts
    s = s.replace("\u00ad", "")               # soft hyphen
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n", "", s)                 # hyphenated line breaks
    s = re.sub(r"[ \t]+", " ", s)             # squeeze spaces
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def drop_link_footnotes(md: str) -> str:
    # html2text adds lines like: [1]: https://example ...
    return re.sub(r"(?m)^\s*\[\d+\]:\s+\S+\s*$", "", md)


def filter_short_noise(lines: List[str], min_len: int = 20) -> List[str]:
    out = []
    for ln in lines:
        t = ln.strip()
        if not t:
            out.append("")  # keep paragraph breaks
            continue
        # keep headings and list bullets even if short
        if re.match(r"^(#{1,6}\s|[-*•]\s|\d+\.\s)", t):
            out.append(t)
            continue
        if len(t) < min_len:
            continue
        out.append(t)
    # collapse extra blanks
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.split("\n")


def drop_repeated_headers_footers(pages: List[str], threshold: int = 0.6) -> List[str]:
    """
    Find first/last non-empty line on each page; if a line occurs on >= threshold of pages,
    treat as header/footer and remove all occurrences (page numbers, running headers, etc.).
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

    def frequent(items):
        counts = {}
        for x in items:
            if not x:
                continue
            counts[x] = counts.get(x, 0) + 1
        n = len(items)
        return {k for k, v in counts.items() if v >= max(2, int(threshold * n))}

    rm_firsts = frequent(firsts)
    rm_lasts = frequent(lasts)

    cleaned_pages = []
    for p in pages:
        lines = p.splitlines()
        # drop bare page numbers
        lines = [re.sub(r"(?m)^\s*\d+\s*$", "", l) for l in lines]
        # drop frequent header/footer candidates
        if lines:
            if lines[0].strip() in rm_firsts:
                lines = lines[1:]
        if lines:
            if lines[-1].strip() in rm_lasts:
                lines = lines[:-1]
        cleaned_pages.append("\n".join(lines))
    return cleaned_pages


def pdf_to_text(p: Path, prefer: str = "pypdf2") -> str:
    """
    Extract via PyPDF2; if too short or empty and pdfminer is available, fallback.
    Also remove repeated headers/footers.
    """
    def extract_with_pypdf2(pp: Path) -> str:
        try:
            reader = PdfReader(str(pp))
        except Exception:
            return ""
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        pages = drop_repeated_headers_footers(pages)
        text = "\n\n".join(pages)
        # remove standalone page numbers again (safety net)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        return text

    def extract_with_pdfminer(pp: Path) -> str:
        if not pdfminer_extract_text:
            return ""
        try:
            txt = pdfminer_extract_text(str(pp)) or ""
        except Exception:
            txt = ""
        # split into pseudo-pages by form feed if present
        pages = [t for t in txt.split("\f") if t]
        if pages:
            pages = drop_repeated_headers_footers(pages)
            return "\n\n".join(pages)
        return txt

    # choose order
    order = ["pypdf2", "pdfminer"] if prefer == "pypdf2" else ["pdfminer", "pypdf2"]
    text = ""
    for engine in order:
        text = extract_with_pypdf2(p) if engine == "pypdf2" else extract_with_pdfminer(p)
        if len(text.strip()) >= 300:  # heuristic: enough content
            break
    return normalize_text(text)


def html_to_text(p: Path) -> str:
    html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    # strip chrome
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript", "aside", "form"]):
        tag.decompose()
    md = html2text.html2text(str(soup))
    md = drop_link_footnotes(md)
    return normalize_text(md)


def process_file(src: Path, out_dir: Path, min_line_len: int) -> Path:
    out = out_dir / (src.stem + ".txt")
    if src.suffix.lower() == ".pdf":
        text = pdf_to_text(src)
    elif src.suffix.lower() in {".html", ".htm"}:
        text = html_to_text(src)
    else:
        text = src.read_text(encoding="utf-8", errors="ignore")

    text = normalize_text(text)
    lines = text.split("\n")
    lines = filter_short_noise(lines, min_len=min_line_len)
    final = "\n".join(lines).strip()
    out.write_text(final, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract & clean PDFs/HTML into plain text.")
    ap.add_argument("--in", dest="inp", default="data/raw", help="Input folder (default: data/raw)")
    ap.add_argument("--out", dest="out", default="data/clean", help="Output folder (default: data/clean)")
    ap.add_argument("--only", dest="only", help="Process a single filename (stem or full)")
    ap.add_argument("--min-line-len", type=int, default=20, help="Drop short noise lines under this length (default 20)")
    args = ap.parse_args()

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
