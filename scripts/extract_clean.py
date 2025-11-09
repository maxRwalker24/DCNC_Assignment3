import re
from pathlib import Path
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import html2text

RAW = Path("data/raw")
OUT = Path("data/intermediate")
OUT.mkdir(parents=True, exist_ok=True)

def normalize_text(s: str) -> str:
    s = s.replace('\u00ad', '')         # soft hyphen
    s = re.sub(r'-\n', '', s)           # hyphenated line breaks
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def pdf_to_text(p: Path) -> str:
    reader = PdfReader(str(p))
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        # Drop bare page numbers and typical header/footer noise
        t = re.sub(r'(?m)^\s*\d+\s*$', '', t)
        pages.append(t)
    return normalize_text("\n\n".join(pages))

def html_to_text(p: Path) -> str:
    html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    # Remove chrome
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript", "aside"]):
        tag.decompose()
    # Convert to markdown-like text (preserves headings/lists reasonably)
    md = html2text.html2text(str(soup))
    return normalize_text(md)

def main():
    for p in RAW.iterdir():
        if not p.is_file():
            continue
        out = OUT / (p.stem + ".txt")
        if p.suffix.lower() == ".pdf":
            text = pdf_to_text(p)
        elif p.suffix.lower() in {".html", ".htm"}:
            text = html_to_text(p)
        else:
            # Allow .txt direct saves if you copied content manually
            text = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))
        out.write_text(text, encoding="utf-8")
        print("wrote:", out)

if __name__ == "__main__":
    main()
