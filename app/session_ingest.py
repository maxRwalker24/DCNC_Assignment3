import re
import io
from typing import List, Tuple
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

def _clean(s: str) -> str:
    s = s.replace('\u00ad', '')
    s = re.sub(r'-\n', '', s)
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    return s.strip()

def extract_text_from_uploads(files) -> List[str]:
    texts = []
    for f in (files or []):
        name = getattr(f, "name", "upload")
        try:
            data = f.read()
        except Exception:
            # Gradio may pass file-like object that supports .read()
            data = f
        if isinstance(data, (bytes, bytearray)):
            buf = io.BytesIO(data)
        else:
            # already raw bytes?
            buf = io.BytesIO(data if isinstance(data, (bytes, bytearray)) else b"")

        if name.lower().endswith(".pdf"):
            try:
                reader = PdfReader(buf)
                pages = [(page.extract_text() or "") for page in reader.pages]
                texts.append(_clean("\n\n".join(pages)))
            except Exception:
                texts.append("")
        else:
            # Treat as text
            try:
                texts.append(_clean(data.decode("utf-8", errors="ignore")))
            except Exception:
                texts.append("")
    return [t for t in texts if t]

def _chunk(text: str, limit=1000):
    sents = re.split(r'(?<=[.?!])\s+', text)
    buf, out = "", []
    for s in sents:
        if len(buf) + len(s) > limit and buf:
            out.append(buf.strip()); buf = s
        else:
            buf = (buf + " " + s).strip()
    if buf:
        out.append(buf)
    return out

class SessionMiniIndex:
    """A small, in-memory index for uploaded chunks using the base TF-IDF vectorizer."""
    def __init__(self, base_vectorizer):
        self.V = base_vectorizer
        self.chunks = []
        self.X = None

    def add_upload_texts(self, texts: List[str]):
        for i, t in enumerate(texts):
            for j, c in enumerate(_chunk(t)):
                self.chunks.append({"id": f"upload:{i}:{j}", "text": c})
        if self.chunks:
            corpus = [c["text"] for c in self.chunks]
            self.X = self.V.transform(corpus)

    def retrieve(self, query: str, top_k=3) -> List[Tuple[float, str, str]]:
        if not self.chunks or self.X is None:
            return []
        qv = self.V.transform([query])
        sims = cosine_similarity(qv, self.X).ravel()
        order = sims.argsort()[::-1][:top_k]
        out = []
        for idx in order:
            c = self.chunks[idx]
            out.append((float(sims[idx]), c["id"], c["text"]))
        return out
