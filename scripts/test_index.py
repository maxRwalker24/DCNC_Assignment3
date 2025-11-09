# scripts/test_index.py
import argparse, json
from pathlib import Path
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def load_kb(kb_path: Path):
    recs = []
    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main():
    ap = argparse.ArgumentParser(description="Test TF-IDF index with a query")
    ap.add_argument("--kb", default="data/clean/kb.jsonl")
    ap.add_argument("--vec", default="data/clean/vectorizer.pkl")
    ap.add_argument("--mat", default="data/clean/matrix.npz")
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5, help="Top K passages")
    args = ap.parse_args()

    kb = load_kb(Path(args.kb))
    vectorizer = joblib.load(args.vec)
    X = sparse.load_npz(args.mat)

    q_vec = vectorizer.transform([args.q])
    sims = cosine_similarity(q_vec, X)[0]
    top_idx = sims.argsort()[::-1][:args.k]

    print(f"Query: {args.q}\n")
    for rank, idx in enumerate(top_idx, start=1):
        rec = kb[idx]
        print(f"[{rank}] score={sims[idx]:.4f} src={Path(rec['source']).name} "
              f"cat={rec['category']} chunk={rec['passage_index']}")
        print(rec["text"][:500].replace("\n", " ") + ("..." if len(rec["text"])>500 else ""))
        print("-" * 80)

if __name__ == "__main__":
    main()
