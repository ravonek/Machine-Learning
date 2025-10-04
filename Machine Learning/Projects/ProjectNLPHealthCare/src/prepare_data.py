import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def build_index(csv_path: str, out_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'question' and 'answer' columns.")

    # Prepare corpus
    df["id"] = np.arange(len(df))
    texts = (df["question"].fillna("") + " \n " + df["answer"].fillna("")).tolist()

    # Embed
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors == inner product
    index.add(embeddings)

    # Save index + mapping
    faiss.write_index(index, os.path.join(out_dir, "medquad.index"))
    df[["id", "question", "answer", "source", "focus_area"]].to_parquet(os.path.join(out_dir, "medquad.parquet"), index=False)

    # Save meta
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"embeddings_model={model_name}\n")
        f.write(f"num_items={len(df)}\n")

    print("Index built:", index.ntotal, "items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to medquad.csv")
    parser.add_argument("--out", default="artifacts", help="Output directory for index and data")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    args = parser.parse_args()
    build_index(args.csv, args.out, args.model)
