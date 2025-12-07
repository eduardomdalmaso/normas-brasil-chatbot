import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = "data/embeddings"
CHUNKS_DIR = "data/processed/chunks"

def load_index():
    # Carrega embeddings e índice FAISS
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss.index"))

    # Carrega nomes dos arquivos de chunks
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.txt"), "r", encoding="utf-8") as f:
        chunk_files = [line.strip() for line in f.readlines()]

    return embeddings, index, chunk_files

def search(query, top_k=5):
    # Carrega modelo
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Carrega índice
    embeddings, index, chunk_files = load_index()

    # Gera embedding da query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Busca no índice FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunk_files):
            chunk_file = os.path.join(CHUNKS_DIR, chunk_files[idx])
            with open(chunk_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            results.append({
                "rank": i+1,
                "distance": float(distances[0][i]),
                "file": chunk_files[idx],
                "text": text
            })
    return results

if __name__ == "__main__":
    # Exemplo de uso
    query = "Quais são as responsabilidades do supervisor de entrada na NR-33?"
    results = search(query, top_k=3)

    print(f"\n>>> Resultados para: {query}\n")
    for r in results:
        print(f"[{r['rank']}] {r['file']} (distância={r['distance']:.4f})")
        print(r["text"][:500], "...\n")