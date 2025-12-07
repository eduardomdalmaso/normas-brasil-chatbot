import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = "data/processed/chunks"
EMBEDDINGS_DIR = "data/embeddings"

def generate_embeddings():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # Carrega modelo de embeddings (leve e eficiente)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = []
    file_names = []

    # Lê todos os chunks
    for filename in os.listdir(CHUNKS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(CHUNKS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
                    file_names.append(filename)

    print(f"Total de chunks carregados: {len(texts)}")

    # Gera embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Cria índice FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Salva embeddings e índice
    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.txt"), "w", encoding="utf-8") as f:
        for fn in file_names:
            f.write(fn + "\n")
    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "faiss.index"))

    print("Embeddings e índice FAISS salvos em:", EMBEDDINGS_DIR)

if __name__ == "__main__":
    generate_embeddings()