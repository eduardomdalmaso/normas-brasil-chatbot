import os
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBEDDINGS_DIR = "data/embeddings"
CHUNKS_DIR = "data/processed/chunks"
LOGS_DIR = "logs"

# Garante que a pasta de logs exista
os.makedirs(LOGS_DIR, exist_ok=True)

def load_index():
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss.index"))
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.txt"), "r", encoding="utf-8") as f:
        chunk_files = [line.strip() for line in f.readlines()]
    return embeddings, index, chunk_files

def semantic_search(query, top_k=5):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings, index, chunk_files = load_index()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunk_files):
            chunk_file = os.path.join(CHUNKS_DIR, chunk_files[idx])
            with open(chunk_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            results.append(text)
    return results

def answer_question(query, top_k=5, max_tokens=2000, save_log=True):
    context_chunks = semantic_search(query, top_k=top_k)
    context = "\n\n".join(context_chunks)

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
        # load_in_4bit=True  # só se tiver GPU NVIDIA com bitsandbytes
    )
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = f"""
Use o contexto abaixo para responder à pergunta de forma clara e objetiva.

Pergunta:
{query}

Resposta:
"""
    output = nlp(prompt, max_new_tokens=max_tokens, do_sample=False)
    generated = output[0]["generated_text"]

    # 🔎 Pós-processamento: pega só o que vem depois de "Resposta:"
    if "Resposta:" in generated:
        resposta = generated.split("Resposta:")[-1].strip()
    else:
        resposta = generated.strip()

    # (opcional) salvar em log
    if save_log:
        from datetime import datetime
        import os
        LOGS_DIR = "logs"
        os.makedirs(LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"resposta_{timestamp}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Pergunta: {query}\n\n")
            f.write(f"Resposta:\n{resposta}\n")

    return resposta