# Normas Brasil Chatbot 🤖

An interactive assistant built with **Streamlit** to query Brazilian technical standards (NRs) from PDF or HTML files.  
This project allows professionals and students to ask questions about any regulation (e.g., NR‑33, NR‑35) and get contextual answers directly from the document.

---

## 🚀 Purpose
The goal of this project is to:
- Provide **fast access** to Brazilian technical standards.  
- Allow professionals in occupational safety, engineers, and students to make **intelligent queries** without manually reading entire documents.  
- Serve as a **support tool** in training, audits, and studies.  

---

## ⚙️ How it works
1. The user provides a **PDF or HTML file** containing the regulation.  
2. The text is split into **chunks** and converted into **embeddings**.  
3. The system uses **FAISS** for semantic search.  
4. The language model receives the most relevant chunks (`top_k`) and generates a contextualized answer.  
5. The **Streamlit interface** simulates a chat for natural interaction.  

---

## 🖥️ System Requirements
Running large models on **CPU** requires many resources (mistralai/Mistral-7B-Instruct-v0.2).  
During tests, this project consumed **~40 GB of RAM** on CPU to load and execute a 7B parameter model.

### CPU vs GPU
- **CPU**:  
  - Slower for parallel computations.  
  - High RAM usage (~40 GB for 7B models).  
  - Longer response times.  

- **GPU**:  
  - Uses **VRAM**, optimized for matrix operations.  
  - The same model can run in **8–16 GB VRAM**, depending on quantization.  
  - Much faster responses.  

  ## 🧩 Example: qa.py

```python
import os
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Diretórios principais
EMBEDDINGS_DIR = "data/embeddings"
CHUNKS_DIR = "data/processed/chunks"
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def load_index():
    """Carrega embeddings, índice FAISS e lista de chunks."""
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
    index = faiss.read_index(os.path.join(EMBEDDINGS_DIR, "faiss.index"))
    with open(os.path.join(EMBEDDINGS_DIR, "chunks.txt"), "r", encoding="utf-8") as f:
        chunk_files = [line.strip() for line in f.readlines()]
    return embeddings, index, chunk_files

def semantic_search(query, top_k=5):
    """Busca semântica usando SentenceTransformer."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings, index, chunk_files = load_index()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunk_files):
            chunk_file = os.path.join(CHUNKS_DIR, chunk_files[idx])
            with open(chunk_file, "r", encoding="utf-8") as f:
                results.append(f.read().strip())
    return results

def answer_question(query, top_k=5, max_tokens=512):
    """Gera resposta contextualizada usando modelo de linguagem."""
    context_chunks = semantic_search(query, top_k=top_k)
    context = "\n\n".join(context_chunks)

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"   # usa GPU se disponível
    )
    nlp = pipeline("text-generation", model=model, tokenizer=tokenizer)

    prompt = f"""
Use o contexto abaixo para responder à pergunta de forma clara e objetiva.

Contexto:
{context}

Pergunta:
{query}

Resposta:
"""
    output = nlp(prompt, max_new_tokens=max_tokens, do_sample=False)
    resposta = output[0]["generated_text"].split("Resposta:")[-1].strip()

    # Salva log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"resposta_{timestamp}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Pergunta: {query}\n\nResposta:\n{resposta}\n")

    return resposta
---
```
## 🎯 O que esse exemplo mostra
- **Carregamento de embeddings e FAISS** para busca semântica.  
- **Uso automático de GPU** (`device_map="auto"`) se disponível.  
- **Pipeline de geração de texto** com HuggingFace Transformers.  
- **Log das respostas** em `logs/` para auditoria.  

👉 Summary: **CPU = accessible but heavy**, **GPU = optimized and fast**.

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/eduardomdalmaso/normas-brasil-chatbot.git
cd normas-brasil-chatbot
pip install -r env/requirements.txt
```
---

### ▶️ Usage

**Pipeline scripts**

The project includes a pipeline to process documents and run the assistant:

1. python src/preprocessing.py   # Cleans and prepares the text from PDF/HTML
2. python src/embeddings.py      # Generates embeddings and builds FAISS index
3. bash src/run.sh               # Launches Streamlit app with the model

Running everything at once

You can also run the full pipeline with:
python run_pipeline.py

This will execute:

    Preprocessing

    Embeddings generation

    Streamlit app launch

---

## 📂 Project Structure
```
emb_nrs/
├── data/
│   ├── embeddings          # embedding for model
│   ├── processed           # processed data
│   ├── raw                 # .html + .pdf data to clean+process
│
├── logs                    # saved answers
│
├── scripts/
│   ├── create_structure.sh # Handles conda enviroment
│   ├── run_pipeline.py     # Run model only without web interface
│   ├── run.sh              # Run full project
│
├── src/
│   ├── preprocessing.py   # Handles PDF/HTML parsing and cleaning
│   ├── embeddings.py      # Generates embeddings and FAISS index
│   ├── qa.py              # Question-answer logic
│   └── search.py          # Embedding + FAISS gen
│
├── streamlit_app/
│   ├── app.py             # Streamlit app

```

---

## 🌟 Future Improvements

GPU optimization for lower memory usage.

Highlighting the exact text chunks used in answers.

---

## 📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it for educational and professional purposes. See the LICENSE file for details.
