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
Running large models on **CPU** requires many resources.  
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

👉 Summary: **CPU = accessible but heavy**, **GPU = optimized and fast**.

---

## 📦 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/eduardomdalmaso/normas-brasil-chatbot.git
cd normas-brasil-chatbot
pip install -r env/requirements.txt

▶️ Usage

Pipeline scripts

The project includes a pipeline to process documents and run the assistant:
python src/preprocessing.py   # Cleans and prepares the text from PDF/HTML
python src/embeddings.py      # Generates embeddings and builds FAISS index
bash src/run.sh               # Launches Streamlit app with the model

Running everything at once

You can also run the full pipeline with:
python run_pipeline.py

This will execute:

    Preprocessing

    Embeddings generation

    Streamlit app launch

📂 Project Structure
normas-brasil-chatbot/
│
├── src/
│   ├── preprocessing.py   # Handles PDF/HTML parsing and cleaning
│   ├── embeddings.py      # Generates embeddings and FAISS index
│   ├── qa.py              # Question-answer logic
│   └── run.sh             # Script to launch Streamlit app
│
├── app.py                 # Streamlit interface
├── run_pipeline.py        # Pipeline runner
├── env/requirements.txt   # Dependencies
├── data/                  # Example PDFs/HTMLs (optional)
└── README.md

🌟 Future Improvements
Support for multiple PDFs simultaneously.

GPU optimization for lower memory usage.

Highlighting the exact text chunks used in answers.

📜 License
This project is licensed under the MIT License. You are free to use, modify, and distribute it for educational and professional purposes. See the LICENSE file for details.
