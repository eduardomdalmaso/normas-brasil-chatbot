import os
import pymupdf  # PyMuPDF
from bs4 import BeautifulSoup

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHUNKS_DIR = "data/processed/chunks"

def extract_text_from_pdfs(pdf_path):
    doc = pymupdf.open(pdf_path)
    all_text = []
    for page in doc:
        text = page.get_text()
        all_text.append(text)
    return "\n".join(all_text)

def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts e estilos
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def generate_chunks(text, base_name, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        chunk_file = os.path.join(CHUNKS_DIR, f"{base_name}_chunk{idx+1}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"{len(chunks)} chunks gerados para {base_name}")

def process_raw_files():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for filename in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, filename)

        if filename.lower().endswith(".pdf"):
            print(f"Processando PDF: {filename}")
            text = extract_text_from_pdfs(path)

        elif filename.lower().endswith(".html"):
            print(f"Processando HTML: {filename}")
            text = extract_text_from_html(path)

        else:
            print(f"Ignorado (formato não suportado): {filename}")
            continue

        # Salva texto completo
        output_file = os.path.join(PROCESSED_DIR, filename.rsplit(".", 1)[0] + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Texto salvo em: {output_file}")

        # Gera chunks
        generate_chunks(text, filename.rsplit(".", 1)[0])

if __name__ == "__main__":
    process_raw_files()
