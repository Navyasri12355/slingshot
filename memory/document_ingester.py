import os
import nltk
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

from onnx_embedder import ONNXEmbedder
from chroma_store import ChromaStore

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

embedder = ONNXEmbedder()
store = ChromaStore()

# ---------------------------
# Correct folder path
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "../demo/sample_docs"
)

# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks

# ---------------------------
# Ingestion
# ---------------------------
def ingest_folder(folder):
    docs = []
    ids = []
    embeddings = []
    metadatas = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        text = ""

        # ---------- TXT ----------
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        # ---------- PDF ----------
        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        else:
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            emb = embedder.embed(chunk)
            if emb is None or len(emb) == 0:
                continue

            docs.append(chunk)
            ids.append(f"{file}_{i}")
            embeddings.append(emb[0].tolist())

            # Ensure metadata is always correct
            metadatas.append({
                "source": file if file else f"unknown_file_{i}",
                "chunk": i
            })

    if len(docs) == 0:
        raise ValueError("No documents to ingest! Check sample_docs folder.")

    store.add_documents(
        ids,
        docs,
        embeddings,
        metadatas
    )

    print(f"Ingested {len(docs)} chunks from {len(os.listdir(folder))} files.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    ingest_folder(DATA_FOLDER)