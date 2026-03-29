"""
Ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

Run once before starting the API server, and re-run whenever the PDF changes.
Usage: python ingest.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def ingest() -> None:
    # --- Step 1: Load PDF ---
    pdf_path = Path(settings.pdf_path)
    if not pdf_path.exists():
        sys.exit(f"ERROR: PDF not found at {pdf_path.resolve()}")

    print(f"Loading PDF: {pdf_path.resolve()}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    # --- Step 2: Chunk ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["regulation"] = "29 CFR § 1910.1030"

    # --- Step 3: Embed + Store ---
    print("Embedding and storing in ChromaDB...")
    embedding = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    chroma = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding,
        persist_directory=settings.chroma_persist_dir,
    )

    # Clear existing vectors to prevent duplicates on re-runs
    existing = chroma.get()
    if existing["ids"]:
        print(f"  Clearing {len(existing['ids'])} existing vectors")
        chroma.delete(ids=existing["ids"])

    chroma.add_documents(chunks)
    print(f"  Stored {len(chunks)} chunks in collection '{settings.chroma_collection_name}'")
    print(f"  ChromaDB persisted at: {Path(settings.chroma_persist_dir).resolve()}")
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
