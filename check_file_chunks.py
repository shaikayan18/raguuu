"""
Diagnostic script: check exactly what's stored in the database for a
specific file. Run this directly (not through Streamlit) to verify
whether a file actually ingested properly, how many chunks it produced,
and what content was actually extracted from it.

Usage:
    python check_file_chunks.py "use case diagram"

The search term is matched against the stored source filename, case-insensitive,
partial match — so "use case diagram" will match "use case diagram.pdf"
regardless of exact casing or extension.
"""

import sys
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_file_chunks.py \"filename or partial filename\"")
        sys.exit(1)

    search_term = sys.argv[1].lower()

    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"No database found at {PERSIST_DIRECTORY}")
        sys.exit(1)

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    total = vectorstore._collection.count()
    print(f"Total chunks in database: {total}\n")

    # Pull everything and filter client-side by filename match —
    # simplest reliable way to search regardless of how the path was stored
    all_data = vectorstore._collection.get(include=["metadatas", "documents"])

    matches = []
    for doc_id, metadata, content in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        source = str(metadata.get("source", "")).lower()
        if search_term in source:
            matches.append((doc_id, metadata, content))

    print(f"Chunks matching '{search_term}': {len(matches)}\n")
    print("=" * 60)

    if not matches:
        print("No chunks found for this file. This means either:")
        print("  1. The file was never actually uploaded/ingested, or")
        print("  2. It was uploaded under a different filename than expected, or")
        print("  3. It failed silently during ingestion (check for errors)")
        print("\nTip: run this script with just a few letters to search more broadly, e.g.:")
        print('    python check_file_chunks.py "use case"')
    else:
        for i, (doc_id, metadata, content) in enumerate(matches, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Source: {metadata.get('source')}")
            print(f"Page: {metadata.get('page', 'n/a')}")
            print(f"Section: {metadata.get('section', 'n/a')}")
            print(f"Chunk ID: {metadata.get('chunk_id', 'n/a')}")
            print(f"Content length: {len(content)} characters")
            print(f"Content preview:\n{content[:500]}")
            print("-" * 60)


if __name__ == "__main__":
    main()