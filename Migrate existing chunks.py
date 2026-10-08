"""
One-time migration: assign a user_id to every chunk currently in the
database that doesn't have one yet.

Why this is needed: multi-user support was added by tagging every chunk
with a "user_id" in its metadata, and filtering every search by it. Any
chunk ingested BEFORE that change has no user_id at all, so it won't match
anyone's filter — it would effectively become invisible in the app, even
though it's still sitting in the database.

This script finds every chunk with no user_id and assigns them all to one
user_id you choose (presumably yourself, since you're the one who uploaded
them originally). After running this once, those documents will show up
again under that user_id in the app.

Usage:
    python migrate_existing_chunks.py your_user_id
"""

import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_existing_chunks.py your_user_id")
        print('Example: python migrate_existing_chunks.py akhib')
        sys.exit(1)

    target_user_id = sys.argv[1].strip().lower().replace(" ", "_")

    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    all_data = vectorstore._collection.get(include=["metadatas"])

    ids_to_update = []
    metadatas_to_update = []

    for doc_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
        metadata = metadata or {}
        if "user_id" not in metadata:
            metadata["user_id"] = target_user_id
            ids_to_update.append(doc_id)
            metadatas_to_update.append(metadata)

    if not ids_to_update:
        print("Nothing to migrate — every chunk already has a user_id.")
        return

    print(f"Found {len(ids_to_update)} chunk(s) with no user_id.")
    print(f"Assigning them all to user_id = '{target_user_id}'...")

    vectorstore._collection.update(ids=ids_to_update, metadatas=metadatas_to_update)

    print(f"Done. {len(ids_to_update)} chunk(s) are now owned by '{target_user_id}'.")
    print(f"Open the app and sign in as '{target_user_id}' to see them again.")


if __name__ == "__main__":
    main()