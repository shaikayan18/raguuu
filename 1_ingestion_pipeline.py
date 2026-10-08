import os
import re
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def _is_heading(line):
    """Detect whether a line is likely a heading/section title, using
    deterministic pattern matching — no LLM call needed. Catches the common
    heading styles seen in real documents:
      - Markdown headers ("## Section Name")
      - Numbered headings ("1. Hard Work:")
      - ALL CAPS section titles ("NEED FOR ENTREPRENEURS")
      - Short Title Case standalone lines ("Introduction to Entrepreneurship")
    """
    line = line.strip()
    if not line or len(line) > 100:
        return False

    if re.match(r'^#{1,6}\s+\S', line):
        return True

    if re.match(r'^\d{1,2}[\.\)]\s+[A-Z][A-Za-z\s]{1,60}:', line):
        return True

    if re.match(r'^[A-Z0-9][A-Z0-9\s\-,\.:/&\']{4,}$', line) and any(c.isalpha() for c in line) and not line.endswith('.'):
        return True

    words = line.split()
    if 2 <= len(words) <= 10 and not line.endswith(('.', ',', ';')):
        capitalized = sum(1 for w in words if w[:1].isupper())
        if capitalized >= len(words) - 1:
            return True

    return False


def _split_into_sections(text):
    """Split raw text into (heading, content) sections at detected heading
    lines. Text with no headings at all comes back as a single section with
    heading=None, so plain prose documents still work exactly as before."""
    lines = text.split("\n")
    sections = []
    heading = None
    buffer = []

    for line in lines:
        if _is_heading(line):
            if buffer or heading:
                sections.append((heading, "\n".join(buffer).strip()))
            heading = line.strip()
            buffer = []
        else:
            buffer.append(line)

    sections.append((heading, "\n".join(buffer).strip()))
    return [(h, c) for h, c in sections if c or h]


def split_documents(documents, chunk_size=2500, chunk_overlap=200):
    """Split documents along meaningful semantic boundaries (headings,
    sections, paragraphs) instead of blind fixed-size slicing.

    WHY THIS APPROACH:
    - It's fully deterministic (regex-based heading detection), so it costs
      nothing extra in API calls or latency — no LLM is used to decide where
      to split, unlike "agentic chunking" approaches that call an LLM per
      document. Given how much rate-limiting pain this project already hit,
      adding more LLM calls into ingestion (which runs on every upload) would
      make the app slower and more fragile for no real accuracy benefit here.
    - Headings are a strong, reliable signal of semantic boundaries in real
      documents (textbooks, slides, reports) — splitting there naturally
      keeps a topic/section's content together instead of cutting it at an
      arbitrary character count.
    - Within a section, we still fall back to RecursiveCharacterTextSplitter
      if the section is too large for one chunk, so no single chunk ever
      exceeds a size that would hurt embedding/retrieval quality.
    - Plain prose with no headings (e.g. a company description) still works
      exactly like before — it becomes one "section" and gets split the same
      way as the previous version of this function.
    """
    print("Splitting documents into semantic chunks (heading/section-aware)...")

    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        sections = _split_into_sections(doc.page_content)

        for section_idx, (heading, content) in enumerate(sections):
            if not content:
                continue

            full_text = f"{heading}\n{content}" if heading else content

            if len(full_text) <= chunk_size:
                sub_texts = [full_text]
            else:
                sub_texts = sub_splitter.split_text(full_text)

            for sub_idx, text in enumerate(sub_texts):
                chunk_counter += 1
                metadata = dict(doc.metadata)
                metadata["section"] = heading if heading else "General"
                metadata["chunk_id"] = (
                    f"{os.path.basename(str(source))}"
                    f"_p{page if page is not None else 0}"
                    f"_s{section_idx}_c{sub_idx}_{chunk_counter}"
                )
                chunks.append(Document(page_content=text, metadata=metadata))

    print(f"Created {len(chunks)} semantic chunks from {len(documents)} document(s)")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Section: {chunk.metadata.get('section')}")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:")
        print(chunk.page_content[:300])
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store. With billing enabled, no
    rate-limit pacing is needed — Gemini's paid tier has much higher limits."""
    print("Creating embeddings and storing in ChromaDB (Gemini)...")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline (Gemini) ===\n")

    docs_path = "docs"
    persistent_directory = "db/chroma_db"

    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")

        embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore

    print("Persistent directory does not exist. Initializing vector store...\n")

    documents = load_documents(docs_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persistent_directory)

    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()