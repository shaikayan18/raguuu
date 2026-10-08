import os
import time
import hashlib
import hmac
import io
import json
import re
import secrets
from datetime import datetime
import importlib.util
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# On Streamlit Community Cloud, API keys come from the app's Secrets manager
# (st.secrets) rather than a local .env file. This bridges that into the
# environment so the rest of the code works unchanged whether running
# locally (.env) or deployed (Streamlit Cloud secrets).
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # No secrets.toml locally — that's fine, .env already covers local use

DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
CHAT_MODEL_NAME = "gemini-3.1-flash-lite"


def load_module_from_file(module_name, file_path):
    """Filenames starting with a digit (e.g. 1_ingestion_pipeline.py) can't use a
    normal `import` statement, so we load them dynamically instead. This keeps
    1_ingestion_pipeline.py and 3_answer_generation.py completely untouched."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the existing ingestion file as a reusable module (its functions only run when called)
ingestion = load_module_from_file("ingestion_pipeline", "1_ingestion_pipeline.py")


@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def load_existing_vectorstore():
    """Load the vector store from disk if it already exists and has data."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_metadata={"hnsw:space": "cosine"}
    )
    if vectorstore._collection.count() > 0:
        return vectorstore
    return None


def describe_page_with_gemini(image_bytes):
    """Send a rendered PDF page image to Gemini's vision model and get back
    a full transcription: regular text, tables reconstructed as markdown,
    and descriptions of any figures/charts/diagrams."""
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Transcribe everything on this page factually and completely:\n"
                "- All regular text, exactly as written\n"
                "- Any tables: reconstruct them as a markdown table with correct rows and columns\n"
                "- Any figures, charts, or diagrams: describe what they show in plain text\n"
                "No commentary, no summarizing — just the full factual content."
            )
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
        }
    ])
    result = model.invoke([message])

    return extract_text_from_gemini_response(result)


def extract_text_from_gemini_response(result):
    """Safely pull text out of a Gemini response, regardless of whether it
    comes back as a plain string, a list of content blocks, or an empty
    response (which can happen if a safety filter silently blocks content)."""
    content = result.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return "\n".join(text_parts)
        return "[No text returned for this page — it may have been blocked by a safety filter or contains no readable content.]"

    return "[Unexpected response format from the model for this page.]"

    if isinstance(result.content, list):
        return result.content[0]['text']
    return result.content


def describe_page_with_gemini_with_retry(image_bytes, max_retries=5):
    """Same as describe_page_with_gemini, but retries with backoff on rate
    limits — important on the free tier, where a page-heavy PDF can easily
    burst past the per-minute limit for the chat/vision model."""
    for attempt in range(max_retries):
        try:
            return describe_page_with_gemini(image_bytes)
        except Exception as e:
            error_text = str(e)
            if ("429" in error_text or "RESOURCE_EXHAUSTED" in error_text) and attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise


def load_pdf_with_vision(file_path, min_text_length=50, page_callback=None, vision_delay=3):
    """Extract text from each PDF page normally. For pages with little
    extractable text, or that contain images (often tables/figures), render
    that page and use Gemini's vision model to read it properly instead.

    On the free tier, vision_delay paces the calls to avoid bursting past
    the per-minute limit in the first place, and retry-with-backoff catches
    it if we hit the limit anyway."""
    import fitz  # PyMuPDF
    from langchain_core.documents import Document

    pdf = fitz.open(file_path)
    documents = []

    for page_num, page in enumerate(pdf, start=1):
        if page_callback:
            page_callback(page_num, len(pdf))

        native_text = page.get_text().strip()
        has_images = len(page.get_images()) > 0

        if len(native_text) >= min_text_length and not has_images:
            content = native_text
        else:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image_bytes = pix.tobytes("png")
            content = describe_page_with_gemini_with_retry(image_bytes)
            time.sleep(vision_delay)

        documents.append(Document(
            page_content=content,
            metadata={"source": file_path, "page": page_num}
        ))

    pdf.close()
    return documents


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image_as_document(file_path):
    """Load a standalone image file (not embedded in a PDF) by sending it
    straight to Gemini's vision model — the same transcription/description
    pipeline already used for image-heavy PDF pages, so a photo of a
    document, a screenshot, a chart, or a diagram all get read the same way
    and become fully searchable text."""
    from langchain_core.documents import Document

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    content = describe_page_with_gemini_with_retry(image_bytes)

    return [Document(
        page_content=content,
        metadata={"source": file_path}
    )]


def load_document_by_extension(file_path, page_callback=None):
    """Load a document. Supports .txt, .docx, .pdf (with vision fallback for
    tables/images/scans), and standalone image files (.jpg/.png/etc). Tries
    several encodings for .txt since Windows-generated text files aren't
    always plain UTF-8."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".txt":
        from langchain_community.document_loaders import TextLoader

        encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"]
        last_error = None

        for encoding in encodings_to_try:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                return loader.load()
            except Exception as e:
                last_error = e
                continue

        raise ValueError(f"Could not read file with any known encoding. Last error: {last_error}")

    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif extension == ".pdf":
        return load_pdf_with_vision(file_path, page_callback=page_callback)

    elif extension in IMAGE_EXTENSIONS:
        return load_image_as_document(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported: .txt, .docx, .pdf, .jpg, .jpeg, .png, .webp, .bmp")


def stamp_user_id(chunks, user_id):
    """Tag every chunk's metadata with the uploading user's ID.

    This is the entire basis of multi-user isolation in this app: there is
    only ONE shared ChromaDB collection on disk (one `db/chroma_db` folder),
    used by every visitor. Without this tag — and without filtering every
    search/count/delete by it afterward — any user could retrieve, count,
    or even wipe any other user's documents, since Chroma has no built-in
    concept of "owner" on its own."""
    for chunk in chunks:
        chunk.metadata["user_id"] = user_id
    return chunks


def count_user_chunks(vectorstore, user_id):
    """Count only the chunks belonging to this user, not the whole shared
    collection. Chroma's `.count()` has no filter, so we fetch the matching
    ids (metadata only, no embeddings/text) and count those instead."""
    if vectorstore is None:
        return 0
    result = vectorstore._collection.get(where={"user_id": user_id}, include=[])
    return len(result["ids"])


def ingest_uploaded_files(uploaded_files, user_id, progress_callback=None):
    """Save uploaded files into a per-user subfolder of docs/, then load and
    embed ONLY those new files. Every resulting chunk is tagged with user_id
    before embedding.

    Files are saved under docs/<user_id>/ rather than directly in docs/ —
    without this, two different users uploading a same-named file (e.g. both
    named "resume.pdf") would silently overwrite each other on disk before
    either one gets embedded."""
    user_docs_path = os.path.join(DOCS_PATH, user_id)
    os.makedirs(user_docs_path, exist_ok=True)

    new_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(user_docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        new_file_paths.append(file_path)

    documents = []
    failed_files = []
    for path in new_file_paths:
        try:
            if progress_callback:
                def page_callback(page_num, total_pages, _path=path):
                    progress_callback(0, 0, status=f"Reading {os.path.basename(_path)}: page {page_num}/{total_pages}...")
                documents.extend(load_document_by_extension(path, page_callback=page_callback))
            else:
                documents.extend(load_document_by_extension(path))
        except Exception as e:
            failed_files.append((os.path.basename(path), str(e)))

    if failed_files:
        for name, error in failed_files:
            st.warning(f"⚠️ Skipped {name}: {error}")

    if not documents:
        return st.session_state.vectorstore

    chunks = ingestion.split_documents(documents)

    if not chunks:
        st.warning("No extractable text found in the uploaded file(s) — it may be a scanned/image-only PDF or an empty document.")
        return st.session_state.vectorstore

    stamp_user_id(chunks, user_id)

    embedding_model = get_embedding_model()

    if progress_callback:
        progress_callback(0, len(chunks))

    vectorstore = embed_chunks_with_retry(
        chunks, embedding_model, st.session_state.vectorstore,
        progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback(len(chunks), len(chunks))

    return vectorstore


def embed_chunks_with_retry(chunks, embedding_model, existing_vectorstore, batch_size=10, max_retries=5, progress_callback=None):
    """Embed chunks in small batches with automatic retry-with-backoff on rate
    limits. Even on a paid tier, a burst of many chunks at once (like a large
    image-heavy PDF that needed vision on every page) can hit a temporary
    rate limit — this recovers instead of crashing the whole upload."""
    vectorstore = existing_vectorstore
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        if progress_callback:
            progress_callback(i, total, status=f"Embedding chunk {i + 1}-{min(i + batch_size, total)}/{total}...")

        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding_model,
                        persist_directory=PERSIST_DIRECTORY,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                error_text = str(e)
                if ("429" in error_text or "RESOURCE_EXHAUSTED" in error_text) and attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    if progress_callback:
                        progress_callback(i, total, status=f"Rate limited, waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
                else:
                    raise

        if i + batch_size < total:
            time.sleep(3)

    return vectorstore


class QueryVariations(BaseModel):
    """Alternative phrasings of the user's question, preserving its exact
    original intent — used to improve retrieval recall."""
    variations: List[str] = Field(
        description=(
            "3 to 4 alternative ways to ask the exact same question. "
            "Preserve the original meaning and intent precisely — do not "
            "introduce new topics, assumptions, or change what is being asked."
        )
    )


def generate_query_variations(query, max_retries=3):
    """Ask Gemini for a few alternative phrasings of the question. Different
    wording can match different chunks in the vector store that the user's
    exact phrasing alone might miss (e.g. "wage employment" vs "salaried job").

    This costs one extra Gemini call per question. If it fails for any reason
    (including a rate limit), we fall back to just the original query rather
    than blocking the whole answer on it."""
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)
    structured_model = model.with_structured_output(QueryVariations)

    prompt = (
        "Generate 3 to 4 alternative phrasings of this question. Preserve its "
        "exact original meaning and intent — do not change what is being "
        "asked or introduce new topics. If the question contains a specific "
        "name, person, file name, or proper noun, keep it EXACTLY as written "
        "in every variation — never replace it with a pronoun, a generic "
        "term, or drop it.\n\n"
        f"Question: {query}"
    )

    for attempt in range(max_retries):
        try:
            result = structured_model.invoke(prompt)
            return [v.strip() for v in result.variations if v.strip()][:4]
        except Exception as e:
            error_text = str(e)
            if ("429" in error_text or "RESOURCE_EXHAUSTED" in error_text) and attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return []  # Fall back to just the original query
    return []


def retrieve_with_multi_query(vectorstore, query, user_id, k_per_query=6):
    """Generate query variations (always including the original), search
    ChromaDB with each one, then combine and deduplicate the results.

    Every search is filtered to this user's own chunks only (filter=
    {"user_id": user_id}) — this is what actually prevents one user's
    question from ever retrieving another user's documents, regardless of
    how similar the content might be.

    Dedup key: chunk_id if the chunk has one (new chunks from the
    structure-aware splitter do); otherwise a content hash, so older chunks
    ingested before this upgrade still dedupe correctly instead of crashing
    or showing up as fake duplicates.

    When a chunk is returned by more than one query variant, we keep its
    highest similarity score — that's a meaningful signal it's genuinely
    relevant, not an artifact of one particular phrasing."""
    variations = generate_query_variations(query)

    all_queries = [query] + variations
    seen = set()
    unique_queries = []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_queries.append(q)

    candidates = {}  # dedup_key -> (doc, best_score)
    user_filter = {"user_id": user_id}

    for q in unique_queries:
        try:
            results = vectorstore.similarity_search_with_relevance_scores(q, k=k_per_query, filter=user_filter)
        except Exception:
            # Older Chroma versions / edge cases may not support relevance scores
            results = [(doc, None) for doc in vectorstore.similarity_search(q, k=k_per_query, filter=user_filter)]

        for doc, score in results:
            dedup_key = doc.metadata.get("chunk_id")
            if not dedup_key:
                dedup_key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

            if dedup_key not in candidates:
                candidates[dedup_key] = (doc, score)
            else:
                existing_doc, existing_score = candidates[dedup_key]
                if score is not None and (existing_score is None or score > existing_score):
                    candidates[dedup_key] = (doc, score)

    return list(candidates.values()), unique_queries


def rerank_with_bm25(query, candidates, top_n=8):
    """Rerank the candidate pool against the ORIGINAL user query using BM25
    keyword relevance (via rank_bm25 — a small, pure-Python library, no
    model download and no extra API calls).

    Why BM25 instead of a neural cross-encoder or an external rerank API:
    - It's genuinely lightweight: no torch/transformers install, no model
      download, installs in seconds.
    - No extra network calls, so it can't fail from a rate limit or add
      latency from an external API — important given how much this project
      has already had to work around free-tier limits.
    - It scores directly against the user's actual question (not the
      generated variations), which keeps the final answer grounded in what
      was really asked rather than diluted across multiple phrasings.
    A heavier cross-encoder model (e.g. sentence-transformers) or a hosted
    reranker (e.g. Cohere Rerank, which has a free trial tier with its own
    rate limits) would likely rerank slightly more accurately, but at real
    cost in install size, latency, or external dependency — BM25 is the
    practical choice for this app's scale."""
    from rank_bm25 import BM25Okapi

    if not candidates:
        return []

    docs = [doc for doc, _ in candidates]
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [doc for doc, _ in scored[:top_n]]


def condense_question_with_history(query, chat_history, max_turns=3, max_retries=3):
    """Rewrite a follow-up question into a standalone one, using recent chat
    history for context — e.g. "what about the second one?" becomes "what
    is the second essential quality of a successful entrepreneur listed by
    Dr. Kiran Mazumdar-Shaw?" based on the previous exchange.

    Only the last few turns are used (max_turns) to keep the prompt small —
    older context matters less for resolving "it"/"that"/"the second one"
    style references, and keeping history short avoids needlessly growing
    the token count (and therefore cost/latency) with every message.

    If there's no history yet, or the rewrite fails for any reason
    (including a rate limit), we fall back to the original question
    unchanged — this should never be the reason a question fails."""
    if not chat_history:
        return query

    recent = chat_history[-(max_turns * 2):]
    history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)

    prompt = f"""Given this recent conversation and a follow-up question, decide whether the follow-up genuinely depends on the conversation to be understood, or whether it introduces a new, different subject.

Rewrite ONLY if the follow-up question contains a vague reference that depends on the immediately preceding exchange to resolve — things like "it", "that", "this", "the second one", "him", "her", "the document" with no name attached. In that case, replace the vague reference with what it refers to.

Do NOT rewrite, and return the follow-up completely UNCHANGED, if:
- It mentions a specific name, document, file, or topic not already the exact subject of the last exchange (a topic change)
- It's already clear and understandable on its own
- You are not confident what the vague reference points to

When in doubt, prefer returning the question UNCHANGED — incorrectly carrying over old context to a new topic is worse than occasionally missing a legitimate follow-up.

Conversation so far:
{history_text}

Follow-up question: {query}

Standalone question:"""

    for attempt in range(max_retries):
        try:
            result = model.invoke([HumanMessage(content=prompt)])
            rewritten = extract_text_from_gemini_response(result).strip()
            return rewritten if rewritten else query
        except Exception as e:
            error_text = str(e)
            if ("429" in error_text or "RESOURCE_EXHAUSTED" in error_text) and attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return query  # Fall back to the original question
    return query


def answer_question(vectorstore, query, user_id, chat_history=None, k_per_query=6, final_k=8):
    """Retrieval + generation for one chat message.

    Pipeline: multi-query retrieval (original question + a few Gemini-generated
    rephrasings, each searched against ChromaDB — filtered to this user's own
    documents only — and deduplicated) -> BM25 reranking against the original
    question (narrows the combined candidate pool down to the most relevant
    chunks) -> those final chunks go to Gemini to generate the answer.

    This replaces the single-query MMR search used previously. Trade-off worth
    knowing: this does more work per question (one extra Gemini call to
    generate variations, plus up to ~5 similarity searches instead of 1), so
    each answer takes a bit longer and uses more of your API quota than
    before — in exchange for meaningfully better recall on questions where
    the exact wording doesn't closely match the source text.

    user_id (multi-user isolation): required. Every search below is scoped
    to only this user's own chunks, so this answer can never be grounded in
    another user's uploaded documents, however similar the topic.

    chat_history (history-aware generation): if the question is a follow-up
    that depends on earlier context ("what about the second one?"), it's
    rewritten into a standalone question first, using recent chat history.
    That standalone version is used for both retrieval and the final answer,
    so follow-ups actually find the right chunks instead of searching for
    the literal (context-free) words "the second one"."""
    standalone_query = condense_question_with_history(query, chat_history)

    total_chunks = count_user_chunks(vectorstore, user_id)

    # Scale search breadth to the size of THIS USER's document collection
    # (not the shared total across all users). A fixed small k_per_query
    # works fine for one document, but becomes a real recall problem once
    # someone has uploaded many — each query variant was only checking 6
    # chunks out of a now much larger, more diverse corpus, so the right
    # chunk could easily not make the cut before reranking even saw it.
    if total_chunks <= 100:
        k_per_query, final_k = 8, 10
    elif total_chunks <= 300:
        k_per_query, final_k = 10, 12
    elif total_chunks <= 600:
        k_per_query, final_k = 12, 15
    else:
        k_per_query, final_k = 15, 18

    k_per_query = min(k_per_query, max(total_chunks, 1))

    candidates, queries_used = retrieve_with_multi_query(vectorstore, standalone_query, user_id, k_per_query=k_per_query)
    relevant_docs = rerank_with_bm25(standalone_query, candidates, top_n=min(final_k, len(candidates)))

    combined_input = f"""Based on the following documents, please answer this question: {standalone_query}

Documents:
{chr(10).join([f"- [{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If information from multiple documents is relevant, address both. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)

    answer_text = extract_text_from_gemini_response(result)

    return answer_text, relevant_docs, standalone_query



# ---------- Password authentication ----------
# Accounts live in db/users.json as {user_id: {"salt": ..., "hash": ...}}.
# Passwords are never stored in plain text: each one is salted and hashed
# with PBKDF2-SHA256 (200k iterations), and checked with a constant-time
# comparison.
USERS_FILE = os.path.join("db", "users.json")
MIN_PASSWORD_LENGTH = 6


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    tmp_path = USERS_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(users, f)
    os.replace(tmp_path, USERS_FILE)  # atomic write


def hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    ).hex()


def create_account(user_id, password):
    users = load_users()
    salt_hex = secrets.token_hex(16)
    users[user_id] = {"salt": salt_hex, "hash": hash_password(password, salt_hex)}
    save_users(users)


def verify_password(user_id, password):
    record = load_users().get(user_id)
    if not record:
        return False
    candidate = hash_password(password, record["salt"])
    return hmac.compare_digest(candidate, record["hash"])



def _add_markdown_text(paragraph, text):
    """Add text to a paragraph, turning **bold** and *italic* into real Word formatting."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        elif part:
            paragraph.add_run(part)


def build_chat_docx(messages, user_id):
    """Build a Word (.docx) transcript of the chat and return it as bytes."""
    from docx import Document
    from docx.shared import Pt, RGBColor

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    doc.add_heading("RAG Chat Transcript", level=1)
    info = doc.add_paragraph()
    info.add_run(f"User: {user_id}\n").italic = True
    info.add_run(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}").italic = True

    for m in messages:
        is_user = m["role"] == "user"
        label = doc.add_paragraph()
        label.paragraph_format.space_before = Pt(12)
        label.paragraph_format.space_after = Pt(2)
        run = label.add_run("You" if is_user else "Assistant")
        run.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79) if is_user else RGBColor(0x2E, 0x7D, 0x32)

        for line in m["content"].split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            heading = re.match(r"^#{1,6}\s+(.*)", stripped)
            bullet = re.match(r"^[-*\u2022]\s+(.*)", stripped)
            numbered = re.match(r"^\d+[.)]\s+(.*)", stripped)
            if heading:
                _add_markdown_text(doc.add_paragraph(), f"**{heading.group(1)}**")
            elif bullet:
                _add_markdown_text(doc.add_paragraph(style="List Bullet"), bullet.group(1))
            elif numbered:
                _add_markdown_text(doc.add_paragraph(style="List Number"), numbered.group(1))
            else:
                _add_markdown_text(doc.add_paragraph(), stripped)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


# ---------- Streamlit UI ----------

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 RAG Chat")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_filenames" not in st.session_state:
    st.session_state.ingested_filenames = set()

# ---- Multi-user identity ----
# There is only ONE shared database on disk — every visitor to this app's
# URL connects to the same db/chroma_db folder. Without an identity here,
# any user's upload or question would mix with every other user's.
#
# This app has no real login system, so identity is a simple, user-chosen
# ID rather than verified authentication — enough to keep casual multi-user
# use (e.g. you and a colleague both using this app) properly separated,
# but NOT a security boundary: anyone who knows or guesses another
# person's ID could filter by it too. For genuinely private multi-tenant
# use, this ID should come from real auth instead of a text box.
#
# The ID can also come from the URL (?user=yourname), so each person can
# bookmark a personal link that auto-fills their ID instead of retyping it.
try:
    url_user = st.query_params.get("user", "")
except Exception:
    url_user = ""

st.subheader("Who's using this?")
raw_user_input = st.text_input(
    "Your name or ID (use the same one each time to see your own documents again)",
    value=url_user,
    placeholder="e.g. akhib, or your email",
)

user_id = raw_user_input.strip().lower().replace(" ", "_")

if not user_id:
    st.info("Enter a name or ID above to start — this keeps your documents separate from anyone else using this app.")
    st.stop()

# ---- Password gate ----
# The user must be logged in as THIS user_id before anything below runs.
# If they change the name box to a different ID, they have to log in again.
if st.session_state.get("authenticated_user") != user_id:
    users = load_users()
    is_new_user = user_id not in users

    if is_new_user:
        st.info(f"No account exists for **{user_id}** yet. Choose a password to create one.")
        with st.form("signup_form"):
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            signup_clicked = st.form_submit_button("Create account")

        if signup_clicked:
            if len(new_password) < MIN_PASSWORD_LENGTH:
                st.error(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                create_account(user_id, new_password)
                st.session_state.authenticated_user = user_id
                st.rerun()
    else:
        with st.form("login_form"):
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Log in")

        if login_clicked:
            if verify_password(user_id, password):
                st.session_state.authenticated_user = user_id
                st.rerun()
            else:
                time.sleep(1)  # slows down password guessing
                st.error("Incorrect password.")

    st.stop()

if url_user.strip().lower().replace(" ", "_") != user_id:
    st.caption(f"💡 Bookmark this link to come straight back as **{user_id}** next time: add `?user={user_id}` to this page's URL.")

st.divider()
st.subheader("Upload documents")
uploaded_files = st.file_uploader(
    "Upload .txt, .docx, .pdf, or image files (.jpg/.png) — they'll be embedded automatically",
    type=["txt", "docx", "pdf", "jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_filenames]

    if new_files:
        st.info(f"Found {len(new_files)} new file(s): {', '.join(f.name for f in new_files)}")
        progress_bar = st.progress(0, text="Starting ingestion...")

        def update_progress(batch_num, total_batches, status=None):
            if status:
                progress_bar.progress(0, text=status)
                return
            if total_batches == 0:
                progress_bar.progress(0, text="No content to embed...")
                return
            progress_bar.progress(
                batch_num / total_batches,
                text=f"Embedding batch {batch_num}/{total_batches}..."
            )

        chunks_before = count_user_chunks(st.session_state.vectorstore, user_id)

        with st.spinner(f"Ingesting {len(new_files)} file(s)..."):
            st.session_state.vectorstore = ingest_uploaded_files(
                new_files, user_id, progress_callback=update_progress
            )

        chunks_after = count_user_chunks(st.session_state.vectorstore, user_id)

        for f in new_files:
            st.session_state.ingested_filenames.add(f.name)

        progress_bar.empty()

        if chunks_after > chunks_before:
            st.success(f"✅ Ingested and embedded: {', '.join(f.name for f in new_files)}")
        else:
            st.warning(f"⚠️ Nothing was embedded from: {', '.join(f.name for f in new_files)} — check that the file(s) actually contain text.")

with st.sidebar:
    st.header("Status")
    st.caption(f"Signed in as: **{user_id}**")
    if st.button("Log out"):
        st.session_state.authenticated_user = None
        st.session_state.messages = []
        st.session_state.ingested_filenames = set()
        st.rerun()

    st.download_button(
        "⬇️ Download chat (.docx)",
        data=build_chat_docx(st.session_state.messages, user_id) if st.session_state.messages else b"",
        file_name=f"chat_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        disabled=not st.session_state.messages,
    )

    my_chunk_count = count_user_chunks(st.session_state.vectorstore, user_id)
    if my_chunk_count > 0:
        st.caption(f"✅ Your documents: {my_chunk_count} chunks")
    else:
        st.caption("⚠️ You haven't ingested any documents yet")

    if st.session_state.ingested_filenames:
        st.caption("Files you've ingested this session:")
        for name in st.session_state.ingested_filenames:
            st.caption(f"• {name}")

    st.divider()
    if st.button("🗑️ Clear MY documents", type="secondary", disabled=my_chunk_count == 0):
        try:
            # Deletes ONLY this user's chunks (filtered by user_id) — never
            # the whole shared collection. delete_collection() / rmtree() are
            # deliberately not used here anymore: on a shared database,
            # either one would wipe every other user's documents too.
            if st.session_state.vectorstore is not None:
                st.session_state.vectorstore._collection.delete(where={"user_id": user_id})

            st.session_state.ingested_filenames = set()
            st.session_state.messages = []
            st.success("Your documents have been cleared. Upload files to start fresh.")
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't clear your documents: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your documents..."):
    # Gate on THIS user's own chunk count, not just whether the shared
    # vectorstore object exists — the shared DB may already have other
    # users' documents in it while this user still has none of their own.
    if st.session_state.vectorstore is None or count_user_chunks(st.session_state.vectorstore, user_id) == 0:
        st.warning("Please upload and ingest your own documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # Pass everything before this question — not including it — as history
            history_for_this_turn = st.session_state.messages[:-1]

            with st.spinner("Thinking..."):
                answer, relevant_docs, standalone_query = answer_question(
                    st.session_state.vectorstore, query, user_id, chat_history=history_for_this_turn
                )
            st.markdown(answer)

            with st.expander("Sources"):
                if standalone_query.strip().lower() != query.strip().lower():
                    st.caption(f"*Understood as: \"{standalone_query}\"*")
                    st.divider()
                for i, doc in enumerate(relevant_docs, 1):
                    source_name = os.path.basename(doc.metadata.get('source', 'unknown'))
                    page = doc.metadata.get('page')
                    page_label = f" (page {page + 1 if isinstance(page, int) else page})" if page is not None else ""
                    st.caption(f"**Source {i}: {source_name}{page_label}**")
                    preview = doc.page_content[:400]
                    st.text(preview + ("..." if len(doc.page_content) > 400 else ""))
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})