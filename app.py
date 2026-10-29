import os
import time
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DB_PATH = str(BASE_DIR / "chroma_db")

COLLECTION_NAME = "week2_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3

MAX_DISTANCE = 0.85

REFUSAL_MESSAGE = (
    "I cannot find the answer in the provided documents."
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
    }

    h1 {
        color: #243447;
        font-weight: 700;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .status-box {
        background-color: #e8f5e9;
        padding: 12px;
        border-radius: 10px;
        color: #166534;
        margin-bottom: 15px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. LOAD API CONFIGURATION
# ============================================================

def get_setting(name, default=None):
    """
    Read a setting from Streamlit Secrets first.
    If unavailable, read it from the local .env file.
    """

    try:
        value = st.secrets.get(name)

        if value:
            return value

    except (FileNotFoundError, KeyError):
        pass

    return os.getenv(name, default)


API_KEY = get_setting("GEMINI_API_KEY")

MODEL_NAME = get_setting(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

if not API_KEY:
    st.error(
        "Gemini API key not found. "
        "Configure GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()


# ============================================================
# 5. INITIALIZE GEMINI CLIENT
# ============================================================

@st.cache_resource
def initialize_gemini(api_key):
    """Initialize and cache the Gemini client."""
    return genai.Client(api_key=api_key)


try:
    gemini_client = initialize_gemini(API_KEY)

except Exception as error:
    st.error(f"Gemini initialization failed: {error}")
    st.stop()


# ============================================================
# 6. LOAD EMBEDDING MODEL AND CHROMADB
# ============================================================

@st.cache_resource
def load_resources(db_path):
    """
    Load the embedding model and ChromaDB collection.

    If the collection is empty, read sample_document.txt,
    split it into chunks, create embeddings, and store them.
    """

    # Load the sentence embedding model.
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    # Connect to the persistent ChromaDB database.
    chroma_client = chromadb.PersistentClient(
        path=db_path
    )

    # Get the collection or create it if it does not exist.
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Index the sample document if the collection is empty.
    if collection.count() == 0:

        document_path = BASE_DIR / "sample_document.txt"

        if not document_path.exists():
            raise FileNotFoundError(
                "sample_document.txt was not found. "
                "Upload it to the same directory as app.py."
            )

        with open(
            document_path,
            "r",
            encoding="utf-8"
        ) as file:
            text = file.read().strip()

        if not text:
            raise ValueError(
                "sample_document.txt is empty."
            )

        # Split the document into overlapping chunks.
        chunk_size = 1000
        overlap = 150

        chunks = []
        start = 0

        while start < len(text):

            chunk = text[
                start:start + chunk_size
            ].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        if not chunks:
            raise ValueError(
                "No document chunks could be created."
            )

        # Generate embeddings for all chunks.
        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=False
        ).tolist()

        # Store document chunks and embeddings.
        collection.add(
            ids=[
                f"document_chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": "sample_document.txt"}
                for _ in chunks
            ]
        )

    return embedding_model, collection


# IMPORTANT:
# Initialize both variables before using them elsewhere.

try:
    embedding_model, collection = load_resources(DB_PATH)

except Exception as error:
    st.error(
        "Could not load the embedding model or document database."
    )
    st.exception(error)
    st.stop()


# ============================================================
# 7. CONVERSATIONAL MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 8. GEMINI API CALL WITH RETRIES
# ============================================================

def call_gemini(prompt, max_attempts=2):
    """
    Send a prompt to Gemini.
    Retry once if the request fails.
    """

    for attempt in range(max_attempts):

        start_time = time.time()

        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "http_options": {
                        "timeout": 30000
                    }
                }
            )

            elapsed = time.time() - start_time

            print(
                f"Gemini response time: {elapsed:.2f} seconds"
            )

            answer = (response.text or "").strip()

            if answer:
                return answer

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        except Exception as error:

            elapsed = time.time() - start_time

            print(
                f"Gemini attempt {attempt + 1} failed "
                f"after {elapsed:.2f} seconds: {error}"
            )

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Gemini could not generate an answer. "
                    "Check the API configuration and app logs."
                ) from error

            time.sleep(1)


# ============================================================
# 9. REWRITE FOLLOW-UP QUESTIONS
# ============================================================

def rewrite_question(question, chat_history):
    """
    Convert a follow-up question into a standalone question.

    Example:
        Previous question: What is machine learning?
        Follow-up: What are its applications?

    Rewritten:
        What are the applications of machine learning?
    """

    if not chat_history:
        return question

    # Use recent conversation messages.
    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:
        role = message["role"]
        content = message["content"]

        history_text += f"{role}: {content}\n"

    prompt = f"""
You are a question-rewriting assistant.

Rewrite the latest user question into a complete,
standalone question using the conversation history.

Rules:
1. Use history only to understand references.
2. Resolve pronouns such as it, its, they, and them.
3. Do not answer the question.
4. Do not invent information.
5. Return only the rewritten question.
6. Preserve the meaning of the original question.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone question:
"""

    try:
        rewritten = call_gemini(prompt)

        return rewritten if rewritten else question

    except Exception as error:
        print(f"Question rewriting failed: {error}")
        return question


# ============================================================
# 10. RETRIEVE TOP 3 DOCUMENT CHUNKS
# ============================================================

def retrieve_documents(question):
    """
    Convert a question into an embedding and retrieve
    the three closest document chunks.
    """

    # Convert the question into a vector.
    question_embedding = embedding_model.encode(
        question,
        show_progress_bar=False
    ).tolist()

    total_chunks = collection.count()

    if total_chunks == 0:
        return [], [], []

    # Retrieve the closest chunks from ChromaDB.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, total_chunks),
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = results["documents"][0] or []
    distances = results["distances"][0] or []
    metadatas = results["metadatas"][0] or []

    return documents, distances, metadatas


# ============================================================
# 11. BUILD DOCUMENT CONTEXT
# ============================================================

def build_context(documents):
    """
    Combine retrieved chunks into a single context string.
    """

    context = ""

    for index, document in enumerate(
        documents,
        start=1
    ):
        context += (
            f"\n[Chunk {index}]\n"
            f"{document}\n"
        )

    return context


# ============================================================
# 12. GENERATE DOCUMENT-BASED ANSWER
# ============================================================

def generate_rag_answer(question, chat_history):
    """
    Conversational Retrieval-Augmented Generation (RAG).

    Steps:
    1. Rewrite the question if it is a follow-up.
    2. Retrieve relevant document chunks.
    3. Check the retrieval distance.
    4. Build the document context.
    5. Ask Gemini to answer using that context.
    """

    start_time = time.time()

    # STEP 1: Rewrite the follow-up question.
    rewrite_start = time.time()

    if chat_history:
        standalone_question = rewrite_question(
            question,
            chat_history
        )
    else:
        standalone_question = question

    rewrite_time = time.time() - rewrite_start

    # STEP 2: Retrieve relevant document chunks.
    retrieval_start = time.time()

    documents, distances, metadatas = retrieve_documents(
        standalone_question
    )

    retrieval_time = time.time() - retrieval_start

    # Prepare diagnostic information.
    details = {
        "original_question": question,
        "rewritten_question": standalone_question,
        "documents": documents,
        "distances": distances,
        "metadatas": metadatas,
        "rewrite_time": round(rewrite_time, 2),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": 0,
        "total_time": 0
    }

    # STEP 3: Check whether any chunks were retrieved.
    if not documents:

        details["total_time"] = round(
            time.time() - start_time,
            2
        )

        return REFUSAL_MESSAGE, details

    # STEP 4: Check the retrieval distance.
    # Lower cosine distance generally means greater similarity.
    if distances and min(distances) > MAX_DISTANCE:

        details["total_time"] = round(
            time.time() - start_time,
            2
        )

        return REFUSAL_MESSAGE, details

    # STEP 5: Build the context from retrieved documents.
    context = build_context(documents)

    # STEP 6: Generate an answer with Gemini.
    generation_start = time.time()

    prompt = f"""
You are a document-based question-answering assistant.

Answer the user's question using only information
supported by the document context below.

Rules:
1. Use simple, clear language.
2. Do not invent facts.
3. If the context does not contain enough information,
   respond exactly:
   {REFUSAL_MESSAGE}
4. When possible, mention the chunk number supporting
   your answer.
5. Conversation history can clarify the question, but
   factual answers must be supported by the document context.

Document context:
{context}

Question:
{standalone_question}

Answer:
"""

    answer = call_gemini(prompt)

    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    details["generation_time"] = round(
        generation_time,
        2
    )

    details["total_time"] = round(
        total_time,
        2
    )

    # Print performance information in the app logs.
    print("\n--- RAG PERFORMANCE ---")
    print(f"Question rewriting: {rewrite_time:.2f} seconds")
    print(f"Document retrieval: {retrieval_time:.2f} seconds")
    print(f"Answer generation: {generation_time:.2f} seconds")
    print(f"Total processing: {total_time:.2f} seconds")
    print("-----------------------\n")

    return answer, details


# ============================================================
# 13. DISPLAY RETRIEVAL DETAILS
# ============================================================

def display_retrieval_details(details):
    """
    Display the original question, rewritten question,
    retrieved chunks, distances, and performance times.
    """

    st.write(
        "**Original Question:**",
        details.get("original_question", "")
    )

    st.write(
        "**Rewritten Question:**",
        details.get("rewritten_question", "")
    )

    documents = details.get("documents", [])
    distances = details.get("distances", [])
    metadatas = details.get("metadatas", [])

    if documents:

        for index, document in enumerate(
            documents,
            start=1
        ):

            st.markdown(f"### Chunk {index}")
            st.write(document)

            distance_index = index - 1

            if distance_index < len(distances):
                st.write(
                    "**Distance:**",
                    round(distances[distance_index], 4)
                )

            if distance_index < len(metadatas):
                metadata = metadatas[distance_index]

                if metadata:
                    st.caption(
                        f"Source: {metadata.get('source', 'Unknown')}"
                    )

    else:
        st.info("No document chunks were retrieved.")

    st.divider()

    st.write(
        "**Question Rewriting Time:**",
        f"{details.get('rewrite_time', 0)} seconds"
    )

    st.write(
        "**Document Retrieval Time:**",
        f"{details.get('retrieval_time', 0)} seconds"
    )

    st.write(
        "**Answer Generation Time:**",
        f"{details.get('generation_time', 0)} seconds"
    )

    st.write(
        "**Total Processing Time:**",
        f"{details.get('total_time', 0)} seconds"
    )


# ============================================================
# 14. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 Document Assistant")

    st.markdown(
        "Your AI-powered document question-answering system."
    )

    st.divider()

    st.subheader("System Information")

    st.success("Conversational Memory Enabled")

    st.write("**LLM:** Gemini")

    st.write("**Vector Database:** ChromaDB")

    st.write("**Embedding Model:** all-MiniLM-L6-v2")

    st.write("**Retrieval:** Top 3 chunks")

    # The collection is initialized before this section.
    st.write(
        f"**Indexed Chunks:** {collection.count()}"
    )

    st.divider()

    st.subheader("Conversation")

    st.write(
        f"Messages: {len(st.session_state.messages)}"
    )

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Built for Uzair AI Studio "
        "Generative AI Engineering — Week 4"
    )


# ============================================================
# 15. MAIN CHAT INTERFACE
# ============================================================

st.title("📚 Document Q&A Assistant")

st.markdown(
    """
    <p class="subtitle">
    Ask questions about your documents and continue
    the conversation naturally.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="status-box">
    System Ready | Document Retrieval Active |
    Conversational Memory Enabled
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 16. DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show retrieval information for assistant responses.
        if (
            message["role"] == "assistant"
            and "retrieval" in message
        ):

            with st.expander("View Retrieval Details"):
                display_retrieval_details(
                    message["retrieval"]
                )


# ============================================================
# 17. CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# 18. PROCESS USER QUESTION
# ============================================================

if user_question:

    # Capture previous conversation before adding
    # the new question.
    previous_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ]

    # Save the user's message in session memory.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Display the new user message.
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate and display the assistant response.
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                answer, details = generate_rag_answer(
                    user_question,
                    previous_history
                )

                st.markdown(answer)

                # Display the retrieval information.
                with st.expander("View Retrieval Details"):
                    display_retrieval_details(details)

                # Save the assistant response and its details.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "retrieval": details
                    }
                )

            except Exception as error:

                st.error(
                    "An error occurred while generating the answer."
                )

                st.code(str(error))

                st.info(
                    "Check your Gemini API configuration, "
                    "internet connection, and application logs."
                )

                # Remove the unanswered user message.
                st.session_state.messages.pop()
