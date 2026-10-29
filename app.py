
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

WEEK4_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEEK4_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "week2_documents"

import os
import time
from pathlib import Path

import streamlit as st
import chromadb

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

    # First check Streamlit Cloud secrets.
    try:
        value = st.secrets.get(name)

        if value:
            return value

    except (FileNotFoundError, KeyError):
        pass

    # Otherwise use the local .env file.
    return os.getenv(name, default)


API_KEY = get_setting("GEMINI_API_KEY")

MODEL_NAME = get_setting(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

if not API_KEY:

    st.error(
        "Gemini API key not found. "
        "Please configure GEMINI_API_KEY."
    )

    st.stop()


# ============================================================
# 5. INITIALIZE GEMINI CLIENT
# ============================================================

@st.cache_resource
def initialize_gemini(api_key):

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


    

@st.cache_resource
def load_resources(db_path):

    # Load the embedding model.
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    # Connect to ChromaDB.
    chroma_client = chromadb.PersistentClient(
        path=db_path
    )

    # Get the existing collection or create it.
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # If the collection is empty, load the sample document.
    if collection.count() == 0:

        document_path = BASE_DIR / "sample_document.txt"

        if not document_path.exists():
            raise FileNotFoundError(
                "sample_document.txt was not found."
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

        # Split the document into manageable chunks.
        chunk_size = 1000
        overlap = 150
        chunks = []
        start = 0

        while start < len(text):
            chunk = text[start:start + chunk_size].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        # Convert document chunks into embeddings.
        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=False
        ).tolist()

        # Save the chunks and embeddings in ChromaDB.
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
    Call Gemini with a timeout, limited retries,
    and performance logging.
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
            print(f"Gemini response time: {elapsed:.2f} seconds")

            answer = (response.text or "").strip()

            if answer:
                return answer

            raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:
            elapsed = time.time() - start_time

            print(
                f"Gemini attempt {attempt + 1} failed "
                f"after {elapsed:.2f} seconds: {error}"
            )

            if attempt == max_attempts - 1:
                raise RuntimeError(
                    "Gemini could not generate an answer. "
                    "Check the terminal for the detailed error."
                ) from error

            time.sleep(1)



# ============================================================
# 9. REWRITE FOLLOW-UP QUESTIONS
# ============================================================

def rewrite_question(question, chat_history):

    """
    Convert a follow-up question into a standalone question.

    Example:

    Previous question:
    What is machine learning?

    Follow-up:
    What are its applications?

    Rewritten:
    What are the applications of machine learning?
    """

    # No previous conversation.
    if not chat_history:
        return question

    # Use the most recent messages.
    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:

        role = message["role"]

        content = message["content"]

        history_text += f"{role}: {content}\n"

    prompt = f"""
You are a question rewriting assistant.

Your task is to rewrite the latest user question
into a complete standalone question.

RULES:

1. Use conversation history only to understand references.
2. Resolve pronouns such as it, its, they, them, and there.
3. Do not answer the question.
4. Do not invent new information.
5. Return only the rewritten question.
6. Preserve the meaning of the original question.

CONVERSATION HISTORY:
{history_text}

LATEST USER QUESTION:
{question}

STANDALONE QUESTION:
"""

    try:

        rewritten = call_gemini(prompt)

        return rewritten if rewritten else question

    except Exception:

        # If rewriting fails, use the original question.
        return question


# ============================================================
# 10. RETRIEVE TOP 3 DOCUMENT CHUNKS
# ============================================================

def retrieve_documents(question):

    """
    Convert the question into an embedding
    and retrieve the three closest chunks.
    """

    # Convert question into vector representation.
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    total_chunks = collection.count()

    if total_chunks == 0:
        return [], [], []

    # Search the existing ChromaDB collection.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, total_chunks),
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = results["documents"][0]

    distances = results["distances"][0]

    metadatas = results["metadatas"][0]

    return documents, distances, metadatas


# ============================================================
# 11. BUILD DOCUMENT CONTEXT
# ============================================================

def build_context(documents):

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
    Conversational RAG pipeline with timing information.
    Skips question rewriting for the first question.
    """

    import time

    start_time = time.time()

    # STEP 1: Rewrite only when there is previous conversation.
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

    # STEP 3: Check whether relevant documents were retrieved.
    if not documents:
        return (
            REFUSAL_MESSAGE,
            {
                "original_question": question,
                "rewritten_question": standalone_question,
                "documents": documents,
                "distances": distances,
                "metadatas": metadatas,
                "rewrite_time": round(rewrite_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(time.time() - start_time, 2),
            }
        )

    # STEP 4: Apply the existing distance threshold.
    if distances and min(distances) > MAX_DISTANCE:
        return (
            REFUSAL_MESSAGE,
            {
                "original_question": question,
                "rewritten_question": standalone_question,
                "documents": documents,
                "distances": distances,
                "metadatas": metadatas,
                "rewrite_time": round(rewrite_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "total_time": round(time.time() - start_time, 2),
            }
        )

    # STEP 5: Build the document context.
    context = build_context(documents)

    # STEP 6: Generate an answer using Gemini.
    generation_start = time.time()

    prompt = f"""
You are a document-based question-answering assistant.

Answer the user's question using only the information
supported by the document context below.

If the context does not contain enough information,
respond exactly:
{REFUSAL_MESSAGE}

Use simple, clear language. Do not invent facts.
Cite supporting chunks when possible.

Document context:
{context}

Question:
{standalone_question}

Answer:
"""

    answer = call_gemini(prompt)

    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    # STEP 7: Return the answer and diagnostic details.
    details = {
        "original_question": question,
        "rewritten_question": standalone_question,
        "documents": documents,
        "distances": distances,
        "metadatas": metadatas,
        "rewrite_time": round(rewrite_time, 2),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(generation_time, 2),
        "total_time": round(total_time, 2),
    }

    print("\n--- RAG PERFORMANCE ---")
    print(f"Question rewriting: {rewrite_time:.2f} seconds")
    print(f"Document retrieval: {retrieval_time:.2f} seconds")
    print(f"Answer generation: {generation_time:.2f} seconds")
    print(f"Total processing: {total_time:.2f} seconds")
    print("-----------------------\n")

    return answer, details



# ============================================================
# 13. SIDEBAR
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

    st.write("**Retrieval:** Top 3 chunks")

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
# 14. MAIN CHAT INTERFACE
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
# 15. DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display saved retrieval details.
        if (
            message["role"] == "assistant"
            and "retrieval" in message
        ):

            details = message["retrieval"]

            with st.expander("View Retrieval Details"):

                st.write(
                    "**Original Question:**",
                    details["original_question"]
                )

                st.write(
                    "**Rewritten Question:**",
                    details["rewritten_question"]
                )

                for index, document in enumerate(
                    details["documents"],
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {index}"
                    )

                    st.write(document)

                    if index <= len(details["distances"]):

                        st.write(
                            "Distance:",
                            round(
                                details["distances"][index - 1],
                                4
                            )
                        )


# ============================================================
# 16. CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about your documents..."
)


# ============================================================
# 17. PROCESS USER QUESTION
# ============================================================

if user_question:

    # Capture previous history before saving new question.
    previous_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ]

    # Save user message.
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    # Display user message.
    with st.chat_message("user"):

        st.markdown(user_question)

    # Generate assistant response.
    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:

                answer, details = generate_rag_answer(
                    user_question,
                    previous_history
                )

                st.markdown(answer)

                # Display retrieval details.
                with st.expander("View Retrieval Details"):

                    st.write(
                        "**Original Question:**",
                        details["original_question"]
                    )

                    st.write(
                        "**Rewritten Question:**",
                        details["rewritten_question"]
                    )

                    if details["documents"]:

                        for index, document in enumerate(
                            details["documents"],
                            start=1
                        ):

                            st.markdown(
                                f"### Chunk {index}"
                            )

                            st.write(document)

                            if index <= len(details["distances"]):

                                st.write(
                                    "Distance:",
                                    round(
                                        details["distances"][index - 1],
                                        4
                                    )
                                )

                    else:

                        st.info(
                            "No document chunks were retrieved."
                        )

                # Save successful assistant response.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "retrieval": details
                })

            except Exception as error:

                st.error(
                    "An error occurred while generating the answer."
                )

                st.code(str(error))

                st.info(
                    "Please check your Gemini API configuration "
                    "and internet connection, then try again."
                )

                # Remove the unanswered user question so
                # it does not confuse future follow-ups.
                st.session_state.messages.pop()
