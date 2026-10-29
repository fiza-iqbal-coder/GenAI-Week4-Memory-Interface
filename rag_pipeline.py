import os
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DB_PATH = "chroma_db"
COLLECTION_NAME = "week2_documents"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3

REFUSAL_MESSAGE = (
    "I cannot find the answer in the provided documents."
)

# This threshold is based on the distances observed
# in your current tests. Tune it for your dataset.
MAX_DISTANCE = 0.85

SYSTEM_PROMPT = f"""
You are a document-grounded question-answering assistant.

RULES:
1. Answer using ONLY the retrieved context.
2. Do not use outside knowledge or assumptions.
3. If the context does not contain enough information
   to answer the question, reply with exactly:
   {REFUSAL_MESSAGE}
4. Cite the source chunk for factual claims using
   the format [Chunk 1], [Chunk 2], etc.
5. Never invent information or source citations.
6. Keep answers clear and concise.
"""


# ==========================================
# 2. INITIALIZE COMPONENTS
# ==========================================

console = Console()


def initialize_system():
    """Initialize the embedding model, database, and Gemini."""

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Please configure it in your .env file."
        )

    console.print(
        "[cyan]Loading embedding model...[/cyan]"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    console.print(
        "[cyan]Connecting to ChromaDB...[/cyan]"
    )

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    console.print(
        "[cyan]Connecting to Gemini...[/cyan]"
    )

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    console.print(
        Panel(
            f"Ready!\n"
            f"Collection: {COLLECTION_NAME}\n"
            f"Indexed chunks: {collection.count()}\n"
            f"Gemini model: {GEMINI_MODEL}",
            title="RAG Pipeline",
            border_style="green"
        )
    )

    return embedding_model, collection, gemini_client


# ==========================================
# 3. RETRIEVE RELEVANT DOCUMENT CHUNKS
# ==========================================

def retrieve_chunks(question, embedding_model, collection):
    """Retrieve the top 3 relevant document chunks."""

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):

        metadata = metadatas[i] or {}

        chunk = {
            "id": i + 1,
            "text": document,
            "source": str(
                metadata.get("source", "Unknown source")
            ),
            "distance": distances[i]
        }

        chunks.append(chunk)

    return chunks


# ==========================================
# 4. DISPLAY RETRIEVED SOURCES
# ==========================================

def display_sources(chunks):
    """Display retrieved chunks in a colored table."""

    table = Table(title="Retrieved Sources")

    table.add_column(
        "Chunk",
        style="cyan",
        justify="center"
    )

    table.add_column(
        "Source",
        style="green"
    )

    table.add_column(
        "Distance",
        style="yellow",
        justify="right"
    )

    for chunk in chunks:
        table.add_row(
            str(chunk["id"]),
            chunk["source"],
            f'{chunk["distance"]:.4f}'
        )

    console.print(table)


# ==========================================
# 5. BUILD THE PROMPT
# ==========================================

def build_prompt(question, context):

    prompt = f"""
Answer the user's question using only the information
provided in the retrieved context.

The context and question are separated below.

<retrieved_context>
{context}
</retrieved_context>

<user_question>
{question}
</user_question>

Instructions:
- Use only the retrieved context.
- Cite factual claims using the relevant chunk ID.
- Do not invent citations.
- If the answer is not supported by the context,
  reply exactly:
  {REFUSAL_MESSAGE}
"""

    return prompt


# ==========================================
# 6. GENERATE AN ANSWER WITH GEMINI
# ==========================================

def generate_answer(question, context, gemini_client):
    """Generate a context-grounded answer with retry handling."""

    prompt = build_prompt(question, context)

    for attempt in range(3):

        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0
                }
            )

            answer = (response.text or "").strip()

            if not answer:
                return REFUSAL_MESSAGE

            return answer

        except Exception as error:

            error_message = str(error).upper()

            temporary_error = any(
                error_code in error_message
                for error_code in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED"
                ]
            )

            if not temporary_error or attempt == 2:
                raise

            wait_seconds = 2 ** (attempt + 1)

            console.print(
                f"[yellow]Temporary API error. "
                f"Retrying in {wait_seconds} seconds... "
                f"(Attempt {attempt + 2}/3)[/yellow]"
            )

            time.sleep(wait_seconds)

    return REFUSAL_MESSAGE


# ==========================================
# 7. BUILD CONTEXT FROM RETRIEVED CHUNKS
# ==========================================

def build_context(chunks):
    """Combine retrieved chunks into one context string."""

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"[Chunk {chunk['id']}]\n"
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


# ==========================================
# 8. PROCESS THE USER QUESTION
# ==========================================

def process_question(
    question,
    embedding_model,
    collection,
    gemini_client
):
    """Retrieve context and generate a grounded answer."""

    try:

        # Retrieve the top 3 chunks.
        chunks = retrieve_chunks(
            question,
            embedding_model,
            collection
        )

        if not chunks:
            console.print(
                Panel(
                    REFUSAL_MESSAGE,
                    title="Answer",
                    border_style="yellow"
                )
            )
            return

        # Display retrieved sources.
        display_sources(chunks)

        # Check whether the closest retrieved result
        # is sufficiently relevant.
        best_distance = min(
            chunk["distance"] for chunk in chunks
        )

        if best_distance > MAX_DISTANCE:

            console.print(
                Panel(
                    REFUSAL_MESSAGE,
                    title="Document-grounded answer",
                    border_style="yellow"
                )
            )
            return

        # Convert chunks into a single context string.
        context = build_context(chunks)

        # Generate the answer.
        answer = generate_answer(
            question,
            context,
            gemini_client
        )

        # Display the answer.
        console.print(
            Panel(
                answer,
                title="Document-grounded answer",
                border_style="green"
            )
        )

    except Exception as error:

        console.print(
            Panel(
                f"{error}",
                title="Question failed",
                border_style="red"
            )
        )


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

def main():

    try:

        (
            embedding_model,
            collection,
            gemini_client
        ) = initialize_system()

        console.print(
            "\n[bold green]Ask questions about your "
            "documents.[/bold green]"
        )

        console.print(
            "[dim]Type 'exit' to close the program.[/dim]\n"
        )

        while True:

            question = console.input(
                "[bold cyan]Question (or exit): [/bold cyan]"
            ).strip()

            if question.lower() == "exit":
                console.print(
                    "[yellow]RAG pipeline closed.[/yellow]"
                )
                break

            if not question:
                console.print(
                    "[yellow]Please enter a question.[/yellow]"
                )
                continue

            process_question(
                question,
                embedding_model,
                collection,
                gemini_client
            )

    except Exception as error:

        console.print(
            Panel(
                str(error),
                title="Initialization failed",
                border_style="red"
            )
        )


# ==========================================
# 10. RUN THE PROGRAM
# ==========================================

if __name__ == "__main__":
    main()
def rewrite_question_with_history(question, chat_history, client, model_name):
    """
    Convert a follow-up question into a standalone question
    using the previous conversation.
    """

    if not chat_history:
        return question

    recent_messages = chat_history[-6:]

    history_text = ""

    for message in recent_messages:
        role = message["role"]
        content = message["content"]

        history_text += f"{role}: {content}\n"

    prompt = f"""
You are a question rewriting assistant.

Use the conversation history to understand the latest question.

Rewrite the latest question as a complete, standalone question.

Do not answer the question.
Do not add new facts.
Return only the rewritten question.

Conversation history:
{history_text}

Latest question:
{question}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        rewritten_question = (response.text or "").strip()

        return rewritten_question if rewritten_question else question

    except Exception:
        return question