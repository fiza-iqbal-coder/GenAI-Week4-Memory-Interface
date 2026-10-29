# GenAI-Week4-Memory-Interface

A conversational Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, ChromaDB, and Google Gemini. The application answers questions using document-based information and maintains conversation history to support follow-up questions.

## Features

- **Document-Based Question Answering:** Retrieves relevant information from indexed documents to generate answers.
- **Conversational Memory:** Maintains conversation history during the current session.
- **Follow-Up Questions:** Uses previous conversation context to understand follow-up questions.
- **Vector Database:** Uses ChromaDB to store and retrieve document embeddings.
- **AI-Powered Responses:** Uses Google Gemini to generate natural-language answers.
- **Interactive Web Interface:** Provides a user-friendly chatbot interface built with Streamlit.
- **Relevant Document Retrieval:** Retrieves relevant document chunks to support answers.

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- Python-dotenv

## How It Works

1. The application loads the document and divides it into smaller chunks.
2. Sentence Transformers converts the text chunks into embeddings.
3. ChromaDB stores the embeddings for similarity-based retrieval.
4. The user submits a question through the Streamlit interface.
5. The application retrieves relevant document chunks.
6. Google Gemini generates an answer using the retrieved context.
7. Conversation history helps the chatbot interpret follow-up questions.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/fiza-iqbal-coder/GenAI-Week4-Memory-Interface.git
```

### 2. Open the Project Folder

```bash
cd GenAI-Week4-Memory-Interface
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API Key

Create a `.env` file in the project directory and add:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
```

Replace `your_gemini_api_key` with your own Google Gemini API key. Never upload your actual API key to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
GenAI-Week4-Memory-Interface/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── sample_document.txt
├── README.md
└── chroma_db/
```

*Note: The actual folder structure may vary depending on how the project is configured.*

## Learning Outcomes

- Implementing a Retrieval-Augmented Generation pipeline.
- Integrating Google Gemini with a Python application.
- Using ChromaDB for vector storage and document retrieval.
- Implementing session-based conversational memory.
- Building an interactive chatbot interface with Streamlit.
- Deploying a GenAI application.

## Live Application

**Streamlit App:** Add your deployed Streamlit URL here.

## GitHub Repository

https://github.com/fiza-iqbal-coder/GenAI-Week4-Memory-Interface
