# RAG Project: Retrieval-Augmented Generation Hub 🧠🚀

Welcome to the **Retrieval-Augmented Generation (RAG) Project** folder. This module is a comprehensive, production-ready implementation of a complete RAG system using **LangChain**, modern Large Language Models (LLMs), local embedding models, and robust document loaders.

Currently, the project implements the core document parsing, structured ingestion pipelines, and LLM-powered orchestration layers.

---

## 📂 Project Structure

```text
RAG_Project/
│
├── document_loaders/
│   ├── GRU.pdf               # Sample research paper (Gated Recurrent Units)
│   ├── notes.txt             # Plaintext sample notes (Deep Learning foundations)
│   ├── pdf.py                # Isolated PyPDFLoader test suite
│   ├── test.py               # Isolated TextLoader test suite
│   └── page.py               # WebBaseLoader URL content loader
│
├── .env                      # [Local Only] API keys and config (Mistral, HuggingFace, Groq)
├── main.py                   # Main RAG orchestration & summarization pipeline
└── requirements.txt          # Ingestion, embedding, LLM, and vector store dependencies
```

---

## ⚡ Features & Architecture

### 1. Multi-Format Document Ingestion 📂
The ingestion pipeline is designed to process both structured, unstructured, and web-based inputs:
* **PyPDF Extraction (`pdf.py`)**: Demonstrates how to load complex PDF documents (`GRU.pdf`) into discrete, metadata-rich page objects using `PyPDFLoader`. It programmatically reads full documents and showcases how to extract specific index ranges (e.g., retrieving the concluding sections dynamically).
* **Plaintext Ingestion (`test.py`)**: Utilizes `TextLoader` to seamlessly convert standard `.txt` text corpora (`notes.txt`) into LangChain `Document` schemas ready for chunking and tokenization.
* **HTML/Web Scraping Ingestion (`page.py`)**: Leverages LangChain's `WebBaseLoader` (utilizing `BeautifulSoup4`) to load, fetch, and extract raw textual content from live web URLs (e.g., product detail pages) into document structures, preparing the ground for hybrid data-sources.

### 2. Intelligent Prompts & Orchestration (`main.py`) 🔗
The main entry point ties document ingestion and model invocation together:
* **Environment Synchronization**: Integrates `dotenv` to load environment keys without exposing credentials in the codebase.
* **Low-Latency LLM Pipeline**: Harnesses `ChatMistralAI` using the high-throughput `open-mistral-7b` model to evaluate loaded text structures.
* **Custom Prompt Templating**: Features a dynamic `ChatPromptTemplate` that structures the instructions for the LLM (`"You are a AI that summarizs the text"`) and feeds in parsed document pages dynamically.

### 3. Future-Proof Ingest System 🏗️
The project dependencies are architected to support future RAG phases:
* **Embeddings**: Prepared for HuggingFace local models via `sentence-transformers` and `langchain-huggingface`.
* **Vector Store**: Pre-configured for high-speed local storage and search using `chromadb`.
* **App/UI Layer**: Dependencies include `fastapi`, `uvicorn`, and `streamlit` to quickly expose APIs and interactive web consoles.

---

## 🛠️ Installation & Execution

### 1. Prerequisite Environment Setup
Make sure you have set up your `.env` file in the `RAG_Project` directory with the following variables:
```env
GROQ_API_KEY="your_groq_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
MISTRAL_API_KEY="your_mistral_api_key"
```

### 2. Install Dependencies
Run the installation in your active virtual environment:
```bash
pip install -r RAG_Project/requirements.txt
```

### 3. Run Document Loading Tests

* **Test PDF Loader**:
  ```bash
  python RAG_Project/document_loaders/pdf.py
  ```
  *This loads the PDF document and prints out the text contents of the last page.*

* **Test Text Loader**:
  ```bash
  python RAG_Project/document_loaders/test.py
  ```
  *This loads the plaintext Deep Learning notes and displays the parsed content in the terminal.*

* **Test Web/Page Loader**:
  ```bash
  python RAG_Project/document_loaders/page.py
  ```
  *This targets a live URL, scrapes the webpage utilizing WebBaseLoader, and prints the raw page content.*

### 4. Execute Main RAG Pipeline
Run the main entrypoint to load the PDF, feed the concluding section into the Mistral model, and generate a dynamic summary:
```bash
python RAG_Project/main.py
```

---

## 📈 Roadmap & Next Steps

- [x] Implement multi-format document loaders (`PyPDFLoader`, `TextLoader`, `WebBaseLoader`).
- [x] Configure LLM models (Mistral AI integration).
- [x] Design prompt structures for parsing extracted document pages.
- [ ] **Chunking & Splitting**: Add `RecursiveCharacterTextSplitter` to optimize token usage.
- [ ] **Vector Ingestion**: Embed document chunks using `sentence-transformers` and save them to `ChromaDB`.
- [ ] **Contextual Retrieval**: Implement semantic search query parsing to fetch only relevant chunks.
- [ ] **Hybrid Search / Re-ranking**: Optimize retrieval scores using advanced ranking strategies.
- [ ] **Conversational UI**: Add a Streamlit visual dashboard for multi-turn questions & answers.
