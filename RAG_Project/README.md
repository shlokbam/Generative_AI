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
├── Test_splitter/            # ✂️ Ingestion Text Splitting Experiments
│   ├── char_split.py         # CharacterTextSplitter testing (notes.txt)
│   ├── token_split.py        # TokenTextSplitter testing (GRU.pdf)
│   └── semantic_split.py     # RecursiveCharacterTextSplitter testing (GRU.pdf)
│
├── Vector_Store/             # 🗄️ Vector Database Storage & Indexing
│   └── db.py                 # ChromaDB and MistralEmbeddings ingestion pipeline
│
├── chroma_db/                # [Local Only - Git Ignored] Persisted vector records
├── Big.pdf                   # Large test corpus PDF for RAG pipeline ingestion
├── .env                      # [Local Only] API keys and config (Mistral, HuggingFace, Groq)
├── Database.py               # 🚀 Main Vector Database build & ingestion pipeline
├── main.py                   # 💬 Clean LLM chat & prompt testing module
└── requirements.txt          # Ingestion, embedding, LLM, and vector store dependencies
```

---

## ⚡ Features & Architecture

### 1. Multi-Format Document Ingestion 📂
The ingestion pipeline is designed to process both structured, unstructured, and web-based inputs:
* **PyPDF Extraction (`pdf.py`)**: Demonstrates how to load complex PDF documents (`GRU.pdf`) into discrete, metadata-rich page objects using `PyPDFLoader`. It programmatically reads full documents and showcases how to extract specific index ranges (e.g., retrieving the concluding sections dynamically).
* **Plaintext Ingestion (`test.py`)**: Utilizes `TextLoader` to seamlessly convert standard `.txt` text corpora (`notes.txt`) into LangChain `Document` schemas ready for chunking and tokenization.
* **HTML/Web Scraping Ingestion (`page.py`)**: Leverages LangChain's `WebBaseLoader` (utilizing `BeautifulSoup4`) to load, fetch, and extract raw textual content from live web URLs (e.g., product detail pages) into document structures, preparing the ground for hybrid data-sources.

### 2. Document Chunking & Text Splitting ✂️
To maximize LLM context window efficiency and prepare content for semantic vector searches, we implemented three distinct chunking strategies:
* **Character-Based Splitting (`char_split.py`)**: Uses `CharacterTextSplitter` to divide text files by strict character lengths (`chunk_size=10`) with specified overlapping parameters.
* **Token-Based Splitting (`token_split.py`)**: Uses `TokenTextSplitter` powered by `tiktoken` to split documents precisely by token budgets (`chunk_size=1000`) rather than character bounds.
* **Recursive Character Splitting (`semantic_split.py` & `main.py`)**: Employs `RecursiveCharacterTextSplitter` as the primary RAG splitter (`chunk_size=1000`, `chunk_overlap=200`). This recursively splits on formatting marks (`\n\n`, `\n`, space, and empty string) to maintain semantic cohesion and prevent paragraphs from being split awkwardly.

### 3. Vector Database Storage, Embeddings Ingestion & Search 🗄️
To store chunks securely and enable fast, semantically accurate document retrieval, we integrated embedding models and an indexing layer:
* **Mistral AI Embeddings (`MistralAIEmbeddings`)**: Uses the state-of-the-art `mistral-embed` model to transform raw text chunks into dense, mathematical vector embeddings.
* **Chroma DB Ingest (`db.py`)**: Harnesses `Chroma` vector store to load text documents, calculate semantic vector embeddings, and save the indexed records locally inside a persistent folder (`RAG_Project/chroma_db/`).
* **Semantic Similarity Search**: Validated the retrieval layer by executing query-based similarity searches (`vectorstore.similarity_search("What is used for data analysis?", k=2)`), outputting clean, structured page contents and source metadata for top matching records.
* **Environment Integrity**: Confined the generated SQLite database binaries within local storage using Git ignore rules, keeping your version control repository lightweight and secure.

### 4. Vector Database Builder Pipeline (`Database.py`) 🚀
A dedicated end-to-end database constructor script:
* **Large Corpus Loading**: Ingests the `Big.pdf` document corpus using `PyPDFLoader`.
* **Standardized Splitting**: Divides the loaded document recursively using `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
* **Embeddings & Persistence**: Embeds the chunks using `MistralAIEmbeddings` (`mistral-embed`) and writes them directly into the persistent local store directory (`RAG_Project/chroma_db/`).

### 5. Simplified Prompt Testing Interface (`main.py`) 💬
A streamlined interface to quickly evaluate base model answers:
* **Mistral Integration**: Couples `python-dotenv` environment loaders with LangChain's `ChatMistralAI` to run base model inferences (`open-mistral-7b`).
* **Prompt Structuring**: Combines `ChatPromptTemplate` instructions dynamically to test basic prompt responsiveness.

### 6. Future-Proof Ingest System 🏗️
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

### 4. Run Text Splitting Tests ✂️

* **Test Character Splitter**:
  ```bash
  python RAG_Project/Test_splitter/char_split.py
  ```
  *Splits Deep Learning notes by character increments.*

* **Test Token Splitter**:
  ```bash
  python RAG_Project/Test_splitter/token_split.py
  ```
  *Splits GRU research paper strictly using token budgets.*

* **Test Recursive Character Splitter**:
  ```bash
  python RAG_Project/Test_splitter/semantic_split.py
  ```
  *Splits GRU research paper recursively using structural boundaries.*

### 5. Run Vector Database Ingestions & Similarity Queries 🗄️

* **Test ChromaDB Experimental Ingest & Retrieval**:
  ```bash
  python RAG_Project/Vector_Store/db.py
  ```
  *This runs experimental ingestions on raw documents, embeds them, and executes a similarity query.*

### 6. Run Official Vector Database Build Pipeline 🚀

* **Populate RAG Vector Store with Big.pdf**:
  ```bash
  python RAG_Project/Database.py
  ```
  *This reads the large Big.pdf document, chunks its pages recursively, generates embeddings, and constructs your persistent chroma_db vector store.*

### 7. Execute Prompt Testing Pipeline 💬

* **Run Simplified LLM Prompter**:
  ```bash
  python RAG_Project/main.py
  ```
  *This invokes the open-mistral-7b model using standard prompts to verify base connectivity.*

---

## 📈 Roadmap & Next Steps

- [x] Implement multi-format document loaders (`PyPDFLoader`, `TextLoader`, `WebBaseLoader`).
- [x] Configure LLM models (Mistral AI integration).
- [x] Design prompt structures for parsing extracted document pages.
- [x] **Chunking & Splitting**: Add `RecursiveCharacterTextSplitter` to optimize token usage.
- [x] **Vector Ingestion**: Embed document chunks using `MistralAIEmbeddings` and save them to local `ChromaDB`.
- [x] **Contextual Retrieval**: Implement semantic search query parsing to fetch only relevant chunks.
- [ ] **Hybrid Search / Re-ranking**: Optimize retrieval scores using advanced ranking strategies.
- [ ] **Conversational UI**: Add a Streamlit visual dashboard for multi-turn questions & answers.
