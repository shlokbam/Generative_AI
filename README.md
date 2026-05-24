# Generative AI & LangChain - Hands-On Codebase 🚀

A comprehensive repository of practical hands-on implementations, architectural modules, and complete codebases for Generative AI and LangChain concepts, structured around the following playlist:
👉 **[Generative AI & LangChain Course Playlist](https://www.youtube.com/playlist?list=PLaldQ9PzZd9oXR4PMGR4pr_DX4wFHkFwR)**

This repository serves as a centralized hub for all my completed implementations, modular integrations, and custom applications pushed for production-ready reference.

---

## 📅 Modules & Implementations Tracker

- [x] **Video 1: Foundations of LangChain, LLM Integrations, Local Embeddings & Streamlit UIs**  
- [x] **Video 2: RAG Project (Phase 1) - Multi-Format Ingestion, Document Loaders & Mistral AI Orchestration**  
- [ ] **Video 3: (Upcoming)**
- [ ] **Video 4: (Upcoming)**
- [ ] **Video 5: (Upcoming)**

---

## 📁 Repository Structure

```text
Generative_AI/
│
├── RAG_Project/                  # 🧠 Retrieval-Augmented Generation Hub
│   ├── document_loaders/
│   │   ├── GRU.pdf               # Research paper (Gated Recurrent Units)
│   │   ├── notes.txt             # Plaintext Deep Learning notes
│   │   ├── pdf.py                # PyPDFLoader integration
│   │   ├── test.py               # TextLoader integration
│   │   └── page.py               # WebBaseLoader URL content loader
│   │
│   ├── Test_splitter/            # ✂️ Ingestion Text Splitting Experiments
│   │   ├── char_split.py         # CharacterTextSplitter testing
│   │   ├── token_split.py        # TokenTextSplitter testing
│   │   └── semantic_split.py     # RecursiveCharacterTextSplitter testing
│   │
│   ├── Vector_Store/             # 🗄️ Vector Database Storage & Indexing
│   │   └── db.py                 # ChromaDB and MistralEmbeddings ingestion
│   │
│   ├── chroma_db/                # [Local Only - Git Ignored] Persisted vector records
│   ├── Big.pdf                   # Large test corpus PDF for RAG pipeline ingestion
│   ├── README.md                 # Detailed RAG project documentation & roadmap
│   ├── Database.py               # Main Vector Database build & ingestion pipeline
│   ├── main.py                   # Clean LLM chat & prompt testing module
│   └── requirements.txt          # Ingestion, embedding, LLM, and vector database packages
│
├── Video_1/                      # 🎥 Foundations & Streamlit Chatbot
│   ├── chat_models/
│   │   ├── chat.py                   # Groq LLM integration
│   │   ├── Hugging_face.py           # HF API Endpoint integration
│   │   ├── chatbot.py                # CLI Interactive chatbot with memory
│   │   └── chatbot_ui.py             # Streamlit chatbot web dashboard
│   │
│   ├── Embedding_models/
│   │   └── huggingface_embeddings.py # Local text embeddings (Sentence-Transformers)
│   │
│   ├── intro.txt                     # Detailed summary of Video 1 tasks
│   └── requirements.txt              # Video 1 package dependencies
│
├── .env.example                      # Template for secure environment keys
├── .gitignore                        # Standard Python gitignore rules
└── README.md                         # Main repository index (this file)
```

---

## 🛠️ Video 1: Completed Implementations & Concepts

The following functional units and configurations have been successfully implemented and verified:
* **Chat Integrations**: 
  * Hooked up **Groq Cloud API** using `llama-3.3-70b-versatile`.
  * Connected **Hugging Face Hub** using `meta-llama/Llama-3.3-70B-Instruct`.
* **State & Memory Management**:
  * Built an interactive CLI chatbot utilizing LangChain message schemas (`SystemMessage`, `HumanMessage`, `AIMessage`) to maintain persistent session history.
* **Streamlit Web Application**:
  * Designed and deployed a feature-rich Streamlit chatbot interface with sidebar customizations (Model selectors, creativity/temperature sliders, and live system prompt tuning).
* **Text Embeddings**:
  * Ran local mathematical text representations using `sentence-transformers/all-MiniLM-L6-v2` to convert text into 384-dimensional vector coordinates.
* **Core Concepts**:
  * Implemented modular **Prompt Templates** (`ChatPromptTemplate` for dynamic variables) and **Structured JSON Outputs** from LLMs.

---

## 🛠️ RAG Project (Phase 1): Ingestion, Splitting, Indexing & Orchestration

A specialized **Retrieval-Augmented Generation (RAG)** pipeline designed to load, partition, embed, index, and synthesize responses using:
* **Multi-Format Ingestion**: 
  * Integrated `PyPDFLoader` to load, parse, and partition mathematical research documents (e.g., `GRU.pdf` and `Big.pdf`) into discrete, metadata-rich page collections.
  * Integrated `TextLoader` to ingest unstructured plaintext assets (e.g., `notes.txt`) into memory-mappable document streams.
  * Integrated `WebBaseLoader` to pull and extract raw textual document streams directly from live website URLs.
* **Document Chunking & Splitting**:
  * Implemented character, token, and recursive text partitioners (`CharacterTextSplitter`, `TokenTextSplitter`, `RecursiveCharacterTextSplitter`).
  * Utilizes `RecursiveCharacterTextSplitter` inside the main orchestration pipeline to structure text data into standardized semantic chunks (`chunk_size=1000`, `chunk_overlap=200`).
* **Vector Database Ingestion, Indexing & Querying**:
  * Integrated `MistralAIEmbeddings` using the `mistral-embed` model to represent raw textual chunks as high-dimensional mathematical vector spaces.
  * Leveraged `Chroma` to persist, search, and manage document indexes locally inside `chroma_db/`.
  * Implemented and executed semantic similarity searches (`similarity_search`) returning clean, formatted page contents and source metadata for the top matching records.
* **Advanced Orchestration & Prompting**:
  * Added `Database.py` as the official, standalone vector database build pipeline which handles parsing `Big.pdf`, splitting chunks semantically, and indexing vectors inside local Chroma storage.
  * Streamlined `main.py` into a clean base LLM testing suite invoking ChatMistralAI (`open-mistral-7b`) with dynamic prompts to verify model answers.

---

## 🚀 Setup and Installation

### 1. Clone & Navigate
```bash
git clone <your-repository-url>
cd Generative_AI
```

### 2. Setup your Environment Variables
Create a `.env` file under both the root directory and/or the `RAG_Project` folder:
```bash
cp .env.example .env
```
Open the `.env` file and insert your API keys:
```env
GROQ_API_KEY=gsk_your_actual_key_here
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Install Dependencies & Run Applications

#### 📺 Video 1: Streamlit Chatbot
1. Install dependencies:
   ```bash
   pip install -r Video_1/requirements.txt
   ```
2. Start the interactive UI:
   ```bash
   python -m streamlit run Video_1/chat_models/chatbot_ui.py
   ```

#### 🧠 RAG Project (Phase 1)
1. Install dependencies:
   ```bash
   pip install -r RAG_Project/requirements.txt
   ```
2. Run document loaders test (PDF):
   ```bash
   python RAG_Project/document_loaders/pdf.py
   ```
3. Run document loaders test (Web scraping):
   ```bash
   python RAG_Project/document_loaders/page.py
   ```
4. Run text splitters test (Recursive Character):
   ```bash
   python RAG_Project/Test_splitter/semantic_split.py
   ```
5. Run vector store ingestion & similarity search test (ChromaDB):
   ```bash
   python RAG_Project/Vector_Store/db.py
   ```
6. Build and populate your RAG vector database (Big.pdf):
   ```bash
   python RAG_Project/Database.py
   ```
7. Run chatbot prompt testing:
   ```bash
   python RAG_Project/main.py
   ```
