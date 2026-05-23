# Generative AI & LangChain Learning Journey 🚀

Welcome to my Generative AI and LangChain learning repository! This project tracks my progress, code implementations, and concepts as I follow along with this excellent YouTube Playlist:
👉 **[Generative AI & LangChain Course Playlist](https://www.youtube.com/playlist?list=PLaldQ9PzZd9oXR4PMGR4pr_DX4wFHkFwR)**

---

## 📅 Roadmap & Progress Tracker

- [x] **Video 1: Foundations of LangChain, LLM Integrations, Local Embeddings & Streamlit UIs**  
- [ ] **Video 2: (Upcoming)**
- [ ] **Video 3: (Upcoming)**
- [ ] **Video 4: (Upcoming)**
- [ ] **Video 5: (Upcoming)**

---

## 📁 Repository Structure

```text
Generative_AI/
│
├── Video_1/
│   ├── chat_models/
│   │   ├── chat.py                   # Initial Groq LLM integration
│   │   ├── Hugging_face.py           # LangChain + Hugging Face Endpoint
│   │   ├── chatbot.py                # CLI Interactive chatbot with memory
│   │   └── chatbot_ui.py             # Premium Streamlit Web Chatbot UI
│   │
│   ├── Embedding_models/
│   │   └── huggingface_embeddings.py # Local text embeddings with Sentence-Transformers
│   │
│   ├── intro.txt                     # Detailed summary of Video 1 tasks
│   └── requirements.txt              # Video 1 package dependencies
│
├── .env.example                      # Template for secure environment keys
├── .gitignore                        # Standard Python gitignore rules
└── README.md                         # Main repository index
```

---

## 🛠️ Video 1: Accomplished Tasks & Concepts

In **Video 1**, I completed the core foundational setup and successfully built:
* **Chat Integrations**: 
  * Hooked up **Groq Cloud API** using `llama-3.3-70b-versatile`.
  * Connected **Hugging Face Hub** using `meta-llama/Llama-3.3-70B-Instruct`.
* **State & Memory Management**:
  * Built a CLI chatbot leveraging LangChain message schemas (`SystemMessage`, `HumanMessage`, `AIMessage`) to maintain interactive conversation history.
* **Streamlit Web Application**:
  * Designed and deployed a beautiful Streamlit chatbot interface with sidebar customizations (Model selectors, creativity/temperature sliders, and live system prompt tuning).
* **Text Embeddings**:
  * Ran local mathematical text representations using `sentence-transformers/all-MiniLM-L6-v2` to convert text into 384-dimensional vector coordinates.
* **Core Concepts**:
  * Mastered **Prompt Templates** (`ChatPromptTemplate` for dynamic variables) and **Structured JSON Outputs** from LLMs.

---

## 🚀 Setup and Installation

### 1. Clone & Navigate
```bash
git clone <your-repository-url>
cd Generative_AI
```

### 2. Setup your Environment Variables
Duplicate the `.env.example` file and rename it to `.env`:
```bash
cp .env.example .env
```
Open your `.env` file and insert your actual API keys:
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 3. Install Dependencies
Ensure you are using your virtual environment and install the required modules:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 4. Run the Chatbot Web UI
Run the Streamlit application:
```bash
python -m streamlit run Video_1/chat_models/chatbot_ui.py
```
This will automatically open the application at `http://localhost:8501`.
