import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# 1. Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="RAG Insight Engine 🧠✨",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment keys
load_dotenv()

# 2. Premium Design System & Styles (Dark Mode, Neon Accents, Glassmorphism & Custom Animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* Global styling overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Main background: smooth dark radial space gradient */
    .stApp {
        background: radial-gradient(circle at 80% 20%, #1e1145 0%, #0c081d 50%, #04020a 100%);
        color: #f1f5f9;
    }

    /* Gradient typography for title */
    .hero-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 35%, #c084fc 70%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2.2rem;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* Sidebar elegant glass panel style */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 6, 24, 0.96);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Chat bubble visual design */
    .chat-bubble {
        padding: 1.25rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
        animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1);
        transition: all 0.3s ease;
    }

    .chat-bubble:hover {
        border-color: rgba(255, 255, 255, 0.14);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        transform: translateY(-1px);
    }

    .user-bubble {
        background: rgba(56, 189, 248, 0.12);
        border-left: 5px solid #38bdf8;
    }

    .assistant-bubble {
        background: rgba(129, 140, 248, 0.09);
        border-left: 5px solid #818cf8;
    }

    .bubble-meta {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .user-meta {
        color: #38bdf8;
    }

    .assistant-meta {
        color: #a78bfa;
    }

    .bubble-body {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #e2e8f0;
    }

    /* Citation document block */
    .citation-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        transition: all 0.25s ease;
    }

    .citation-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(129, 140, 248, 0.3);
    }

    .citation-title {
        font-weight: 600;
        color: #38bdf8;
        font-size: 0.9rem;
    }

    .citation-path {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
        word-break: break-all;
    }

    .citation-content {
        margin-top: 0.4rem;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
        white-space: pre-wrap;
    }

    /* Micro-animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Control Panel Configuration
st.sidebar.markdown("<h2 style='color: #818cf8; margin-top: 0.5rem;'>🛠️ Control Panel</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Tailor retrieval parameters & LLM configurations below.")
st.sidebar.markdown("---")

# Section A: Retriever settings
st.sidebar.markdown("### 🔍 Ingestion & Search")
search_strategy = st.sidebar.selectbox(
    "Search Algorithm",
    options=["mmr", "similarity"],
    index=0,
    help="MMR (Maximal Marginal Relevance) balances query relevance with context chunk diversity. Similarity retrieves purely by semantic closeness."
)

k_chunks = st.sidebar.slider(
    "Chunks to Retrieve (k)",
    min_value=1,
    max_value=12,
    value=4,
    step=1,
    help="Number of vector space fragments passed to the prompt context."
)

if search_strategy == "mmr":
    fetch_k = st.sidebar.slider(
        "Pool Candidates (fetch_k)",
        min_value=k_chunks,
        max_value=30,
        value=10,
        step=1,
        help="Initial list size of semantic match candidates before applying diversity filters."
    )
    lambda_mult = st.sidebar.slider(
        "Diversity Factor (lambda)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="1.0 is pure relevance; 0.0 is maximum diversity of returned chunks."
    )
else:
    fetch_k = 10
    lambda_mult = 0.5

# Section B: LLM parameters
st.sidebar.markdown("### 🤖 Generative LLM")
llm_model = st.sidebar.selectbox(
    "Mistral LLM Model",
    options=["mistral-small-latest", "open-mistral-7b"],
    index=0,
    help="Select the Mistral generative model for response synthesis."
)

temperature = st.sidebar.slider(
    "Creativity (Temperature)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Lower values make the output focused and highly deterministic based on the provided context."
)

st.sidebar.markdown("---")

# Display Vector Database Status
st.sidebar.markdown("### 🗄️ Database Directory")
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "chroma_db")
st.sidebar.code(persist_dir, language="bash")

if os.path.exists(persist_dir):
    st.sidebar.success("🟢 Vector DB Connected")
else:
    st.sidebar.error("🔴 Vector DB Not Found")

# Section C: Reset Action
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 4. Main App Layout & Header
st.markdown("<h1 class='hero-title'>RAG Insight Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Interactive conversational assistant powering context-grounded intelligence from your Vector DB</p>", unsafe_allow_html=True)

# Initialize messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache resources for database loading
@st.cache_resource
def load_chroma_db(path):
    if not os.path.exists(path):
        return None
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vectorstore = Chroma(
            persist_directory=path,
            embedding_function=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Failed to load database: {str(e)}")
        return None

# Load Vector Database
vectorstore = load_chroma_db(persist_dir)

if vectorstore is None:
    st.markdown(f"""
    <div style="background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem;">
        <h4 style="color: #f87171; margin-top: 0;">⚠️ Ingestion Pipeline Index Missing</h4>
        <p style="color: #fca5a5; margin-bottom: 1rem;">
            Chroma vector storage folder was not detected at: <code>{persist_dir}</code>.
            Please populate your database first by running your builder script:
        </p>
        <pre style="background-color: rgba(0, 0, 0, 0.4); border-radius: 6px; padding: 0.75rem; color: #f8fafc; font-family: monospace;">python RAG_Project/Database.py</pre>
    </div>
    """, unsafe_allow_html=True)
else:
    # 5. Build Dynamic Retriever Configuration
    if search_strategy == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_chunks,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k_chunks
            }
        )

    # Setup Generative LLM with chosen parameters
    llm = ChatMistralAI(
        model=llm_model,
        temperature=temperature
    )

    # Prompt Template matching main.py guidelines
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

                Use ONLY the provided context to answer the question.

                If the answer is not present in the context,
                say: "I could not find the answer in the document."
                
                Keep your answers highly detailed and professional, directly referencing details from the context.
                """
            ),
            (
                "human",
                """Context:
                {context}

                Question:
                {question}
                """
            )
        ]
    )

    # 6. Render Extant Message History
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        meta_class = "user-meta" if msg["role"] == "user" else "assistant-meta"
        meta_label = "User Query" if msg["role"] == "user" else "RAG Assistant"

        st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <div class="bubble-meta {meta_class}">{meta_label}</div>
            <div class="bubble-body">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Draw associated search citations if assistant message contains source history
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("🔍 Citations & Extracted Document Chunks"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"""
                    <div class="citation-card">
                        <span class="citation-title">Source Fragment #{idx+1}</span>
                        <div class="citation-path">📄 {src.get('source', 'ChromaDB Chunk')}</div>
                        <div class="citation-content">{src.get('page_content', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # 7. Chat Query Input
    user_query = st.chat_input("Enter your query about the ingested document...")

    if user_query:
        # User Message Visual Render & State Ingestion
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <div class="bubble-meta user-meta">User Query</div>
            <div class="bubble-body">{user_query}</div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Retrieval & Generation Pipelines
        with st.spinner("🧠 Querying ChromaDB and generating answer via Mistral AI..."):
            t_start = time.time()
            
            # Semantic search
            retrieved_docs = retriever.invoke(user_query)
            latency_retrieval = time.time() - t_start
            
            # Combine documents content
            context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # LLM Prompt Synthesis
            formatted_prompt = prompt_template.format_messages(
                context=context_string,
                question=user_query
            )
            
            # Generative LLM invocation
            response = llm.invoke(formatted_prompt)
            answer_content = response.content

        # Assistant Response Visual Render
        st.markdown(f"""
        <div class="chat-bubble assistant-bubble">
            <div class="bubble-meta assistant-meta">RAG Assistant</div>
            <div class="bubble-body">{answer_content}</div>
        </div>
        """, unsafe_allow_html=True)

        # Parse source items
        sources_list = [
            {"page_content": doc.page_content, "source": doc.metadata.get("source", "Unknown Source")}
            for doc in retrieved_docs
        ]

        # Draw brand new expandable citations drawer
        with st.expander("🔍 Citations & Extracted Document Chunks", expanded=False):
            st.caption(f"Retrieved {len(retrieved_docs)} fragments in {latency_retrieval:.3f}s via {search_strategy.upper()} strategy.")
            for idx, src in enumerate(sources_list):
                st.markdown(f"""
                <div class="citation-card">
                    <span class="citation-title">Source Fragment #{idx+1}</span>
                    <div class="citation-path">📄 {src['source']}</div>
                    <div class="citation-content">{src['page_content']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Append response and references to conversational memory state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_content,
            "sources": sources_list
        })
