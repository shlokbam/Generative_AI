import os
# Force pure-Python protobuf implementation to resolve cloud runtime compatibility conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="RAG Insight Engine 🧠✨",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment keys
load_dotenv()

# Check for environment Mistral API Key (for fallback/local testing)
env_mistral_key = os.environ.get("MISTRAL_API_KEY")

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

    /* Dynamic Document Info Card */
    .doc-info-card {
        background: rgba(129, 140, 248, 0.04);
        border: 1px solid rgba(129, 140, 248, 0.18);
        border-radius: 16px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.5s ease;
    }

    .doc-info-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .doc-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }

    .doc-info-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }

    .doc-info-item:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(129, 140, 248, 0.2);
    }

    .doc-info-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .doc-info-value {
        font-size: 1rem;
        font-weight: 700;
        color: #f8fafc;
        word-break: break-all;
    }

    /* Empty state layout */
    .empty-state-container {
        text-align: center;
        padding: 4rem 2rem;
        background: rgba(255, 255, 255, 0.015);
        border: 2px dashed rgba(129, 140, 248, 0.15);
        border-radius: 24px;
        margin-top: 1.5rem;
        box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.6s ease;
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1.25rem;
        animation: pulseIcon 2.2s infinite ease-in-out;
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

    @keyframes pulseIcon {
        0% { transform: scale(1); opacity: 0.85; filter: drop-shadow(0 0 0px rgba(129,140,248,0)); }
        50% { transform: scale(1.06); opacity: 1; filter: drop-shadow(0 0 15px rgba(129,140,248,0.45)); }
        100% { transform: scale(1); opacity: 0.85; filter: drop-shadow(0 0 0px rgba(129,140,248,0)); }
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
st.sidebar.markdown("Configure authentication & search properties.")
st.sidebar.markdown("---")

# Section: User API Key input
st.sidebar.markdown("### 🔑 API Authentication")
user_api_key = st.sidebar.text_input(
    "Mistral API Key",
    type="password",
    value="",
    placeholder="Paste your MISTRAL_API_KEY...",
    help="Your API Key is processed securely in memory and is never saved on our servers. Get one at console.mistral.ai."
)

# Active API key evaluation (user text input overrides environment fallback)
active_key = user_api_key.strip() if user_api_key.strip() else env_mistral_key

# Section A: Retriever settings
st.sidebar.markdown("---")
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

# Display Document Status in Sidebar
st.sidebar.markdown("### 🗄️ Ingestion Status")
if "processed_file_name" in st.session_state and st.session_state.processed_file_name:
    st.sidebar.success(f"🟢 Active: {st.session_state.processed_file_name}")
else:
    st.sidebar.error("🔴 No Document Active")

# Section C: Reset Action
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# 4. Main App Layout & Header
st.markdown("<h1 class='hero-title'>RAG Insight Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Dynamic document-grounded intelligence powered by in-memory Vector DB & Mistral AI</p>", unsafe_allow_html=True)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# PDF Processing Pipeline function
def process_uploaded_pdf(uploaded_file, api_key):
    try:
        t_start = time.time()
        # Save file to a secure temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        num_pages = len(docs)
        
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except OSError:
            pass
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        num_chunks = len(chunks)
        
        # Build ephemeral Chroma Vector Store
        embeddings = MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
        t_duration = time.time() - t_start
        return vectorstore, num_pages, num_chunks, t_duration
    except Exception as e:
        st.error(f"Error parsing PDF: {str(e)}")
        return None, 0, 0, 0.0

# 5. Core Interface Flow
if not active_key:
    # Render premium API Key Required notice
    st.markdown("""
    <div style="background-color: rgba(129, 140, 248, 0.06); border: 1px solid rgba(129, 140, 248, 0.2); border-radius: 20px; padding: 2.5rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3); animation: fadeInUp 0.5s ease;">
        <div style="font-size: 3.5rem; margin-bottom: 1.25rem; animation: pulseKey 2s infinite ease-in-out;">🔑</div>
        <h3 style="color: #818cf8; margin-top: 0; font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; letter-spacing: -0.5px;">Mistral API Key Required</h3>
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6; max-width: 580px; margin: 0 auto 1.75rem auto; font-weight: 300;">
            To ingest documents and generate intelligent answers, please enter your personal <strong>Mistral API Key</strong> in the sidebar Control Panel. Your key remains safe, in-session, and is never shared or stored.
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem;">
            <a href="https://console.mistral.ai/" target="_blank" style="background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%); color: white; padding: 0.7rem 1.4rem; border-radius: 10px; text-decoration: none; font-weight: 600; font-size: 0.95rem; box-shadow: 0 4px 15px rgba(129, 140, 248, 0.35); transition: all 0.3s ease;">
                Get a Mistral API Key 🚀
            </a>
        </div>
    </div>
    <style>
        @keyframes pulseKey {
            0% { transform: scale(1); filter: drop-shadow(0 0 0px rgba(129,140,248,0)); }
            50% { transform: scale(1.08); filter: drop-shadow(0 0 12px rgba(129,140,248,0.4)); }
            100% { transform: scale(1); filter: drop-shadow(0 0 0px rgba(129,140,248,0)); }
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Render File Uploader when API key is active
    uploaded_file = st.file_uploader(
        "Drag & drop or browse for a PDF document to analyze",
        type=["pdf"],
        help="We will parse the file, calculate semantic vector embeddings, and prepare your context-grounded session."
    )

    # Handle Ingestion Logic on Upload State Switch
    if uploaded_file is not None:
        # If it is a completely new file or nothing has been loaded
        if st.session_state.processed_file_name != uploaded_file.name or st.session_state.vectorstore is None:
            with st.spinner("🧠 Initializing Ingestion: Parsing, chunking, and embedding document..."):
                vstore, pages, chunks, duration = process_uploaded_pdf(uploaded_file, active_key)
                
                if vstore is not None:
                    # Update Session State with details
                    st.session_state.vectorstore = vstore
                    st.session_state.processed_file_name = uploaded_file.name
                    
                    # Format file size nicely
                    size_bytes = uploaded_file.size
                    if size_bytes < 1024 * 1024:
                        formatted_size = f"{size_bytes / 1024:.1f} KB"
                    else:
                        formatted_size = f"{size_bytes / (1024 * 1024):.2f} MB"
                    
                    st.session_state.doc_info = {
                        "file_name": uploaded_file.name,
                        "file_size": formatted_size,
                        "pages": pages,
                        "chunks": chunks,
                        "duration": f"{duration:.2f}s"
                    }
                    
                    # Reset chat history for the fresh document
                    st.session_state.messages = []
                    st.toast("🎉 Document successfully ingested and embedded!", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
    else:
        # Reset State if the file is explicitly cleared/removed
        if st.session_state.processed_file_name is not None:
            st.session_state.vectorstore = None
            st.session_state.processed_file_name = None
            st.session_state.doc_info = None
            st.session_state.messages = []
            st.rerun()

    # 6. Check Active Vectorstore State to decide page layout
    if st.session_state.vectorstore is not None and st.session_state.doc_info is not None:
        # Render premium HTML Doc Info Card
        info = st.session_state.doc_info
        st.markdown(f"""
        <div class="doc-info-card">
            <div class="doc-info-header">
                <span>📄</span> Ingested Document Insight Card
            </div>
            <div class="doc-info-grid">
                <div class="doc-info-item">
                    <div class="doc-info-label">File Name</div>
                    <div class="doc-info-value">{info['file_name']}</div>
                </div>
                <div class="doc-info-item">
                    <div class="doc-info-label">File Size</div>
                    <div class="doc-info-value">{info['file_size']}</div>
                </div>
                <div class="doc-info-item">
                    <div class="doc-info-label">Page Count</div>
                    <div class="doc-info-value">{info['pages']} pages</div>
                </div>
                <div class="doc-info-item">
                    <div class="doc-info-label">Vector Chunks</div>
                    <div class="doc-info-value">{info['chunks']} semantic chunks</div>
                </div>
                <div class="doc-info-item">
                    <div class="doc-info-label">Ingestion Latency</div>
                    <div class="doc-info-value">{info['duration']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 7. Setup Dynamic Retriever & Generative LLM
        vectorstore = st.session_state.vectorstore
        
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

        # Generative LLM instantiation using active key
        llm = ChatMistralAI(
            mistral_api_key=active_key,
            model=llm_model,
            temperature=temperature
        )

        # Prompt Template definition
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

        # 8. Render Message History
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

            # Draw citations
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

        # 9. Query Input
        user_query = st.chat_input("Ask a question relevant to the uploaded document...")

        if user_query:
            # Render User message instantly
            st.markdown(f"""
            <div class="chat-bubble user-bubble">
                <div class="bubble-meta user-meta">User Query</div>
                <div class="bubble-body">{user_query}</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Run Pipelines
            with st.spinner("🧠 Querying ChromaDB and generating answer via Mistral AI..."):
                t_start = time.time()
                
                # Retrieval
                retrieved_docs = retriever.invoke(user_query)
                latency_retrieval = time.time() - t_start
                
                # Context mapping
                context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Generation
                formatted_prompt = prompt_template.format_messages(
                    context=context_string,
                    question=user_query
                )
                
                response = llm.invoke(formatted_prompt)
                answer_content = response.content

            # Render Assistant response
            st.markdown(f"""
            <div class="chat-bubble assistant-bubble">
                <div class="bubble-meta assistant-meta">RAG Assistant</div>
                <div class="bubble-body">{answer_content}</div>
            </div>
            """, unsafe_allow_html=True)

            # Map sources
            sources_list = [
                {"page_content": doc.page_content, "source": f"Page {doc.metadata.get('page', 0) + 1} of {st.session_state.processed_file_name}"}
                for doc in retrieved_docs
            ]

            # Render Citations expander
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

            # Add to memory
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_content,
                "sources": sources_list
            })

    else:
        # 10. Empty State Screen (Upload greeting)
        st.markdown("""
        <div class="empty-state-container">
            <div class="empty-state-icon">🧠✨</div>
            <h3 style="margin-bottom: 0.6rem; color: #818cf8; font-size: 1.6rem;">Upload a PDF to Start Chatting</h3>
            <p style="color: #94a3b8; font-size: 1rem; max-width: 580px; margin: 0 auto; line-height: 1.6; font-weight: 300;">
                Welcome to the RAG Insight Engine. To begin, drop a PDF research paper, article, or document into the uploader above. We will dynamically extract its pages, index them into an ephemeral vector store using your Mistral AI Key, and ground all answers directly in the text with precise page citations.
            </p>
        </div>
        """, unsafe_allow_html=True)
