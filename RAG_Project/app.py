import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Page configurations
st.set_page_config(
    page_title="RAG AI Assistant 🧠✨",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Mode, Glassmorphism, and Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    /* Font configuration */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Main background & glassmorphic containers */
    .stApp {
        background: radial-gradient(circle at top right, #1a103c 0%, #0d091b 60%, #050409 100%);
        color: #f3f4f6;
    }
    
    /* Elegant gradient title */
    .gradient-text {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 30%, #a18cd1 70%, #fbc2eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        animation: textShimmer 4s ease infinite alternate;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Custom Chat Bubbles */
    .chat-bubble {
        padding: 1.2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.4s ease-out;
        transition: all 0.3s ease;
    }
    
    .chat-bubble:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .user-bubble {
        background: rgba(79, 172, 254, 0.15);
        border-left: 5px solid #4facfe;
    }
    
    .assistant-bubble {
        background: rgba(161, 140, 209, 0.12);
        border-left: 5px solid #a18cd1;
    }
    
    .bubble-header {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .user-header {
        color: #60a5fa;
    }
    
    .assistant-header {
        color: #c084fc;
    }
    
    .bubble-body {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e2e8f0;
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 9, 27, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Source Metadata Cards */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    .source-title {
        font-weight: 600;
        color: #38bdf8;
    }

    /* Keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Load environment keys
load_dotenv()

# Sidebar Setup
st.sidebar.markdown("<h2 style='color: #4facfe; margin-bottom: 0.5rem;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Fine-tune your document retrieval and generative parameters.")
st.sidebar.markdown("---")

# Retrieval parameters
st.sidebar.markdown("### 🗄️ Ingestion Ingest Parameters")
retrieval_strategy = st.sidebar.selectbox(
    "Search Strategy",
    options=["mmr", "similarity"],
    index=0,
    help="Similarity matches raw distance coordinates. MMR optimizes to balance relevance and chunk diversity."
)

k_chunks = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4, step=1)

if retrieval_strategy == "mmr":
    fetch_k = st.sidebar.slider("Initial Candidates (fetch_k)", min_value=k_chunks, max_value=30, value=12, step=1)
    lambda_mult = st.sidebar.slider("Diversity (lambda)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Higher values increase relevance; lower values increase diversity.")
else:
    fetch_k = 10
    lambda_mult = 0.5

# Model configurations
st.sidebar.markdown("### 🤖 Generative LLM Parameters")
llm_model = st.sidebar.selectbox("LLM Model", options=["mistral-small-latest", "open-mistral-7b"], index=0)
temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

st.sidebar.markdown("---")
# Quick Info / Stats
st.sidebar.markdown("### 📂 Ingested Corpus Status")
st.sidebar.info("Persistent DB: `RAG_Project/chroma_db`\n\nTarget Corpus: `RAG_Project/Big.pdf`")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Title and Headers
st.markdown("<h1 class='gradient-text'>RAG Conversational Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Context-aware document assistant indexing high-dimensional structures via ChromaDB</p>", unsafe_allow_html=True)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Core RAG Components
@st.cache_resource
def init_rag():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(current_dir, "chroma_db")
    
    if not os.path.exists(persist_dir):
        return None, None
        
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return vectorstore, embeddings

vectorstore, embeddings = init_rag()

if vectorstore is None:
    st.error("⚠️ Local Chroma Database not found! Please run the builder pipeline first to construct and populate your index: `python RAG_Project/Database.py`")
else:
    # Setup Dynamic Retriever
    if retrieval_strategy == "mmr":
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

    # Setup Generative LLM
    llm = ChatMistralAI(
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

    # Display Chat Messages
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        header_class = "user-header" if msg["role"] == "user" else "assistant-header"
        header_text = "You" if msg["role"] == "user" else "Assistant"
        
        st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <div class="bubble-header {header_class}">{header_text}</div>
            <div class="bubble-body">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # If there are sources, show them in expandable layout
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("🔍 Citations & Extracted Document Chunks"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-title">Source Chunk #{idx+1}</span><br/>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Source: {src.get('source', 'Unknown')}</span><br/>
                        <p style="margin-top: 0.3rem; color: #cbd5e1;">{src.get('page_content')}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # User Chat Input
    user_query = st.chat_input("Enter your query about the ingested document...")

    if user_query:
        # Show User Message
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <div class="bubble-header user-header">You</div>
            <div class="bubble-body">{user_query}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Retrieve and Synthesize Answer
        with st.spinner("🔍 Querying vector store and synthesizing response..."):
            start_time = time.time()
            retrieved_docs = retriever.invoke(user_query)
            retrieval_time = time.time() - start_time
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            final_prompt = prompt_template.format_messages(
                context=context,
                question=user_query
            )
            
            response = llm.invoke(final_prompt)
            answer = response.content

        # Show Assistant Bubble
        st.markdown(f"""
        <div class="chat-bubble assistant-bubble">
            <div class="bubble-header assistant-header">Assistant</div>
            <div class="bubble-body">{answer}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Format Sources to save in history
        sources_list = [
            {"page_content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
            for doc in retrieved_docs
        ]
        
        # Display new sources
        with st.expander("🔍 Citations & Extracted Document Chunks"):
            st.caption(f"Retrieved {len(retrieved_docs)} chunks in {retrieval_time:.3f} seconds using {retrieval_strategy.upper()} strategy.")
            for idx, src in enumerate(sources_list):
                st.markdown(f"""
                <div class="source-card">
                    <span class="source-title">Source Chunk #{idx+1}</span><br/>
                    <span style="color: #94a3b8; font-size: 0.85rem;">Source: {src['source']}</span><br/>
                    <p style="margin-top: 0.3rem; color: #cbd5e1;">{src['page_content']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_list
        })
