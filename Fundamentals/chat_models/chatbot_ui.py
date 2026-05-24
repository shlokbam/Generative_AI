import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# App Page Configuration
st.set_page_config(
    page_title="Regal Chatbot",
    page_icon="👑",
    layout="centered",
)

# Premium Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">King Bot 👑</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your funny, regal AI companion powered by Llama 3 & Groq</div>', unsafe_allow_html=True)

# Sidebar settings for premium feel
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )
    
    # Custom system instruction
    system_instruction = st.text_area(
        "System Instruction",
        value="You are a funny AI agent. Use puns, light-hearted jokes, and self-reference as 'King' when appropriate.",
        height=100
    )
    
    st.markdown("---")
    
    # Reset Chat button
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()

# Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display current chat messages from session state
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="👑"):
            st.markdown(msg.content)

# Input prompt from user
user_input = st.chat_input("Ask King anything...")

if user_input:
    # Display human message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Add to list
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Reconstruct messages list starting with SystemMessage
    langchain_messages = [SystemMessage(content=system_instruction)] + st.session_state.messages
    
    # Call Groq LLM
    with st.chat_message("assistant", avatar="👑"):
        message_placeholder = st.empty()
        with st.spinner("King is thinking... 💭"):
            try:
                # Initialize LLM with selected config
                llm = ChatGroq(
                    model=model_option,
                    temperature=temperature,
                )
                
                # Fetch response
                response = llm.invoke(langchain_messages)
                ai_response = response.content
                
                # Render response
                message_placeholder.markdown(ai_response)
                
                # Save response to history
                st.session_state.messages.append(AIMessage(content=ai_response))
            except Exception as e:
                st.error(f"Error fetching response: {e}")
