import streamlit as st
import os
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient
from langgraph.prebuilt import create_react_agent


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="City Agent",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Background */
.stApp { background: linear-gradient(160deg, #07090f 0%, #0d1120 60%, #080b15 100%) !important; }

/* Hide clutter */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 13, 25, 0.98) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.12) !important;
}

/* Chat messages — dark glass */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(8px) !important;
}

/* User message highlight */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
}

/* Message text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span { 
    color: #e2e8f0 !important; 
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

/* Send button */
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border-radius: 12px !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
}

/* Expander (trace panel) */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.25) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    margin-top: 4px !important;
}
[data-testid="stExpander"] summary {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #94a3b8 !important; }

/* Sidebar buttons */
.stButton button {
    background: rgba(99, 102, 241, 0.08) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #a5b4fc !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton button:hover {
    background: rgba(99, 102, 241, 0.18) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: white !important;
}

/* Tool call notification pill */
.tool-call-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.25);
    color: #fbbf24;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Status dot */
.status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #34d399;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.dot {
    width: 8px; height: 8px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(0.8); }
}

/* Trace message boxes */
.trace-human {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: #c7d2fe;
}
.trace-ai {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: #a7f3d0;
}
.trace-tool-call {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: #fde68a;
}
.trace-tool-result {
    background: rgba(168,85,247,0.08);
    border-left: 3px solid #a855f7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: #e9d5ff;
}
.trace-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.7;
}

/* Stat pill */
.stat-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 4px 0;
    color: #64748b;
    font-size: 0.8rem;
}
.stat-pill .val { color: #a5b4fc; font-weight: 700; font-size: 1rem; }

hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)


# ── Agent Init ────────────────────────────────────────────────────────────────
@st.cache_resource
def init_agent():
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city in India."""
        api_key = os.getenv("OPENWEATHER_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
        data = requests.get(url).json()
        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather')}"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        feels = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        return f"Weather in {city}: {desc}, Temp: {temp}°C (feels like {feels}°C), Humidity: {humidity}%"

    @tool
    def get_news(city: str) -> str:
        """Get the latest news about a city."""
        tc = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tc.search(query=f"latest news in {city}", search_depth="basic", max_results=3)
        results = response.get("results", [])
        if not results:
            return "No news found."
        news_list = []
        for r in results:
            title   = r.get("title", "No title")
            url     = r.get("url", "")
            snippet = r.get("content", "")
            news_list.append(f"- **{title}**\n  🔗 {url}\n  {snippet[:120]}...")
        return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

    llm = ChatMistralAI(model_name="mistral-small")
    agent = create_react_agent(
        model=llm,
        tools=[get_weather, get_news],
        prompt=(
            "You are City Agent — a friendly assistant that provides real-time weather and news "
            "for any city. Always respond in clear, formatted markdown. Use bullet points, bold "
            "text, and emojis where appropriate."
        )
    )
    return agent


# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # list of {role, content, tools_used, trace}
if "tool_calls_total" not in st.session_state:
    st.session_state.tool_calls_total = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌆 City Agent")
    st.markdown('<div class="status"><div class="dot"></div>Agent Online</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🔧 Tools Available**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
        border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:1.6rem">🌤</div>
            <div style="color:#a5b4fc;font-size:0.75rem;font-weight:600;margin-top:4px">Weather</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
        border-radius:12px;padding:14px;text-align:center;">
            <div style="font-size:1.6rem">📰</div>
            <div style="color:#a5b4fc;font-size:0.75rem;font-weight:600;margin-top:4px">News</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    msgs = len(st.session_state.history)
    tools = st.session_state.tool_calls_total
    st.markdown(f"""
    <div class="stat-pill"><span>💬 Messages</span><span class="val">{msgs}</span></div>
    <div class="stat-pill"><span>🔧 Tool calls</span><span class="val">{tools}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Quick ask**")
    suggestions = ["Weather in Mumbai", "News in Delhi", "Weather & news in Bangalore", "How's Pune today?"]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            st.session_state["prefill"] = s
            st.rerun()

    st.markdown("---")

    # Transparency toggle
    show_trace = st.toggle("🔍 Show agent trace by default", value=False)

    st.markdown("---")
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.tool_calls_total = 0
        st.rerun()

    st.markdown('<p style="color:#1e293b;font-size:0.7rem;text-align:center;margin-top:12px;">Mistral AI · LangGraph · Tavily · OpenWeather</p>', unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 4px;">
    <div style="font-size:2.2rem;font-weight:800;color:white;letter-spacing:-0.5px;">🌆 City Agent</div>
    <div style="color:#334155;font-size:0.9rem;margin-top:6px;">Real-time weather & news · Powered by Mistral AI</div>
</div>
""", unsafe_allow_html=True)


# ── Render Chat History ────────────────────────────────────────────────────────
def render_trace(trace_messages: list):
    """Render the full agent trace in an expander."""
    tool_icons = {"get_weather": "🌤", "get_news": "📰"}

    for m in trace_messages:

        if isinstance(m, HumanMessage):
            st.markdown('<div class="trace-human"><div class="trace-label">👤 Human Message</div>', unsafe_allow_html=True)
            st.markdown(str(m.content))
            st.markdown('</div>', unsafe_allow_html=True)

        elif isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    icon     = tool_icons.get(tc.get("name", ""), "🔧")
                    tname    = tc.get("name", "unknown")
                    args_str = str(tc.get("args", {}))
                    st.markdown(f"""
                    <div class="trace-tool-call">
                        <div class="trace-label">⚙️ Tool Call → {tname}</div>
                        {icon} Args: <code>{args_str}</code>
                    </div>""", unsafe_allow_html=True)

            if m.content:
                content = str(m.content)
                preview = content[:600] + ("..." if len(content) > 600 else "")
                st.markdown('<div class="trace-ai"><div class="trace-label">🤖 AI Message</div>', unsafe_allow_html=True)
                st.markdown(preview)
                st.markdown('</div>', unsafe_allow_html=True)

        elif isinstance(m, ToolMessage):
            content   = str(m.content)
            preview   = content[:400] + ("..." if len(content) > 400 else "")
            st.markdown('<div class="trace-tool-result"><div class="trace-label">📦 Tool Result</div>', unsafe_allow_html=True)
            st.markdown(preview)
            st.markdown('</div>', unsafe_allow_html=True)



if not st.session_state.history:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px 40px;color:#1e293b;">
        <div style="font-size:3.5rem;margin-bottom:12px;">🌍</div>
        <div style="font-size:1.1rem;font-weight:600;color:#334155;">Ask about any city</div>
        <div style="font-size:0.85rem;margin-top:6px;color:#1e293b;">Weather · Latest News · Anything city-related</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for entry in st.session_state.history:
        with st.chat_message("user" if entry["role"] == "user" else "assistant", avatar="👤" if entry["role"] == "user" else "🌆"):
            if entry["role"] == "user":
                st.markdown(entry["content"])
            else:
                # Tool call badges
                if entry.get("tools_used"):
                    tool_icons_map = {"get_weather": "🌤", "get_news": "📰"}
                    tool_labels    = {"get_weather": "WEATHER TOOL", "get_news": "NEWS TOOL"}
                    badges = " ".join([
                        f'<span class="tool-call-pill">{tool_icons_map.get(t,"🔧")} {tool_labels.get(t,t)}</span>'
                        for t in entry["tools_used"]
                    ])
                    st.markdown(badges, unsafe_allow_html=True)

                # Actual response (rendered as markdown)
                st.markdown(entry["content"])

                # Trace expander
                if entry.get("trace"):
                    with st.expander("View agent trace — messages, tool calls & results", expanded=show_trace, icon="🔍"):
                        render_trace(entry["trace"])


# ── Chat Input ─────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input(
    placeholder="Ask about weather or news in any city...",
    key="chat_input"
)

# Handle suggestion prefill
if prefill and not user_input:
    user_input = prefill


# ── Process Input ──────────────────────────────────────────────────────────────
if user_input and user_input.strip():
    query = user_input.strip()

    # Show user message immediately
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # Agent thinking
    with st.chat_message("assistant", avatar="🌆"):
        # Animated live tool-call notification
        tool_status_placeholder = st.empty()
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<div style="color:#6366f1;font-size:0.85rem;font-style:italic;">⏳ Thinking...</div>',
            unsafe_allow_html=True
        )

        try:
            agent = init_agent()

            # Stream-style: run agent and collect trace
            all_tool_names = []
            raw_trace = []

            # Invoke with streaming awareness
            result = agent.invoke({"messages": [HumanMessage(content=query)]})

            raw_trace = result["messages"]

            # Detect tool calls from trace
            for m in raw_trace:
                if isinstance(m, AIMessage) and m.tool_calls:
                    for tc in m.tool_calls:
                        tname = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                        if tname and tname not in all_tool_names:
                            all_tool_names.append(tname)
                            st.session_state.tool_calls_total += 1

            final_answer = result["messages"][-1].content
            thinking_placeholder.empty()
            tool_status_placeholder.empty()

            # Render tool badges
            if all_tool_names:
                tool_icons_map = {"get_weather": "🌤", "get_news": "📰"}
                tool_labels    = {"get_weather": "WEATHER TOOL", "get_news": "NEWS TOOL"}
                badges = " ".join([
                    f'<span class="tool-call-pill">{tool_icons_map.get(t,"🔧")} {tool_labels.get(t,t)}</span>'
                    for t in all_tool_names
                ])
                st.markdown(badges, unsafe_allow_html=True)

            # Final answer (markdown rendered)
            st.markdown(final_answer)

            # Trace expander
            with st.expander("View agent trace — messages, tool calls & results", expanded=show_trace, icon="🔍"):
                render_trace(raw_trace)

            # Save to history
            st.session_state.history.append({
                "role": "assistant",
                "content": final_answer,
                "tools_used": all_tool_names,
                "trace": raw_trace
            })

        except Exception as e:
            thinking_placeholder.empty()
            err = f"⚠️ **Error:** {str(e)}"
            st.markdown(err)
            st.session_state.history.append({
                "role": "assistant",
                "content": err,
                "tools_used": [],
                "trace": []
            })

    st.rerun()
