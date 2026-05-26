# Agents Module 🤖

> **Part of**: [Generative AI & LangChain Course](https://www.youtube.com/playlist?list=PLaldQ9PzZd9oXR4PMGR4pr_DX4wFHkFwR) — **Video 3: Runnables, Tools & Agents**

This module covers **LangChain Expression Language (LCEL)** — the composable, pipe-based system for wiring together prompts, models, parsers, and custom logic into production-grade chains.

---

## 📁 Folder Structure

```text
Agents/
│
├── Runnables/
│   ├── Sequence_runnables.py     # Basic LCEL pipe chain
│   ├── parallel_runnables.py     # Concurrent multi-branch chains
│   └── passthrough_runnables.py  # Piping raw output downstream
│
├── Tools/
│   ├── custom_tool.py            # @tool decorator — creating custom tools
│   ├── call_bind_execute_tool.py # Tool binding, LLM tool calls & manual execution
│   ├── news_summarizer.py        # TavilySearch tool + LCEL summarization chain
│   ├── Agent.py                  # LangGraph ReAct agent with weather & news tools
│   └── streamlit_app.py          # Premium Streamlit UI with agent trace transparency
│
├── .env                          # API keys (git-ignored)
├── requirements.txt              # Module dependencies
└── README.md                     # This file
```

---

## 🔗 What are Runnables?

In LangChain, a **Runnable** is any object that implements `.invoke()`, `.stream()`, or `.batch()`. The `|` pipe operator chains Runnables together — the output of one becomes the input of the next. This is called **LangChain Expression Language (LCEL)**.

```
prompt | model | parser
```

Every component (`ChatPromptTemplate`, `ChatMistralAI`, `StrOutputParser`, `RunnableLambda`, etc.) is a Runnable — making them infinitely composable.

---

## 🛠️ Implementations

### 1. `Sequence_runnables.py` — Basic LCEL Chain

The simplest LCEL pattern: a linear sequence where each step feeds into the next.

```python
chain = prompt | llm | parser
response = chain.invoke({"topic": "Machine Learning"})
```

**Key Concepts:**
- `ChatPromptTemplate.from_template()` — injects variables into a prompt string
- `ChatMistralAI` — LLM runnable that accepts a prompt and returns an AI message
- `StrOutputParser` — strips the `AIMessage` wrapper and returns raw text
- The `|` operator — equivalent to `parser.invoke(llm.invoke(prompt.invoke(input)))`

---

### 2. `parallel_runnables.py` — `RunnableParallel`

Runs multiple independent chains **concurrently** in a single `.invoke()` call, returning a dict of results.

```python
chain = RunnableParallel(
    short   = short_prompt | model | parser,
    detailed = long_prompt | model | parser
)

response = chain.invoke({'short': 'Machine Learning', 'detailed': 'Deep Learning'})
# response = {'short': '...', 'detailed': '...'}
```

**Key Concepts:**
- `RunnableParallel` fans the **same input dict** out to all branches simultaneously
- Each branch gets the full input and picks its own key(s)
- Branches run in separate threads — faster than sequential execution
- Results are collected back into a single dict keyed by branch name

> ⚠️ **Gotcha**: `.invoke()` accepts only **one** input dict. The second positional argument is treated as LangChain `config`, not a second data dict. Always combine all inputs into one dict.

---

### 3. `passthrough_runnables.py` — `RunnablePassthrough`

Passes the output of one chain **unchanged** into the next stage, while simultaneously running another branch that transforms it.

```python
seq1 = code_prompt | model | parser  # Generates code (string output)

seq2 = RunnableParallel(
    code        = RunnablePassthrough(),           # Passes raw code string through
    explanation = explain_prompt | model | parser   # Explains the code
)

final_seq = seq1 | seq2
response = final_seq.invoke({"topic": "Write a palindrome checker in Java"})
# response = {'code': '...java code...', 'explanation': '...plain english...'}
```

**Key Concepts:**
- `RunnablePassthrough` is a no-op runnable — it returns its input unchanged
- Useful for preserving intermediate values while other branches transform the data
- When `seq1`'s string output hits `seq2` (a `RunnableParallel`), both branches receive the same string
- `RunnablePassthrough()` echoes it; `explain_prompt | model | parser` generates an explanation

> ✅ **Pattern**: This is the standard LCEL pattern for "generate → (keep original + transform)" pipelines.

---

## 🔧 Tools

LangChain **Tools** are functions the LLM can decide to call when needed. A Tool is just a Python function decorated with `@tool` — LangChain automatically extracts its name, description, and typed arguments for the LLM to reason about.

### 4. `custom_tool.py` — `@tool` Decorator

The simplest way to create a tool — wrap any Python function:

```python
from langchain.tools import tool

@tool
def get_greetings(name: str) -> str:
    """This function is used to generate greetings for the user"""
    return f"Hello {name}, Welcome to AI World"

result = get_greetings.invoke({"name": "Shlok"})
print(get_greetings.name)         # get_greetings
print(get_greetings.description)  # This function is used to...
print(get_greetings.args)         # {'name': {'type': 'string', ...}}
```

**Key Concepts:**
- The docstring becomes the tool's description (what the LLM reads to decide when to call it)
- Type hints become the tool's args schema
- `.name`, `.description`, `.args` are auto-populated from the function signature

---

### 5. `call_bind_execute_tool.py` — Tool Binding & Manual Execution

Demonstrates the full **bind → call → execute** lifecycle:

```python
# 1. Create tool
@tool
def get_text_length(text: str) -> str:
    """Calculate the length of the text"""
    return f"The length is {len(text)} characters"

# 2. Bind to LLM
llm_with_tool = llm.bind_tools([get_text_length])

# 3. LLM decides to call the tool → extract call → execute manually → feed result back
result = llm_with_tool.invoke(messages)
if result.tool_calls:
    tool_output = tools[result.tool_calls[0]['name']].invoke(result.tool_calls[0])
    messages.append(ToolMessage(content=tool_output, tool_call_id=...))
```

**Key Concepts:**
- `llm.bind_tools([...])` — registers tools with the LLM so it knows when to call them
- The LLM returns `tool_calls` in the `AIMessage` when it wants to use a tool
- You manually execute the tool and feed the `ToolMessage` result back into the conversation
- This manual loop is what `AgentExecutor` / LangGraph automates

---

### 6. `news_summarizer.py` — `TavilySearchResults` + LCEL Chain

Combines a built-in LangChain tool with an LCEL summarization chain:

```python
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=5)
news_result = search_tool.invoke("Latest AI news of 2026")

chain = prompt | llm | parser
response = chain.invoke({"news": news_result})
```

**Key Concepts:**
- Pre-built community tools (`TavilySearchResults`) work exactly like `@tool` functions
- Tool output (a list of dicts) is passed directly into an LCEL chain as context
- This is the foundation of a RAG-style tool-augmented pipeline

---

### 7. `Agent.py` — LangGraph ReAct Agent

A full **ReAct (Reason + Act)** agent that autonomously decides which tools to call:

```python
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str: ...  # OpenWeatherMap API

@tool  
def get_news(city: str) -> str: ...    # Tavily search

agent = create_react_agent(
    model=llm,
    tools=[get_weather, get_news],
    prompt="You are a helpful City Agent assistant."
)

result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
```

**Key Concepts:**
- `create_react_agent` from LangGraph builds a full agentic loop automatically
- The agent reasons about the user query, decides which tool(s) to call, executes them, and synthesizes a final answer
- Supports multi-tool calls in a single turn (e.g. weather + news simultaneously)
- Returns the full message history including intermediate tool messages

---

### 8. `streamlit_app.py` — City Agent Streamlit UI

A premium dark-themed Streamlit web app wrapping the ReAct agent:

**Features:**
- 🌤 Real-time weather (OpenWeatherMap API)
- 📰 Live news summaries (Tavily API)
- 🔍 **Agent Trace Panel** — per-response expandable trace showing every internal step:
  - 👤 Human messages
  - ⚙️ Tool calls with args
  - 📦 Tool results (raw API data)
  - 🤖 AI reasoning messages
- 🎛 Sidebar toggle to auto-expand traces
- 💡 Quick-ask suggestion buttons
- 📊 Live message & tool-call counters

```bash
streamlit run Tools/streamlit_app.py
```

---

## 🚀 Setup

### 1. Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in this directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

### 4. Run Scripts
```bash
# Basic sequence chain
python Runnables/Sequence_runnables.py

# Parallel multi-branch chain
python Runnables/parallel_runnables.py

# Passthrough code + explanation chain
python Runnables/passthrough_runnables.py

# Custom tool demo
python Tools/custom_tool.py

# Tool binding + manual execution loop
python Tools/call_bind_execute_tool.py

# News summarizer with Tavily
python Tools/news_summarizer.py

# ReAct agent CLI
python Tools/Agent.py

# Premium Streamlit UI
streamlit run Tools/streamlit_app.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain-core` | Runnable primitives, prompt templates, output parsers |
| `langchain-mistralai` | `ChatMistralAI` LLM integration |
| `langchain-community` | Pre-built tools (`TavilySearchResults`, etc.) |
| `langgraph` | `create_react_agent` — full agentic loop |
| `tavily-python` | Tavily web search API client |
| `requests` | OpenWeatherMap HTTP calls |
| `streamlit` | Web UI framework |
| `python-dotenv` | Load `.env` API keys |

---

## 🧩 Key Concepts Cheat Sheet

### LCEL Runnables
| Runnable | Description |
|---|---|
| `prompt \| model \| parser` | Sequential chain via pipe operator |
| `RunnableParallel(a=..., b=...)` | Run branches concurrently, return `{'a': ..., 'b': ...}` |
| `RunnablePassthrough()` | Echo input unchanged — preserves data through a parallel branch |
| `RunnableLambda(fn)` | Wrap any Python function as a Runnable |

### Tools & Agents
| Concept | Description |
|---|---|
| `@tool` | Decorator that turns any Python function into an LLM-callable tool |
| `llm.bind_tools([...])` | Register tools with the LLM so it knows when to use them |
| `result.tool_calls` | List of tool calls the LLM wants to make |
| `ToolMessage` | Message type that carries a tool's output back into the conversation |
| `create_react_agent` | LangGraph prebuilt that runs a full ReAct reasoning + tool execution loop |

---

*Video 3 ✅ of the [Generative AI & LangChain playlist](https://www.youtube.com/playlist?list=PLaldQ9PzZd9oXR4PMGR4pr_DX4wFHkFwR) — Runnables, Tools & Agents fully covered.*
