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
```

### 4. Run Scripts
```bash
# Basic sequence chain
python Runnables/Sequence_runnables.py

# Parallel multi-branch chain
python Runnables/parallel_runnables.py

# Passthrough code + explanation chain
python Runnables/passthrough_runnables.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain-core` | Runnable primitives, prompt templates, output parsers |
| `langchain-mistralai` | `ChatMistralAI` LLM integration |
| `python-dotenv` | Load `.env` API keys |

---

## 🧩 LCEL Runnable Cheat Sheet

| Runnable | Description |
|---|---|
| `prompt \| model \| parser` | Sequential chain via pipe operator |
| `RunnableParallel(a=..., b=...)` | Run branches concurrently, return `{'a': ..., 'b': ...}` |
| `RunnablePassthrough()` | Echo input unchanged — preserves data through a parallel branch |
| `RunnableLambda(fn)` | Wrap any Python function as a Runnable |

---

*Video 3 of the [Generative AI & LangChain playlist](https://www.youtube.com/playlist?list=PLaldQ9PzZd9oXR4PMGR4pr_DX4wFHkFwR) — covers through Runnables. Tools & Agents coming next.*
