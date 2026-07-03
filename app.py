"""
StudyMate — Personal Research & Homework Assistant
A single-agent Streamlit app: web search + calculator + RAG over your own uploaded document,
with conversation memory. Deploy this file directly on Streamlit Cloud.

Setup on Streamlit Cloud:
1. Push this file and requirements.txt to a GitHub repo.
2. On streamlit.io/cloud, create a new app pointing at this file.
3. In the app's Settings -> Secrets, add:
     GROQ_API_KEY = "..."
     TAVILY_API_KEY = "..."
"""

import json
import streamlit as st
import groq
from tavily import TavilyClient
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import io

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="centered")
st.title("📚 StudyMate")
st.caption("Your research & homework assistant — upload your notes, then ask anything.")

# ---------- clients (built once, using secrets) ----------
client = groq.Groq(api_key=st.secrets["GROQ_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# Groq-hosted model used for the agent's reasoning + tool calling.
# openai/gpt-oss-120b is Groq's current recommended model for strong tool-use quality.
MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are StudyMate, a helpful research and homework assistant. "
    "Always check the user's own notes (search_my_notes) before searching the open web, "
    "when the question could plausibly be covered in their uploaded document. "
    "Be clear and concise. If neither tool has the answer, say so honestly instead of guessing."
)


@st.cache_resource
def get_collection():
    chroma_client = chromadb.Client()
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return chroma_client.get_or_create_collection(name="my_notes", embedding_function=embed_fn)


collection = get_collection()

# ---------- session state ----------
# The conversation list IS the agent's memory, and its first entry is the
# system prompt (Groq/OpenAI-style messages put the system prompt in the
# messages list itself, rather than passing it as a separate argument).
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False


# ---------- document ingestion ----------
def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


with st.sidebar:
    st.header("Your documents")
    uploaded_file = st.file_uploader("Upload notes (PDF or .txt)", type=["pdf", "txt"])
    if uploaded_file is not None and st.button("Index this document"):
        with st.spinner("Reading and indexing..."):
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)
            existing = collection.count()
            collection.add(
                documents=chunks,
                ids=[f"chunk_{existing + i}" for i in range(len(chunks))],
            )
            st.session_state.doc_indexed = True
        st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")
    if st.session_state.doc_indexed:
        st.info("StudyMate will check your notes first before searching the web.")
    if st.button("Clear conversation"):
        st.session_state.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ---------- tools ----------
def search_my_notes(query: str) -> str:
    if collection.count() == 0:
        return "No documents have been uploaded yet."
    results = collection.query(query_texts=[query], n_results=3)
    if not results["documents"][0]:
        return "No relevant content found in the uploaded document."
    return "\n\n".join(results["documents"][0])


def web_search(query: str) -> str:
    results = tavily.search(query=query, max_results=3)
    return "\n".join(f"- {r['title']}: {r['content'][:200]}" for r in results["results"])


def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Groq's API is OpenAI-compatible, so tools are described in OpenAI's
# "function calling" schema: each tool is wrapped in a {"type": "function", "function": {...}}
# object, with parameters as JSON Schema.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_my_notes",
            "description": "Search the user's own uploaded document/notes for relevant content. ALWAYS try this first for anything that could be covered in the user's material before searching the open web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for in the notes."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the open web for current facts, definitions, or general knowledge not found in the user's own notes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression, e.g. for grade averages, percentages, or unit conversions.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "A Python-evaluable math expression."}},
                "required": ["expression"],
            },
        },
    },
]


def run_tool(name, tool_input):
    if name == "search_my_notes":
        return search_my_notes(**tool_input)
    elif name == "web_search":
        return web_search(**tool_input)
    elif name == "calculator":
        return calculator(**tool_input)
    return f"Error: tool '{name}' does not exist."


def run_agent(messages, max_iterations=6):
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        message = response.choices[0].message
        # Groq's SDK wants the assistant message appended as a plain dict.
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content, messages

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            result = run_tool(tool_name, tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
    return "Sorry, I couldn't finish that in time.", messages


# ---------- chat UI ----------
for msg in st.session_state.conversation:
    if msg["role"] in ("user", "assistant") and isinstance(msg.get("content"), str) and msg["content"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("Ask about your notes, or anything else...")
if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, st.session_state.conversation = run_agent(st.session_state.conversation)
            st.write(answer)
