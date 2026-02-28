import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load .env and API key
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# -----------------------------
# Chat history
# -----------------------------
CHAT_FILE = "chat_history.json"

if os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "r") as f:
        st.session_state.messages = json.load(f)
else:
    st.session_state.messages = []

def save_messages():
    with open(CHAT_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

# -----------------------------
# Page setup and CSS
# -----------------------------
st.set_page_config(page_title="AI Ethics Chatbot", page_icon="🍀", layout="wide")

st.markdown("""
<style>
body { font-family: Georgia, serif; background-color: #F5F0FF; }
.user-bubble { text-align: left; background-color: #f6e6f1; padding: 10px; border-radius: 10px; margin: 5px 0; }
.assistant-bubble { text-align: left; background-color: #e7dff6; padding: 10px; border-radius: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

st.title("💬 AI Ethics Chatbot")
st.markdown("Ask me anything about AI ethics. I’ll respond thoughtfully!")

if st.button("✨ New Conversation"):
    st.session_state.messages = []
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
    st.success("Started a new conversation!")

# Display chat history
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# -----------------------------
# Load embeddings & FAISS once
# -----------------------------
@st.cache_resource
def load_embedding_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("ethics_index.faiss") # make sure this file exists
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, chunks, metadata

embedding_model, index, chunks, metadata = load_embedding_resources()

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve_context(query, k=4):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = []
    for idx in indices[0]:
        source = metadata[idx]
        chunk_text = chunks[idx][:1000]  # truncate to 1000 chars for speed
        results.append(f"[Source: {source['title']} - {source['url']}]\n{chunk_text}")
    return "\n\n".join(results)

# -----------------------------
# Chat input & RAG response
# -----------------------------
if prompt := st.chat_input("Type your question about AI ethics..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages()
    st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    # Retrieve context
    retrieved_context = retrieve_context(prompt)

    # Build system prompt
    system_prompt = f"""
You are an AI Ethics research assistant grounded in UCSB policy,
campus journalism, and peer-reviewed research.

Use the following sources when relevant:

{retrieved_context}

Cite the source title when using information.
"""

    # Build conversation
    conversation = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    # Call OpenRouter
    try:
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=conversation,
            extra_body={"reasoning": {"enabled": True}}
        )
        assistant_msg = response.choices[0].message
        reply_text = assistant_msg.content
    except Exception as e:
        reply_text = f"❌ API request failed: {e}"

    # Save and display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    save_messages()
    st.markdown(f"<div class='assistant-bubble'>{reply_text}</div>", unsafe_allow_html=True)