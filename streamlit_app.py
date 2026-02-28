import streamlit as st
from dotenv import load_dotenv
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env")
    st.stop()

from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

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
# UI
# -----------------------------
st.set_page_config(page_title="AI Ethics Educator", page_icon="🍀", layout="wide")

st.markdown("""
<style>
body { font-family: Georgia, serif; background-color: #F5F0FF; }
.user-bubble { text-align: left; background-color: #f6e6f1; padding: 10px; border-radius: 10px; margin: 5px 0; }
.assistant-bubble { text-align: left; background-color: #e7dff6; padding: 10px; border-radius: 10px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

st.title("💬 AI Ethics Educator")
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
# Load chunks + embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    # Your pre-saved chunks and metadata
    import pickle
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return chunks, metadata, model, embeddings

chunks, metadata, embedding_model, embeddings = load_embeddings()

# -----------------------------
# Retrieval function (cosine similarity)
# -----------------------------
def retrieve_context(query, top_k=4):
    query_emb = embedding_model.encode([query])[0]
    cos_sim = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    top_indices = np.argsort(-cos_sim)[:top_k]
    results = []
    for idx in top_indices:
        chunk_text = chunks[idx][:1000]  # truncate for speed
        source = metadata[idx]
        results.append(f"[Source: {source['title']} - {source['url']}]\n{chunk_text}")
    return "\n\n".join(results)

# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("Type your question about AI ethics..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages()
    st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    retrieved_context = retrieve_context(prompt)

    system_prompt = f"""
You are an AI Ethics research assistant grounded in UCSB policy,
campus journalism, and peer-reviewed research.

Use the following sources when relevant:

{retrieved_context}

Cite the source title when using information.
"""

    conversation = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    try:
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=conversation,
            extra_body={"reasoning": {"enabled": True}}
        )
        reply_text = response.choices[0].message.content
    except Exception as e:
        reply_text = f"❌ API request failed: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    save_messages()
    st.markdown(f"<div class='assistant-bubble'>{reply_text}</div>", unsafe_allow_html=True)