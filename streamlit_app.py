import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import uuid
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load API key
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
# Load RAG knowledge base
# -----------------------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_knowledge_base():
    with open(BASE_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(BASE_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    with open(BASE_DIR / "embeddings.pkl", "rb") as f:
        embeddings = np.array(pickle.load(f))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return chunks, metadata, embeddings, embedding_model


chunks, metadata, embeddings, embedding_model = load_knowledge_base()

# -----------------------------
# Retrieve relevant RAG sources
# -----------------------------
def retrieve_relevant_chunks(question, k=3):
    # Turn the user's question into an embedding
    question_embedding = embedding_model.encode(question)

    # Calculate cosine similarity between the question and every saved chunk
    question_norm = np.linalg.norm(question_embedding) + 1e-10
    chunk_norms = np.linalg.norm(embeddings, axis=1) + 1e-10

    similarity_scores = (embeddings @ question_embedding) / (
        chunk_norms * question_norm
    )

    # Find the indices of the top-k most relevant chunks
    top_indices = np.argsort(similarity_scores)[-k:][::-1]

    # Format them as reference material for the AI
    results = []
    for i in top_indices:
        results.append(
            f"[Source: {metadata[i]['title']}]\n"
            f"URL: {metadata[i]['url']}\n"
            f"{chunks[i]}"
        )

    return "\n\n---\n\n".join(results)

    
# -----------------------------
# Assign a unique session ID for each user
# -----------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Each user gets their own chat history file (optional persistent storage)
CHAT_FILE = f"chat_{st.session_state.user_id}.json"

# Load chat history for this user (if exists)
if os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "r") as f:
        st.session_state.messages = json.load(f)
else:
    st.session_state.messages = []

def save_messages():
    with open(CHAT_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

# -----------------------------
# Streamlit UI
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

st.markdown("### Grounded in UCSB policy, campus journalism, and peer-reviewed research.")

st.markdown("---")

st.markdown("## Research Question")
st.markdown("""
How is the rise of AI tools like ChatGPT reshaping academic integrity,
learning practices, and ethical decision-making among students at UCSB?
""")
# -----------------------------
# Start a new conversation button
# -----------------------------
if st.button("✨ New Conversation"):
    st.session_state.messages = []
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
    st.success("Started a new conversation!")

# -----------------------------
# Display chat history
# -----------------------------
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("Type your question about AI ethics..."):
    # Save user message in their session
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages()
    st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    # Retrieve the most relevant UCSB research chunks for this question
    retrieved_context = retrieve_relevant_chunks(prompt, k=3)

    # Temporary display: proves which RAG sources were found
    with st.expander("Sources retrieved for this answer"):
        st.write(retrieved_context)

    conversation = [{
        "role": "system",
        "content": f"""You are an AI Ethics research assistant.

Answer clearly and concisely, using the retrieved source material below.
Do not invent UCSB-specific facts that are not supported by these sources.
If the sources do not contain enough information, say that clearly.
Mention the source title when making a factual claim.
Focus on one main point only.
Keep the answer to 2–3 short sentences.
Use a natural, student-friendly tone.
Use Markdown bolding sparingly, only for one or two genuinely important words or phrases. Do not bold full sentences or whole paragraphs.
Do not use headings, bullet points, tables, or generic AI phrases such as “in summary.”

RETRIEVED SOURCE MATERIAL:
{retrieved_context}
"""
    }]

    for msg in st.session_state.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    # Call OpenRouter
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=conversation,
            extra_body={"reasoning": {"enabled": True}}
        )
        assistant_msg = response.choices[0].message
        reply_text = assistant_msg.content
    except Exception as e:
        reply_text = f"❌ API request failed: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    save_messages()
    st.markdown(f"<div class='assistant-bubble'>{reply_text}</div>", unsafe_allow_html=True)

    # Detect Streamlit theme (light or dark)
theme = st.get_option("theme.base")  # returns "light" or "dark"

# Set bubble colors based on theme
if theme == "dark":
    user_bg = "#4a2c4c"        # dark purple for user bubble
    assistant_bg = "#6a4c8b"   # lighter dark purple for assistant bubble
    text_color = "#ffffff"      # white text
else:
    user_bg = "#f6e6f1"         # light pink for user
    assistant_bg = "#e7dff6"    # light purple for assistant
    text_color = "#111111"      # dark text

# Inject CSS
st.markdown(f"""
<style>
.user-bubble {{
    text-align: left;
    background-color: {user_bg};
    color: {text_color};
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}}
.assistant-bubble {{
    text-align: left;
    background-color: {assistant_bg};
    color: {text_color};
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}}
body {{
    font-family: Georgia, serif;
    background-color: {'#111111' if theme == 'dark' else '#F5F0FF'};
}}
</style>
""", unsafe_allow_html=True)