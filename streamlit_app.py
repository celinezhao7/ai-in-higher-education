import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# Load .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Missing OPENROUTER_API_KEY in .env")
    st.stop()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# File to store chat history
CHAT_FILE = "chat_history.json"

# Load chat history into session_state
if os.path.exists(CHAT_FILE):
    with open(CHAT_FILE, "r") as f:
        st.session_state.messages = json.load(f)
else:
    st.session_state.messages = []

# Function to save messages
def save_messages():
    with open(CHAT_FILE, "w") as f:
        json.dump(st.session_state.messages, f)

# Page config
st.set_page_config(
    page_title="AI Ethics Chatbot",
    page_icon="🍀",
    layout="wide",
)

# Custom CSS for theme
st.markdown("""
<style>
body {
    font-family: Georgia, serif;
    background-color: #F5F0FF;  /* light purple page background */
}
.user-bubble {
    text-align: left;
    background-color: #f6e6f1;  /* light pink for user */
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.assistant-bubble {
    text-align: left;
    background-color: #e7dff6;  /* slightly darker purple for assistant */
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 AI Ethics Chatbot")
st.markdown("Ask me anything about AI ethics. I’ll respond thoughtfully!")

# Button to start a new conversation
if st.button("✨ New Conversation"):
    st.session_state.messages = []
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
    st.success("Started a new conversation!")  # optional feedback
    
# Display chat history
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your question about AI ethics..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_messages()
    st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    # Build conversation for context
    conversation = []
    for msg in st.session_state.messages:
        conversation.append({"role": msg["role"], "content": msg["content"]})

    try:
        # Call OpenRouter model with reasoning
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
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