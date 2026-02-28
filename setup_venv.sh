#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Setup script for AI Ethics Chatbot
# - Creates a Python virtual environment in .venv
# - Installs required dependencies from requirements.txt
# ----------------------------------------------------------------------

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing default packages..."
    python -m pip install streamlit python-dotenv openai requests
fi

# Instructions
echo ""
echo "✅ Virtual environment created at .venv and dependencies installed."
echo "💡 Copy .env.example to .env and set your OPENROUTER_API_KEY:"
echo "   OPENROUTER_API_KEY=sk-xxxxxxx"
echo "💡 Activate the environment before running Streamlit:"
echo "   source .venv/bin/activate"
echo "💡 Run your app:"
echo "   streamlit run streamlit_app.py"