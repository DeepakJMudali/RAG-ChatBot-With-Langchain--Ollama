# Chat with Ollama (Streamlit)

Lightweight Streamlit app that lets you chat with a local Ollama model and augment answers with web search results (DuckDuckGo). This variant does not use RAG/vector DB — it calls the model directly and injects retrieved web snippets into the prompt.

## Contents
- `app.py` — Streamlit app that uses `langchain_community` Ollama wrapper, `LLMChain` and `DuckDuckGoSearchResults`.

## Prerequisites
- Python 3.9+ (use a virtual environment)
- Ollama installed and running locally with the model you want (e.g. `mistral:7b` or `gemma3:latest`)
  - Install Ollama: https://ollama.com/docs
  - Pull a model (example): `ollama pull mistral:7b`
- Windows / Powershell commands shown below — adapt for other OS.

## Install (recommended)
Open your project virtual environment and run:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Powershell
python -m pip install --upgrade pip
python -m pip install streamlit python-dotenv langchain langchain-community

If you use the Ollama client package from langchain-community only, ensure it's installed:
python -m pip install langchain-community

If you see import issues like ModuleNotFoundError: No module named 'langchain.chains':
python -m pip install --upgrade langchain langchain-community
python -m pip show langchain

Usage
1-Ensure Ollama daemon is available and your chosen model is pulled:
ollama pull mistral:7b

2-Run the Streamlit app:
streamlit run app.py

3-Open the URL shown by Streamlit in your browser (default: http://localhost:8501).

4-Enter a query and press "Get Answer". The app will:
   i- Run a DuckDuckGo web search for current/retrieved info.
   ii- Pass the search text + user question into the LLM prompt template.
   iii- Display the model's answer.
