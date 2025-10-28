

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📘 RAG App using Hugging Face + LangChain")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    pdf_path = "./temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # ------------------------------
    # Load PDF
    # ------------------------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # ------------------------------
    # Create Embeddings + Vector Store
    # ------------------------------
    st.info("🔎 Creating embeddings and vector store...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # ------------------------------
    # Use Hugging Face Hub LLM
    # ------------------------------
    st.info("🧠 Loading Hugging Face LLM...")

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # 💡 You can change this to any HF model
        model_kwargs={
            "temperature": 0.3,
            "max_new_tokens": 512,
        },
        huggingfacehub_api_token=HF_TOKEN,
    )

    # ------------------------------
    # Prompt Template
    # ------------------------------
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the following context to answer the user's question.
    If you don't know, say "I'm not sure based on the document."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    # ------------------------------
    # Create RAG Chain
    # ------------------------------
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ------------------------------
    # User Query
    # ------------------------------
    query = st.text_input("💬 Ask a question based on the document:")

    if st.button("Get Answer") and query:
        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({"input": query})
            st.success("✅ Answer:")
            st.write(response["answer"])
