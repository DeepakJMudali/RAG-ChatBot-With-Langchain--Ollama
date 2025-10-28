from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import streamlit as st

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📘 RAG App using Ollama + LangChain")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    pdf_path = f"./temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # ------------------------------
    # Load and split the PDF
    # ------------------------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # ------------------------------
    # Create Embeddings and Vector Store
    # ------------------------------
    st.info("Creating embeddings and vector store...")

    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    retriever = vectorstore.as_retriever()

    # ------------------------------
    # LLM and prompt setup
    # ------------------------------
    llm = Ollama(model="gemma:2b")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the provided context to answer the question.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    # ------------------------------
    # Create RAG chain
    # ------------------------------
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ------------------------------
    # User query input
    # ------------------------------
    query = st.text_input("💬 Ask a question based on the document:")

    if st.button("Get Answer") and query:
        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({"input": query})
            st.success("✅ Answer:")
            st.write(response["answer"])
