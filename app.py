# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms.huggingface_hub import HuggingFaceHub
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# import streamlit as st

# # ------------------------------
# # Streamlit UI
# # ------------------------------
# st.title("📘 RAG App using Ollama + LangChain")

# uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

# if uploaded_file:
#     # Save uploaded file temporarily
#     pdf_path = f"./temp.pdf"
#     with open(pdf_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # ------------------------------
#     # Load and split the PDF
#     # ------------------------------
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     # ------------------------------
#     # Create Embeddings and Vector Store
#     # ------------------------------
#     st.info("Creating embeddings and vector store...")

#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
 
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vectorstore = FAISS.from_documents(docs, embedding=embeddings)

#     retriever = vectorstore.as_retriever()

#     # ------------------------------
#     # LLM and prompt setup
#     # ------------------------------
#     llm = Ollama(model="gemma:2b")

#     prompt = ChatPromptTemplate.from_template("""
#     You are a helpful assistant. Use the provided context to answer the question.

#     Context:
#     {context}

#     Question:
#     {input}

#     Answer:
#     """)

#     # ------------------------------
#     # Create RAG chain
#     # ------------------------------
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     # ------------------------------
#     # User query input
#     # ------------------------------
#     query = st.text_input("💬 Ask a question based on the document:")

#     if st.button("Get Answer") and query:
#         with st.spinner("Generating answer..."):
#             response = retrieval_chain.invoke({"input": query})
#             st.success("✅ Answer:")
#             st.write(response["answer"])

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import streamlit as st

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📘 RAG App using LangChain + LLM (Ollama/HuggingFace)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

# Detect environment (Ollama locally, HuggingFace in production)
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if uploaded_file:
    pdf_path = "./temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # ------------------------------
    # Load PDF and create embeddings
    # ------------------------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    st.info("Creating embeddings and vector store...")

    if USE_OLLAMA:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # ------------------------------
    # Choose LLM
    # ------------------------------
    if USE_OLLAMA:
        llm = Ollama(model="gemma:2b")
    else:
        # Hugging Face model (requires your token)
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={"temperature": 0.3, "max_new_tokens": 512},
        )

    # ------------------------------
    # Prompt & Retrieval Chain
    # ------------------------------
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the provided context to answer the question accurately.
    If the answer isn't in the document, say "I couldn't find an exact answer in the document."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ------------------------------
    # User query
    # ------------------------------
    query = st.text_input("💬 Ask a question based on the document:")

    if st.button("Get Answer") and query:
        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({"input": query})
            st.success("✅ Answer:")
            st.write(response["answer"])
