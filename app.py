import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chains import RetrievalQA  # now valid after upgrade to langchain>=0.3.0

# --- Load environment variables ---
load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(page_title="RAG App with Ollama", page_icon="🤖", layout="wide")

st.markdown("""
    <h1 style="text-align:center; color:#4CAF50;">📚 RAG Chatbot using LangChain + Ollama</h1>
    <p style="text-align:center; color:gray;">Upload a document, and ask anything about it!</p>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("🧩 Configuration")
top_k = st.sidebar.slider("Top K (Number of Chunks to Retrieve)", 1, 10, 3)
st.sidebar.info("Higher K = more context, but slower response.")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload your document", type=["pdf", "txt", "docx"])

# --- Model & Embeddings ---
llm = Ollama(model="mistral:7b") 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Document Processing ---
if uploaded_file is not None:
    with st.spinner("Processing your document..."):
        # Save uploaded file temporarily
        file_path = os.path.join("temp_files", uploaded_file.name)
        os.makedirs("temp_files", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Choose loader based on file type
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.error("Unsupported file type.")
            st.stop()

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("✅ Document processed successfully!")

        # --- User Query Input ---
        st.markdown("### 💬 Ask a question about your document:")
        user_query = st.text_input("Enter your question here...")

        if st.button("Get Answer"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    result = qa_chain.invoke({"query": user_query})

                st.markdown("### 🧠 **Answer:**")
                st.write(result["result"])

                # Show retrieved context
                with st.expander("📄 View Retrieved Context"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.write(doc.page_content[:500])
                        st.markdown("---")
else:
    st.info("⬆️ Please upload a document to begin.")
