import os
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# --- Load environment variable ---
# Make sure you set HF_TOKEN in Streamlit Secrets or .env
hf_token = os.getenv("HF_TOKEN")

# --- Load PDF and split ---
loader = PyPDFLoader("file-example_PDF_500_kB.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# --- Create embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Create vectorstore ---
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# --- Initialize LLM ---
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=hf_token,  # must be valid token
    task="text-generation",              # ✅ required in new version
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 512,
    },
)

# --- Create retrieval chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# --- Query example ---
query = "Summarize the document briefly."
result = qa_chain.invoke({"query": query})
print(result)
