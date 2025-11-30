import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()
## load the GROQ API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

#Create the prompt template

prompt =ChatPromptTemplate.from_template("""
If the answer is found in the context, use the context.
If the answer is NOT in the context, then answer using your own general knowledge.
<context>{context}</context>
question:{input}
""")

#create the embedding

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state["embeddings"] = OpenAIEmbeddings()
        st.session_state["loader"] = PyPDFDirectoryLoader("Research_papers") # data ingestion step
        st.session_state["documents"] = st.session_state["loader"].load() # loading of documents
        st.session_state["text_splitter"] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Creating text_splitter
        st.session_state["finalResults"] = st.session_state["text_splitter"].split_documents(st.session_state["documents"][:50]) # splitted results
        st.session_state["vectorStore"] = FAISS.from_documents(st.session_state["finalResults"],st.session_state["embeddings"] )

        # creating retriever for runnables

        st.session_state["retriever"] =  st.session_state["vectorStore"].as_retriever()

        #create streamlit element
        st.session_state["rag_chain"]=(RunnableParallel({"context":st.session_state["retriever"] ,"input":RunnablePassthrough()}) | prompt | llm)

        # To mark that vectors are created
        st.session_state["vectors"] = True

# STREAMLIT UI
st.title("RAG Document Q&A with GROQ + OpenAI Embeddings")

# Initialize vector store + chain
create_vector_embeddings()

question = st.text_input("Ask a question")

if st.button("Submit") and question:
    response = st.session_state["rag_chain"].invoke(question)
    st.write("### Answer:")
    st.write(response.content)
