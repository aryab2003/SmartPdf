import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from rag_pipeline import answer_query, llm_model  # Make sure this does NOT use Streamlit

# === Configuration ===
PDF_DIR = 'pdfs/'
FAISS_DB_PATH = 'vectorstore/'
OLLAMA_MODEL = 'deepseek-r1:1.5b'

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# === Helper Functions ===
def upload_and_save_pdf(file):
    file_path = os.path.join(PDF_DIR, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def parse_pdf_to_docs(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_and_save_faiss(text_chunks, path, model):
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local(path)
    return vectorstore

def load_faiss_store(path, model):
    embeddings = OllamaEmbeddings(model=model)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# === Streamlit App ===

st.title("📄 Smart PDFer")
st.caption("Upload any PDF and ask questions using AI!")

file = st.file_uploader("📎 Upload PDF", type="pdf", key="pdf_uploader")
query = st.text_area("💬 Enter your prompt", height=120, placeholder="Ask me anything about your document.")
btn = st.button("🚀 Ask AI")
create_db_btn = st.button("🔨 Create Vector DB")

# Processing for "Create Vector DB"
if create_db_btn and file:
    with st.spinner("Processing PDF and generating vector store..."):
        file_path = upload_and_save_pdf(file)
        docs = parse_pdf_to_docs(file_path)
        chunks = split_into_chunks(docs)
        faiss_db = create_and_save_faiss(chunks, FAISS_DB_PATH, OLLAMA_MODEL)
    st.success("✅ Vector store created successfully!")

# Check if FAISS DB exists or not
if os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
    faiss_db = load_faiss_store(FAISS_DB_PATH, OLLAMA_MODEL)
else:
    faiss_db = None

# Respond to query
if btn:
    if file and query.strip():
        st.chat_message("user").write(query)

        with st.spinner("🔎 Retrieving relevant content..."):
            relevant_docs = faiss_db.similarity_search(query, k=5)
            response = answer_query(relevant_docs, llm_model, query)
        st.chat_message("SmartPDF AI").write(response.content)

    elif not file:
        st.error("❗ Please upload a PDF first.")
    elif not query.strip():
        st.error("❗ Please enter a valid question.")
