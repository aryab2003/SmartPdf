import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Directory setup
pdf_dir = 'pdfs/'
FAISS_DB_PATH = "vectorstore/"
ollama_model = "deepseek-r1:1.5b"

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)

# Upload and save file
def upload_file(file):
    with open(pdf_dir + file.name, "wb") as f:
        f.write(file.getbuffer())

# PDF to documents
def load_parse_file(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Documents to chunks
def make_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    text_chunks = splitter.split_documents(documents)
    return text_chunks

# Get embeddings
def get_embeddings(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings

# Save vector store
def create_vector_store(db_faiss_path, text_chunks, ollama_model_name):
    faiss_db = FAISS.from_documents(text_chunks, get_embeddings(ollama_model_name))
    faiss_db.save_local(db_faiss_path)
    return faiss_db

# === Streamlit UI ===
st.title("📄Smart PDFer")

file = st.file_uploader("Upload PDF", type="pdf")

if file is not None:
    with st.spinner("Processing..."):
        upload_file(file)
        file_path = pdf_dir + file.name
        documents = load_parse_file(file_path)
        text_chunks = make_chunks(documents)
        create_vector_store(FAISS_DB_PATH, text_chunks, ollama_model)
    st.success("✅ Vector store created and saved successfully!")

if os.path.exists(os.path.join(FAISS_DB_PATH, "index.faiss")):
    embeddings = OllamaEmbeddings(model=ollama_model)
    faiss_db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    faiss_db = None