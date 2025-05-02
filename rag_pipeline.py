from langchain_groq import ChatGroq
from vector_db import faiss_db
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the GROQ API key from the environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the key is present
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY is not set in the environment variables.")

# Set the API key to the environment variable for Groq API
os.environ["GROQ_API_KEY"] = groq_api_key

# Initialize the LLM model using Groq
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Function to retrieve documents from FAISS vector store
def retrieve_docs(query):
    if faiss_db is None:
        raise ValueError("❌ Vector store not found. Please upload and process a PDF first.")
    return faiss_db.similarity_search(query)

# Function to create context from the retrieved documents
def create_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Custom prompt to guide the model's response
custom_prompt = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Question: {question}
Context: {context}
Answer:
"""

# Function to answer the user's query using the context
def answer_query(documents, model, query):
    context = create_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

# === Example Query ===
query = "Which articles are violated when a government forbids the right to assemble peacefully"

# Retrieve the relevant documents
relevant_docs = retrieve_docs(query)

# Get the answer using the retrieved documents
print("AI Lawyer:", answer_query(relevant_docs, llm_model, query))
