import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. setup
load_dotenv()
client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# 2. Load the vector database (Knowledge Base) created in Ingest_CreateDatabase.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

print("--- RAG Chatbot Connected (Reading: Christopher Hitchens) ---")

