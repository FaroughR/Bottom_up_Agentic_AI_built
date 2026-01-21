# Import necessary libraries
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load your PDF (Use the full path we discussed if it fails)
loader = PyPDFLoader("/Users/faroughrahimzadeh/Desktop/Udemy_Langchain_RAG_LLM_Course/LangChain_agent/Christopher Hitchens - God Is Not Great .pdf") 
data = loader.load()

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in ChromaDB (Fixed line)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, # In langchain-chroma, this is 'embedding' again
    persist_directory="./chroma_db"
)

print("Finished! The 'chroma_db' folder has been created.")