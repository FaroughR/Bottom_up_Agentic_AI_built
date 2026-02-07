import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from RAG_query import get_rag_response


# 1. Initilise HuggingFace Inference Client (API key)
load_dotenv()

# 2. Initialise the Embedding model
# This must match the model used during the "Ingest" phase
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Connect the Chroma database
vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 4. Initilise the LLM Client (Hugging Face)
client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# 5. Streamlit UI Setup
st.set_page_config(page_title="Hitchens RAG chatbot", layout="wide")
st.title("📚 Caht with Ch Hitchens about 'God is not Great'")

# 2. Initialise Resources (Only do this once!)
@st.cache_resource
def load_resources():
    load_dotenv()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Path logic (adjust to your structure)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "chroma_db")
    
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    client = InferenceClient(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return vector_db, client

vector_db, client = load_resources()

# 3. Initialise Session State for History
if "history_string" not in st.session_state:
    st.session_state.history_string = ""

# 4. User Input
user_input = st.chat_input("Ask something...")

if user_input:
    # A. Show the user's message immediately
    st.chat_message("user").write(user_input)
    
    # B. Get the response from your RAG_query file
    with st.spinner("Thinking..."):
        answer = get_rag_response(
            user_input, 
            st.session_state.history_string, 
            vector_db, 
            client
        )
    
    # C. Show the AI's response
    st.chat_message("assistant").write(answer)
    
    # D. UPDATE THE LOCKER: Add this turn to the history
    st.session_state.history_string += f"User: {user_input}\nAssistant: {answer}\n\n"