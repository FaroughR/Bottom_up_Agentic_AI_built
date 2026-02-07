import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
print("--- RAG Chatbot Connected (Reading: Christopher Hitchens) ---")

# 5. Ask for a user prompt
user_query = input("Ask a question about God is not great:")

# 6. find the three top most relevant to the prompt snippets from the PDF
docs = vector_db.similarity_search(user_query, k=10)

# 7. Context Assembly: Combine snippets into one string.
retrieved_context = "\n".join([d.page_content for d in docs])

#  8. Prompt Eng: tell LLM to use the context
structured_prompt = f"""
You are a helpful assistant. Use the following context to answer the question.
Context: {retrieved_context}
Question: {user_query}
Answer:
"""

# 9. Generation: Send to Hugging Face model
response = ""
output = client.chat_completion(
    model="google/gemma-2-9b-it",
    messages=[{"role":"user", "content": structured_prompt}],
    max_tokens=500
)

RAG_asnwer = output.choices[0].message.content
print(RAG_asnwer)