import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient

# 1. Load your .env file
load_dotenv(find_dotenv())

# 2. Initialize the official client
# It will automatically find HUGGINGFACEHUB_API_TOKEN in your .env
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=api_key)

# 3. Ask a question
# We use the chat completion method because it's the standard for 2026
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct", # Updated to a 2026-standard small model
    messages=[{"role": "user", "content": "What are the three rules of robotics?"}],
    max_tokens=500,
)

# 4. Print the answer
print(response.choices[0].message.content)