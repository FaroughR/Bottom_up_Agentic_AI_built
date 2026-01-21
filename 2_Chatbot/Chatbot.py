import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Load your .env file
# This looks for a file named '.env' in your folder and loads variables into the system environment.
load_dotenv()

# 2. Retrieve the API key from the environment.
# This keeps your secret key out of the actual code.
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 3. Initialize the client with your key
client = InferenceClient(api_key=api_key)

# 4. Initialize the conversation memory
messages = []
print("--- Chatbot Connected ---")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["quit", "exit", "close", "stop"]:
        break
    
    # Add what you said to the memory
    messages.append({"role": "user", "content": user_input})
    
    # Request a completion from the model
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct", 
        messages=messages,
        max_tokens=500
    )
    
    # Get the text from the response
    bot_reply = response.choices[0].message.content
    print(f"Bot: {bot_reply}")
    
    # Add what the bot said to the memory so it remembers the context later
    messages.append({"role": "assistant", "content": bot_reply})