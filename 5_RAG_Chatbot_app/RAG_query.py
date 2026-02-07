# This function retireoved the RAG reposnse given a user query, conversation history, vector database and LLM client

def get_rag_response(user_query, conversation_history=None, vector_db=None, client=None):
    # Core RAG Function: Search DB -> Get Contect -> Ask LLM -> Return Answer

    # A. find the three top most relevant snippets from the PDF
    docs = vector_db.similarity_search(user_query, k=3)

    # B. Context Assembly: Combine snippets into one string.
    retrieved_context = "\n".join([d.page_content for d in docs])

    #  C. Prompt Eng: tell LLM to use the context
    structured_prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    Context: {retrieved_context}
    conversation_history
    User: {user_query}
    Assistant:
    """

    # D. Generation: Send to Hugging Face model
    output = client.chat_completion(
        model="google/gemma-2-9b-it",
        messages=[{"role":"user", "content": structured_prompt}],
        max_tokens=500
    )

    return output.choices[0].message.content
