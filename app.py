import os
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
import openai

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

from prompts import SYSTEM_PROMPT, DATA

api_key_openai = os.getenv("OPENAI_API_KEY")

#initialize the client
client = wrap_openai(openai.AsyncClient(api_key=api_key_openai,
                            base_url="https://api.openai.com/v1",
                            ))

#use this part in the client.chat.completions.create() method
open_ai_model = "chatgpt-4o-latest"
model_kwargs = {
    "model": open_ai_model,
    "temperature": 0.3,
    "max_tokens": 1500
}

# vars for rag indexing
retriever = None
llama_index_location = './data_index_llama/'

@cl.on_chat_start
async def start_main():

    # Load dataset from local file if it exists, otherwise creates a new index
    if os.path.exists(llama_index_location):
        storage_context = StorageContext.from_defaults(persist_dir=llama_index_location)
        # load index
        index = load_index_from_storage(storage_context)
    else:
        # Generate dataset if local file doesn't exist
        # Load documents from a directory (you can change this path as needed)
        index = VectorStoreIndex.from_documents(DATA)
        index.storage_context.persist(persist_dir=llama_index_location)
    global retriever
    retriever = index.as_retriever(retrieval_mode='similarity', k=3)

@traceable
@cl.on_message
async def on_message(message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get('messages', [])

    # Check if message history is blank, and if it is, insert the system prompt
    if not message_history or message_history[0].get("role") != "system":
        message_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # get relevant docs from rag/index
    relevant_docs = retriever.retrieve(message.content)
    doc_content = ""
    for i, doc in enumerate(relevant_docs):
        doc_content += doc.node.get_content()
    if len(doc_content) > 0:
        # if previous content is present, remove it. This is because we only want to keep relevant documents in the prompt.
        if len(message_history) > 2 and message_history[1].get("role") == "system":
            message_history.pop(1)
        message_history.insert(1, {"role": "system", "content": doc_content})

    # now append the user message to the message history
    message_history.append({"role": "user", "content": message.content})

    #initialize response message
    response_message = cl.Message(content="")
    await response_message.send()

    #generate the response
    stream = await client.chat.completions.create(messages=message_history, stream=True, **model_kwargs)
    async for completion in stream:
        if token := completion.choices[0].delta.content or "":
            await response_message.stream_token(token)

    #now append the assistant response to the message history
    message_history.append({"role": "assistant", "content": response_message.content})

    #save history to user session
    cl.user_session.set('messages', message_history)

    await response_message.update()

