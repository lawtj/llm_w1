import os
from dotenv import load_dotenv
import json

load_dotenv()

import chainlit as cl
import openai

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownTextSplitter

from prompts import SYSTEM_PROMPT
from data_sources import LANGCHAIN_DATA, SCRAPED_MARKDOWN
from functions import get_citation_count

# initialize the client
client = wrap_openai(
    openai.AsyncClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )
)

# use this part in the client.chat.completions.create() method
open_ai_model = "gpt-4o-mini"
model_kwargs = {"model": open_ai_model, "temperature": 0.3, "max_tokens": 1500}

### RAG PARAMETERS ###
# If true, it'll generate golden answers
GENERATE_GOLDEN_ANSWERS = False

# If true, user history is saved
HISTORY_ON = True
### END RAG PARAMETERS ###

# vars for rag indexing
retriever = None

@traceable
@cl.on_chat_start
async def start_main():
    text_splitter = MarkdownTextSplitter(chunk_size=20000, chunk_overlap=2000)
    splits = text_splitter.split_documents(LANGCHAIN_DATA)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    global retriever
    retriever = FAISS.from_documents(documents=splits, embedding=embedding_model)


def retrieve_relevant_docs(query, retriever, k=10):
    context = ""
    # Vectorstore returns the most similar documents based on the query
    relevant_docs = retriever.similarity_search(query, k=k)
    # Concatenate the content of the retrieved documents
    for doc in relevant_docs:
        print(doc.metadata['source'])
        context += "\n\n".join([doc.page_content])
    return context


async def generate_response(message_history):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **model_kwargs
    )
    async for completion in stream:
        if token := completion.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
    return response_message.content


@traceable
@cl.on_message
async def on_message(message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("messages", [])
    # Check if message history is blank, and if it is, insert the system prompt
    if not message_history or message_history[0].get("role") != "system":
        message_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    # add user message to message history
    message_history.append({"role": "user", "content": message.content})

    if GENERATE_GOLDEN_ANSWERS:
        # add reduced paper data fed directly into system prompt for comparison
        doc_context = SCRAPED_MARKDOWN
    else:
        # get relevant docs from rag/index
        doc_context = retrieve_relevant_docs(message.content, retriever)

    if len(doc_context) > 0:
        # if previous content is present, remove it. This is because we only want to keep relevant documents in the prompt.
        if len(message_history) > 2 and message_history[1].get("role") == "system":
            message_history.pop(1)
        message_history.insert(1, {"role": "system", "content": doc_context})

    # generate the response
    response_content = await generate_response(message_history)

    if response_content.startswith("{"):
        function_call = json.loads(response_content)
        function_name = function_call.get("function")
        if function_name == "get_citation_count":
            query = function_call.get("parameters").get("query")
            citation_count = await get_citation_count(query)
            paper_title = citation_count[0]['title']
            citation_count = citation_count[0]['inline_links']['cited_by']['total']
            message_history.append({"role": "system", "content": f"Here are the results of the user's query: the paper '{paper_title}' has been cited {citation_count} times"})
            response_content = await generate_response(message_history)

    # now append the assistant response to the message history
    message_history.append({"role": "assistant", "content": response_content})

    # save history to user session
    if not HISTORY_ON:
        cl.user_session.set("messages", message_history)
