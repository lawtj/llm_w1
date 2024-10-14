import os
from dotenv import load_dotenv
import json
import litellm
litellm.success_callback = "langsmith"

load_dotenv()

import chainlit as cl
import openai

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from prompts import SYSTEM_PROMPT, SHOULD_FETCH_NEW_DOCS_PROMPT
from data_sources import TTM, STEROID_SUMMARIES, DIALYSIS_SUMMARIES
from functions import get_citation_count

# Define the model
open_ai_model = "llama3.1"
model_kwargs = {"model": open_ai_model, "temperature": 0, "max_tokens": 1500}

# Initialize the client based on the model
client = wrap_openai(
    openai.AsyncClient(
        api_key=f"{'sk-1234' if open_ai_model not in ['gpt-4o-mini', 'gpt-4o'] else os.getenv('OPENAI_API_KEY')}",
        base_url=f"{'http://localhost:4000' if open_ai_model not in ['gpt-4o-mini', 'gpt-4o'] else 'https://api.openai.com/v1'}"
    )
)

### RAG PARAMETERS ###
# If true, it'll generate golden answers
GENERATE_GOLDEN_ANSWERS = False

# Select the reference dataset for golden answers
# Options: "TTM", "STEROID_SUMMARIES", "DIALYSIS_SUMMARIES"
GOLDEN_ANSWER_DATASET = "TTM"

# If true, user history is saved
HISTORY_ON = True
### END RAG PARAMETERS ###

# vars for rag indexing
retriever = None

@traceable
@cl.on_chat_start
async def start_main():
    global retriever
    embedding_model = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2"             # Lightweight and fast
        # model_name="sentence-transformers/all-MPNet-base-v2"            # Higher quality for semantic tasks
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"    # Optimized for QA
    )
    retriever = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

def retrieve_relevant_docs(query, retriever, k=10):
    # Vectorstore returns the most similar documents based on the query
    return retriever.similarity_search(query, k=k)
    # return retriever.similarity_search("TTM", k=10) # Compare to basic keyword search

def create_doc_context(relevant_docs):
    # Concatenate the content of the retrieved documents
    context = ""
    print("DEBUG number of docs used: ", len(relevant_docs))
    for doc in relevant_docs:
        print("DEBUG doc used: ", doc.metadata['source'])
        metadata = doc.metadata
        source = metadata.get("source", "Unknown source")
        doc_id = metadata.get("doc_id", "Unknown ID")
        context += f"\n\nSource: {source}\nDocument ID: {doc_id}\n{doc.page_content}"
    return context

def get_golden_doc_context():
    # Select the appropriate dataset based on the GOLDEN_ANSWER_DATASET variable
    if GOLDEN_ANSWER_DATASET == "TTM":
        return TTM
    elif GOLDEN_ANSWER_DATASET == "STEROID_SUMMARIES":
        return STEROID_SUMMARIES
    elif GOLDEN_ANSWER_DATASET == "DIALYSIS_SUMMARIES":
        return DIALYSIS_SUMMARIES
    else:
        raise ValueError(f"Invalid GOLDEN_ANSWER_DATASET: {GOLDEN_ANSWER_DATASET}")

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

documents = []

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

    global documents
    # get relevant docs from rag/index
    retrieved_docs = retrieve_relevant_docs(message.content, retriever)
    documents = retrieved_docs

    # if previous content is present, remove/replace it.
    if len(message_history) > 2 and message_history[1].get("role") == "system":
        message_history.pop(1)
    message_history.insert(1, {"role": "system", "content": get_golden_doc_context() if GENERATE_GOLDEN_ANSWERS else create_doc_context(documents)})

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
    if HISTORY_ON:
        cl.user_session.set("messages", message_history)

@traceable
async def fetch_new_docs(message_history):
    new_topic_message_history = [{"role": "system", "content": SHOULD_FETCH_NEW_DOCS_PROMPT}, {"role": "user", "content": message_history[-1]["content"]}]

    response_content = await generate_response(new_topic_message_history)
    print("DEBUG fetch_new_docs ", response_content)

    if response_content.strip().startswith('{'):
        response = json.loads(response_content.strip())
        return response['fetch_new_docs']
    else:
        return True
