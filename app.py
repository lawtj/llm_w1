import os
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
import openai

from langsmith.wrappers import wrap_openai
from langsmith import traceable


from prompts import SYSTEM_PROMPT, scraped_markdown

api_key_openai = os.getenv("OPENAI_API_KEY")

#initialize the client
client = wrap_openai(openai.AsyncClient(api_key=api_key_openai, 
                            base_url="https://api.openai.com/v1", 
                            ))

#use this part in the client.chat.completions.create() method
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.3,
    "max_tokens": 1500
}
@traceable
@cl.on_message
async def on_message(message):

    #get the history. check if message history is blank, and if it is, insert the system prompt
    message_history = cl.user_session.get('messages', [])
    if not message_history:
        message_history.insert(0, {"role": "system", "content": SYSTEM_PROMPT + scraped_markdown})
    
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

