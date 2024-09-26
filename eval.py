from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from langsmith.wrappers import wrap_openai
from langsmith import traceable
client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['messages']
    outputs = example.outputs
    # print(outputs['generations'][0]['text'])

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['generations'][0]['text']

    judge_prompt_1 = '''
    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. 
    Provide a score from 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.
    Also provide a brief explanation for your score.

    Respond in the following JSON format:
    {{
        "score": <int>,
        "explanation": "<string>"
    }}
    '''

    judge_prompt_2 = '''

    Based on the above information, your task is to provide a total rating score based on how well the model output complies with the system prompt and context of the conversation..
    Give your answer on a scale of 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.

    Here is the scale you should use:
    0: Completely non-compliant: The model output is completely unrelated to the system prompt and context of the conversation.
    1-3: Mostly non-compliant: The model output is mostly unrelated to the system prompt and context of the conversation.
    4-6: Partially compliant: The model output is partially related to the system prompt and context of the conversation.
    7-9: Mostly compliant: The model output is mostly related to the system prompt and context of the conversation.
    10: Perfectly compliant: The model output is perfectly related to the system prompt and context of the conversation.

    Provide a brief explanation for your score. 

    Respond in the following JSON format:
    {{
        "score": <int>,
        "explanation": "<string>"
    }}

    '''

    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    {judge_prompt_2}


    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "score": result["score"] / 10,  # Normalize to 0-1 range
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
        return {
            "key": "prompt_compliance",
            "score": 0,
            "reason": "Failed to parse evaluator response"
        }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "TTM"

# A string to prefix the experiment name with.
experiment_prefix = "Evidence summarizer for TTM articles"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)