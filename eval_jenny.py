from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable
openai_client = wrap_openai(OpenAI())

from langchain.evaluation import load_evaluator
from langsmith import Client

def generate_golden_ref() -> dict:
    golden_ref = {}

    # Initialize LangSmith client
    langsmith_client = Client()

    # Retrieve the LangSmith dataset for golden ref
    dataset_name = 'TTM golden ref'
    golden_data = langsmith_client.list_examples(dataset_name=dataset_name)

    # Iterate over the golden data and store ref data
    # Basic assumption of unique key for user input for now..
    for entry in golden_data:
        inputs = entry.inputs['input']
        human_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'human'), "")
        output = entry.outputs['output']['data']['content']
        # print(human_prompt)
        # print(output)
        golden_ref[human_prompt] = output
    
    return golden_ref

golden_ref_DB = generate_golden_ref()

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")
    # print(system_prompt)

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

            if (msg['type'] == 'human'):
                human_prompt = msg['data']['content']
                if (human_prompt in golden_ref_DB):
                    ref = golden_ref_DB[human_prompt]
                else:
                    ref = "No reference"


    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']

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

    # This judge compares the model output to the reference output
    judge_prompt_3 = '''
    Provide a score from 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.
    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. The Model Output should closely resemble and be as comprehensive as the Reference Output (if it exists).
    Also provide a brief explanation for your score. The golden reference for the perfect answer would be

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

    Reference Output: {ref}

    {judge_prompt_3}


    """

    response = openai_client.chat.completions.create(
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
# data = "TTM golden ref"
# data = "TTM RAG with reduced data"
# data = "TTM RAG with full data"
data = "TTM llama"

# A string to prefix the experiment name with.
experiment_prefix = "Evidence summarizer for TTM articles"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

for i in range(5):
    # Evaluate the target task
    results = evaluate(
        lambda inputs: inputs,
        data=data,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
    )

    print(results)