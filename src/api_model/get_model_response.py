"""Dispatch a prompt to either OpenAI's Responses API or OpenRouter.

Authentication is read from environment variables only — no hardcoded keys.
Set ``OPENAI_API_KEY`` and/or ``OPENROUTER_API_KEY`` before calling.
"""

import json
import os

import requests
from openai import OpenAI


# Models served via OpenAI's Responses API (direct, not OpenRouter).
OPENAI_MODEL_LIST = {
    'gpt-5-mini-2025-08-07',
    'o3-mini-2025-01-31',
}

# Models served via OpenRouter.
OPENROUTER_LIST = {
    'anthropic/claude-3.5-haiku',
    'google/gemini-2.5-pro',
    'qwen/qwen3-30b-a3b-thinking-2507',
    'deepseek/deepseek-chat-v3.1',
}


def get_model_response(model_name, reasoning, prompt):
    if model_name in OPENAI_MODEL_LIST:
        return get_openai_model_response(model_name, reasoning, prompt)
    if model_name in OPENROUTER_LIST:
        return get_openrouter_model_response(model_name, reasoning, prompt)
    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        f"Add it to OPENAI_MODEL_LIST or OPENROUTER_LIST in {__file__}."
    )


def get_openai_model_response(model_name, reasoning, prompt):
    client = OpenAI()  # picks up OPENAI_API_KEY from env
    if 'gpt-5' in model_name and reasoning:
        response = client.responses.create(
            model=model_name,
            input=prompt,
            reasoning={'effort': reasoning},
            text={'verbosity': 'low'},
        )
    else:
        response = client.responses.create(model=model_name, input=prompt)
    return response, response.output_text


def get_openrouter_model_response(model_name, reasoning, prompt):
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    if reasoning:
        payload['reasoning'] = {'effort': reasoning}

    response = requests.post(
        url='https://openrouter.ai/api/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}'},
        data=json.dumps(payload),
    )

    try:
        resp_json = response.json()
        return resp_json, resp_json['choices'][0]['message']['content']
    except (ValueError, KeyError) as e:
        print('OpenRouter response parsing failed:', e)
        return None, None
