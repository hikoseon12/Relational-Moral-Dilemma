"""Mformer-based moral foundation scoring.

Uses Josh Nguyen's joshnguyen/mformer-{foundation} classifiers
(https://github.com/joshnguyen99/moral_axes) to score the model's free-text
reasoning along five MFT foundations: care, fairness, loyalty, authority, sanctity.

Inputs : results/{perspective}/{model}.csv  (raw inference output)
Outputs: results_mformer/{perspective}/{model}.csv  (with *_mscore columns)
"""

import json
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def extract_answer_reason(text):
    """Pull (answer, reason) out of a possibly-malformed JSON string."""
    if pd.isna(text):
        return None, None
    try:
        text = text.replace("\\'", "'")
        text = ''.join(c if c >= ' ' else ' ' for c in text)  # strip control chars
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None, None
        json_str = match.group(0)
        data = json.loads(json_str)
        return data.get('answer'), data.get('reason') or data.get('reasoning')
    except Exception as e:
        print('extract_answer_reason error:', e)
        print(text)
        return None, None


def load_input_df(perspective, model_name, results_dir='results'):
    df = pd.read_csv(os.path.join(results_dir, perspective, f'{model_name}.csv'))

    reason_column = 'reasoning' if 'reasoning' in list(df.columns) else 'reason'

    df[['answer', 'reason']] = df['response'].apply(
        lambda x: pd.Series(extract_answer_reason(x))
    )
    df = df.dropna(subset=['answer', reason_column]).copy().reset_index(drop=True)
    return df, df[[reason_column]]


def load_model_tokenizer(foundation='authority'):
    model_name = f'joshnguyen/mformer-{foundation}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map='auto')
    return model, tokenizer


def get_moral_prob(model, tokenizer, instances):
    inputs = tokenizer(instances, padding=True, truncation=True, return_tensors='pt').to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[:, 1]
    return probs.detach().cpu().tolist()


def get_model_perspective_moral_list():
    model_names = [
        'claude-3.5-haiku',
        'gemini-2.5-pro',
        'gpt-5-mini-2025-08-07',
        'o3-mini-2025-01-31',
        'qwen3-30b-a3b-thinking-2507',
        'deepseek-chat-v3.1',
    ]
    # By default we score the run-1 outputs; loop over runs externally if needed.
    perspectives = [
        'prescriptive_1',
        'descriptive_1',
        'action_1',
    ]
    moral_values = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
    return model_names, perspectives, moral_values


def load_model_reasoning_df(model_name, perspective, mformer_dict, tokenizer_dict,
                             moral_values, results_dir='results'):
    df, _ = load_input_df(perspective, model_name, results_dir=results_dir)
    instances = df['reason'].tolist()
    for foundation in moral_values:
        model, tokenizer = mformer_dict[foundation], tokenizer_dict[foundation]
        df[f'{foundation}_mscore'] = get_moral_prob(model, tokenizer, instances)
    return df


def get_model_result_df(model_names, perspectives, moral_values,
                         results_dir='results', out_dir='results_mformer'):
    mformer_dict, tokenizer_dict = {}, {}
    for foundation in moral_values:
        model, tokenizer = load_model_tokenizer(foundation)
        mformer_dict[foundation] = model
        tokenizer_dict[foundation] = tokenizer

    for model_name in model_names:
        for perspective in perspectives:
            score_df = load_model_reasoning_df(
                model_name, perspective, mformer_dict, tokenizer_dict, moral_values,
                results_dir=results_dir,
            )
            os.makedirs(os.path.join(out_dir, perspective), exist_ok=True)
            out_path = os.path.join(out_dir, perspective, f'{model_name}.csv')
            score_df.to_csv(out_path, index=False)
            print(f'Saved -> {out_path}  ({len(score_df)} rows)')


if __name__ == '__main__':
    model_names, perspectives, moral_values = get_model_perspective_moral_list()
    get_model_result_df(model_names, perspectives, moral_values)
