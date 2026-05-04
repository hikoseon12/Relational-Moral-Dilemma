"""Run the Whistleblower's Dilemma prompt grid against one model.

Outputs are written to:
  results/{template_type}/{model_short_name}[_{reasoning}].csv

The prompt grid is identical for run 1, 2, 3 of the same perspective
(e.g. action_1/_2/_3) — the suffix only affects which output directory
the responses go to.
"""

import argparse
import concurrent.futures
import os

import pandas as pd
from tqdm import tqdm

from src.api_model.get_model_response import get_model_response
from src.prompt.whistleblowing_prompt import generate_prompts
from src.utils import save_csv


def add_model_response_parallel(df, model_name, reasoning, max_workers=8):
    def process_prompt(prompt):
        try:
            raw_response, response = get_model_response(model_name, reasoning, prompt)
            return (model_name, response, raw_response)
        except Exception as e:
            error_message = str(e)
            return (model_name, error_message, error_message)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_prompt, df['prompt']),
            total=len(df),
            desc='Processing prompts',
            unit='prompt',
        ))

    df['model'], df['response'], df['raw_response'] = zip(*results)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help="e.g. anthropic/claude-3.5-haiku, openai/gpt-5-mini-2025-08-07")
    parser.add_argument('--template_type', type=str, required=True,
                        help="e.g. prescriptive_1, descriptive_2, action_3")
    parser.add_argument('--reasoning', type=str, default='',
                        help="reasoning effort for reasoning-capable models: minimal/low/medium/high")
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--prompts_dir', type=str, default='data/prompts')
    parser.add_argument('--sample', type=int, default=0,
                        help="If >0, randomly sample this many prompts for a quick test run.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed used by --sample.")
    args = parser.parse_args()

    df = generate_prompts(args.template_type, prompts_dir=args.prompts_dir)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} prompts (seed={args.seed}).")
    print(df)

    df_with_responses = add_model_response_parallel(df, args.model_name, args.reasoning)
    print(df_with_responses.head())

    save_model_name = args.model_name.split('/')[-1]
    save_dir = os.path.join(args.results_dir, args.template_type)
    os.makedirs(save_dir, exist_ok=True)

    if args.reasoning:
        save_path = os.path.join(save_dir, f'{save_model_name}_{args.reasoning}.csv')
    else:
        save_path = os.path.join(save_dir, f'{save_model_name}.csv')
    save_csv(save_path, df_with_responses)
    print(f'Saved -> {save_path}')


if __name__ == '__main__':
    main()
