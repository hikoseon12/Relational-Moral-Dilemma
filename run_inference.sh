#!/usr/bin/env bash
# Run the 6 paper models across 3 perspectives x 3 independent runs.
#
# Required environment variables:
#   OPENAI_API_KEY      for o3-mini and gpt-5-mini
#   OPENROUTER_API_KEY  for everything else
# (See .env.example.)

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Warning: OPENAI_API_KEY is not set; OpenAI models will fail." >&2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Warning: OPENROUTER_API_KEY is not set; OpenRouter models will fail." >&2
fi

# Three perspectives x three independent runs (same prompt, separate output folder).
TEMPLATES=(
  'prescriptive_1' 'prescriptive_2' 'prescriptive_3'
  'descriptive_1'  'descriptive_2'  'descriptive_3'
  'action_1'       'action_2'       'action_3'
)

# 6 models reported in the paper.
MODELS=(
  'anthropic/claude-3.5-haiku'
  'google/gemini-2.5-pro'
  'gpt-5-mini-2025-08-07'
  'o3-mini-2025-01-31'
  'qwen/qwen3-30b-a3b-thinking-2507'
  'deepseek/deepseek-chat-v3.1'
)

REASONING="${REASONING:-}"  # set to "low" / "medium" / "high" to enable reasoning effort

for model in "${MODELS[@]}"; do
  for template in "${TEMPLATES[@]}"; do
    echo "Running template=$template model=$model reasoning=${REASONING:-(none)}"
    python run_inference.py \
      --template_type "$template" \
      --model_name "$model" \
      --reasoning "$REASONING"
  done
done
