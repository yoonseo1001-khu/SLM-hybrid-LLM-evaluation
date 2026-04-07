#!/bin/bash

cd ~/SLM_hybrid_project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_eval2

echo "===== START ALL EXPERIMENTS ====="

MODELS=("llama31" "mistral7b" "qwen2_7b")
METHODS=("zeroshot" "cot" "fewshot")

for MODEL in "${MODELS[@]}"
do
  for METHOD in "${METHODS[@]}"
  do
    echo "=============================="
    echo "Running: $MODEL | $METHOD"
    echo "=============================="

    python run_medqa_from_notebook.py --model $MODEL --method $METHOD

  done
done

echo "===== ALL DONE ====="
