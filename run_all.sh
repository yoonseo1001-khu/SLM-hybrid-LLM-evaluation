#!/bin/bash

cd ~/SLM_hybrid_project
conda activate llm_eval2

echo "Running experiments..."
python run_all_notebooks.py

echo "Saving results..."
git add .
git commit -m "Auto experiment run"
git push
