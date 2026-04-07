#!/bin/bash

echo "===== START EXPERIMENTS ====="

cd ~/SLM_hybrid_project

# 환경 로드
source ~/.bashrc
conda activate llm_eval2

# 모델 / 방법 정의
MODELS=("llama31" "mistral7b" "qwen2_7b")
METHODS=("zeroshot" "cot" "fewshot")

# 로그 폴더
mkdir -p logs

for MODEL in "${MODELS[@]}"; do
  for METHOD in "${METHODS[@]}"; do

    echo "=============================="
    echo "Running: $MODEL | $METHOD"
    echo "=============================="

    LOG_FILE="logs/${MODEL}_${METHOD}.log"

    # 실행
    python run_fewshot_medqa.py --model $MODEL --method $METHOD > $LOG_FILE 2>&1

    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "SUCCESS: $MODEL $METHOD"

        # 자동 저장
        git add .
        git commit -m "Auto: $MODEL $METHOD completed"
        git push

    else
        echo "FAILED: $MODEL $METHOD → RETRY"

        # 재시도
        python run_fewshot_medqa.py --model $MODEL --method $METHOD >> $LOG_FILE 2>&1

        if [ $? -eq 0 ]; then
            echo "RETRY SUCCESS: $MODEL $METHOD"

            git add .
            git commit -m "Retry success: $MODEL $METHOD"
            git push

        else
            echo "FAILED AGAIN: $MODEL $METHOD"
        fi
    fi

  done
done

echo "===== ALL EXPERIMENTS DONE ====="
