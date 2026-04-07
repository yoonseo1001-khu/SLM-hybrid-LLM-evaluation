#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

cd /gpfs/data/oermannlab/gaifl/users/yl14814/SLM_hybrid_project

source /gpfs/data/oermannlab/gaifl/users/yl14814/miniconda3/etc/profile.d/conda.sh
conda activate llm_eval2

echo "GPU check:"
nvidia-smi

echo "START TRAIN"
python train_medqa.py
