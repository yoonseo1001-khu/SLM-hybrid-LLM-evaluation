#!/bin/bash
#SBATCH --job-name=triplet
#SBATCH --output=triplet.out
#SBATCH --error=triplet.err
#SBATCH --partition=gpu4_short
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source /gpfs/data/oermannlab/gaifl/users/yl14814/miniconda3/etc/profile.d/conda.sh
conda activate llm_eval2

echo "Python path:"
which python

echo "Start running..."

python generate_triplet.py
