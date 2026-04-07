import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(path, save_dir):

    df = pd.read_csv(path)

    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    calibration = train.sample(frac=0.1, random_state=42)

    os.makedirs(save_dir, exist_ok=True)

    train.to_csv(f"{save_dir}/train.csv", index=False)
    val.to_csv(f"{save_dir}/val.csv", index=False)
    test.to_csv(f"{save_dir}/test.csv", index=False)
    calibration.to_csv(f"{save_dir}/calibration.csv", index=False)

    print(f"Saved splits in {save_dir}")


# MedQA
split_dataset(
    "00_data/splits/medqa/train.csv",
    "00_data/splits/medqa_final"
)

# JAMA
split_dataset(
    "00_data/splits/jama_annotated.csv",
    "00_data/splits/jama_final"
)
