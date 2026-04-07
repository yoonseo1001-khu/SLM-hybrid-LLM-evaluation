import pandas as pd

df = pd.read_csv("00_data/splits/jama_train.csv")

# 🔥 percentile 기반 (핵심)
lengths = df["question"].apply(len)

low = lengths.quantile(0.33)
high = lengths.quantile(0.66)

def label_difficulty(text):
    l = len(text)

    if l < low:
        return "easy"
    elif l < high:
        return "medium"
    else:
        return "hard"

df["difficulty"] = df["question"].apply(label_difficulty)

df.to_csv("00_data/splits/jama_annotated.csv", index=False)

print("Saved:", len(df))
print(df["difficulty"].value_counts())
