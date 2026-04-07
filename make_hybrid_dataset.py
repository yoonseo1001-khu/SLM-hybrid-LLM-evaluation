import pickle
import random

baseline_path = "medqa_triplets.pkl"
llm_path = "triplets_llm.pkl"

baseline = pickle.load(open(baseline_path, "rb"))
llm = pickle.load(open(llm_path, "rb"))

print("Baseline:", len(baseline))
print("LLM:", len(llm))


def make_hybrid(ratio_baseline=0.7, save_path="hybrid.pkl"):
    total_size = min(len(baseline), len(llm))

    n_base = int(total_size * ratio_baseline)
    n_llm = total_size - n_base

    random.shuffle(baseline)
    random.shuffle(llm)

    hybrid = baseline[:n_base] + llm[:n_llm]
    random.shuffle(hybrid)

    pickle.dump(hybrid, open(save_path, "wb"))

    print(f"Saved {save_path}")
    print(f"Baseline: {n_base}, LLM: {n_llm}")


make_hybrid(0.9, "hybrid_9_1.pkl")
make_hybrid(0.7, "hybrid_7_3.pkl")
make_hybrid(0.5, "hybrid_5_5.pkl")
