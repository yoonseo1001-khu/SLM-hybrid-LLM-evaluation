import json
import matplotlib.pyplot as plt

# load
data = json.load(open("../03_results/medqa_final/final.json"))

baseline = data["baseline"]
hybrid = data["hybrid"]
curriculum = data["curriculum"]

plt.figure(figsize=(6,4))

plt.plot(baseline, label="Baseline")
plt.plot(hybrid, label="Hybrid (Hard Negative)")
plt.plot(curriculum, label="Curriculum")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Triplet Training on MedQA")

plt.legend()
plt.grid(True)

plt.savefig("../03_results/medqa_final/final_plot_pretty.png", dpi=300)
