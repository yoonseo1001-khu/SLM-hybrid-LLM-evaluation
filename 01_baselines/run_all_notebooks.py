import subprocess

notebooks = [
    "llama31/MedQA/llama31_MedQA_prompt_benchmark.ipynb",
    "mistral7b/MedQA/mistral7b_MedQA_prompt_benchmark.ipynb",
    "qwen2_7b/MedQA/qwen2_7b_MedQA_prompt_benchmark.ipynb"
]

for nb in notebooks:

    print("\n==============================")
    print("Running:", nb)
    print("==============================\n")

    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", nb,
        "--output", nb
    ])
