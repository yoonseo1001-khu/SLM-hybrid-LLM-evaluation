import subprocess

models = ["llama31", "mistral7b", "qwen2_7b"]
datasets = ["MedQA", "HealthBench", "ReMedE"]

# notebook 이름 규칙 (현재 구조 기준)
NOTEBOOK_NAME = {
    "MedQA": "llama31_MedQA_prompt_benchmark.ipynb",
    "HealthBench": "llama31_HealthBench_prompt_benchmark.ipynb",
    "ReMedE": "llama31_ReMedE_prompt_benchmark.ipynb"
}

for model in models:
    for dataset in datasets:

        notebook_path = f"{model}/{dataset}/{NOTEBOOK_NAME[dataset]}"

        print("\n==============================")
        print("Running:", notebook_path)
        print("==============================\n")

        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", notebook_path,
            "--output", notebook_path
        ])
