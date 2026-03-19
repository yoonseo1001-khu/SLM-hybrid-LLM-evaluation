# SLM Hybrid LLM Evaluation for Clinical Question Answering

## 1. Introduction

Recent advances in large language models (LLMs) have demonstrated strong performance across biomedical and clinical reasoning tasks. However, their high computational cost, latency, and infrastructure requirements limit their applicability in real-time clinical environments, particularly in emergency and trauma care.

This project investigates an alternative paradigm: the use of small, task-focused language models (SLMs) for clinical question answering. The central hypothesis is that smaller models, when properly prompted and combined through hybrid architectures, can achieve competitive performance while maintaining efficiency and deployability.

---

## 2. Objectives

The objectives of this project are:

* To benchmark multiple small language models on clinical question answering tasks
* To evaluate prompting strategies: zero-shot, chain-of-thought (CoT), and few-shot
* To analyze model failure modes such as invalid outputs and reasoning inconsistencies
* To design hybrid model architectures for improved performance and efficiency

---

## 3. Dataset

The experiments use the MedQA dataset (BigBio version), consisting of USMLE-style multiple-choice clinical questions.

Each sample includes:

* A clinical question
* Four answer choices
* A ground-truth label

Dataset loading:

```python id="0q39gc"
dataset = load_dataset(
    "bigbio/med_qa",
    "med_qa_en_4options_bigbio_qa"
)
```

The dataset is dynamically loaded via HuggingFace to ensure reproducibility without storing large files locally.

---

## 4. Models

The following models are evaluated:

* LLaMA-3.1-8B-Instruct
* Mistral-7B
* Qwen2-7B

These models represent efficient, deployable alternatives to large frontier models.

---

## 5. Prompting Strategies

### 5.1 Zero-shot

Direct answer selection without intermediate reasoning.

### 5.2 Chain-of-Thought (CoT)

Step-by-step reasoning prior to final answer.

### 5.3 Few-shot

In-context examples to guide model predictions.

---

## 6. Methodology

The inference pipeline consists of:

1. Prompt construction
2. Model generation
3. Answer extraction (A/B/C/D)
4. Comparison with ground-truth
5. Accuracy aggregation

Batch inference is used for efficiency, and a unified pipeline ensures consistency across models and prompting strategies.

---

## 7. Project Structure

The project is organized to ensure modularity, reproducibility, and scalability.

```text id="9z3x9x"
SLM_hybrid_project/
│
├── 00_data/                         # Data management layer
│   ├── raw/                         # Original datasets (not stored)
│   ├── processed/                   # Preprocessed data (optional)
│   └── README.md                    # Dataset documentation
│
├── 01_baselines/                    # Baseline model experiments
│   ├── llama31/
│   │   └── MedQA/
│   │       └── llama31_MedQA_prompt_benchmark.ipynb
│   │
│   ├── mistral7b/
│   │   └── MedQA/
│   │       └── mistral7b_MedQA_prompt_benchmark.ipynb
│   │
│   └── qwen2_7b/
│       └── MedQA/
│           └── qwen2_7b_MedQA_prompt_benchmark.ipynb
│
├── 02_hybrid/                       # Hybrid model architectures
│   ├── routing/                    # Model selection logic
│   ├── cascade/                    # Multi-stage inference
│   ├── ensemble/                   # Prediction aggregation
│   └── .gitkeep
│
├── 03_results/                     # Experiment outputs
│   ├── baselines/
│   │   ├── llama31/
│   │   ├── mistral7b/
│   │   └── qwen2_7b/
│   │
│   ├── debug/                     # Sample outputs
│   └── summary/
│       └── MedQA_accuracy_summary.csv
│
├── 04_analysis/                    # Evaluation & visualization
│   ├── compare_models.ipynb
│   ├── error_analysis.ipynb
│   └── plots/
│
├── run_fewshot_medqa.py
└── README.md
```

---

## 8. Results

### 8.1 Overall Performance (MedQA)

The following table summarizes performance across all models and prompting strategies:

| Model      | Zero-shot | Few-shot | CoT    |
| ---------- | --------- | -------- | ------ |
| LLaMA-3.1  | 0.6096    | 0.6041   | 0.4014 |
| Mistral-7B | 0.5153    | 0.4941   | 0.3928 |
| Qwen2-7B   | 0.5169    | 0.4933   | 0.3645 |

---

### 8.2 Cross-Model Trends

Across all models, consistent patterns are observed:

* Zero-shot achieves the highest performance
* Few-shot provides similar but slightly lower performance
* Chain-of-Thought (CoT) consistently underperforms

This indicates that the observed behavior is systematic rather than model-specific.

---

### 8.3 Performance Gap

A clear performance hierarchy emerges:

```text id="n7b2l0"
Zero-shot ≈ Few-shot  >>  CoT
```

---

### 8.4 Model Comparison

* LLaMA-3.1 shows the strongest overall performance
* Mistral-7B and Qwen2-7B show similar performance levels
* CoT degradation is most severe in Qwen2-7B

---

## 9. Analysis

### 9.1 Failure of Chain-of-Thought in Small Models

Chain-of-Thought prompting leads to significant performance degradation across all models.

Possible causes:

* Limited reasoning capacity
* Drift in long generation sequences
* Increased hallucination

This suggests that reasoning-based prompting does not generalize well to smaller models.

---

### 9.2 Invalid Output Ratio

A large proportion of outputs are invalid, especially in CoT prompting.

Failure cases include:

* Missing answer choice
* Multiple answers
* Answer buried in long text

These errors directly reduce accuracy.

---

### 9.3 Output Length vs Efficiency

CoT significantly increases output length, leading to:

* Higher latency
* Increased token usage
* Higher parsing failure rate

Despite higher cost, no performance gain is observed.

---

### 9.4 Key Insight

```text id="6t7k4m"
Explicit reasoning (CoT) is not inherently beneficial for small language models and may degrade performance in structured clinical QA tasks.
```

---

### 9.5 Implications

* Simpler prompting strategies are more reliable
* Stability and format compliance are critical
* Efficiency should be prioritized over reasoning complexity

---

## 10. Future Work

Future work will focus on hybrid architectures that selectively apply reasoning:

* Confidence-based routing
* Cascaded inference pipelines
* Ensemble-based prediction

These approaches aim to combine efficiency and reasoning capability.

---

## 11. Reproducibility

* Dataset loaded via HuggingFace
* Modular inference pipeline
* JSON-based result storage
* Fully reproducible experiments

---

## 12. Author

Yoonseo Lee
AI Researcher in Clinical NLP and Language Model Systems

---

