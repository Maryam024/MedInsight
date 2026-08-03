# Paper Overview

| Section | Description |
|---|---|
| Abstract | Summary of the research problem, methodology, results, and conclusions. |
| I. Introduction | Motivation, research objectives, and contributions. |
| II. Related Work | Medical vision-language models, retrieval-augmented generation, and medical VQA. |
| III. Datasets | Overview of the ROCOv2 retrieval corpus and VQA-RAD evaluation benchmark. |
| IV. Method | Baseline model, retrieval pipeline, and retrieval-augmented architecture. |
| V. Experimental Setup | Evaluation metrics, hardware, datasets, and implementation details. |
| VI. Results | Quantitative comparison between the baseline and RAG models. |
| VII. Discussion | Interpretation of results, error analysis, and retrieval quality. |
| VIII. Limitations | Current limitations and future improvements. |
| IX. Conclusion | Summary of findings and future research directions. |
| References | IEEE-formatted bibliography. |

## Title

**MedInsight: Evaluating Retrieval-Augmented Vision-Language Models for Evidence-Grounded Medical Image Understanding**

## Contributions

1. A reproducible retrieval-augmented medical VQA pipeline built using openly available datasets.
2. A comparison of baseline and retrieval-augmented vision-language models on the VQA-RAD benchmark.
3. A modular, configurable codebase designed for future extension to larger medical imaging datasets.

## Reproducing the Paper

```bash
python scripts/run_baseline.py
python scripts/run_rag_eval.py
python scripts/run_experiments.py --top-k-values 1 3 5 10
python scripts/generate_paper_results.py
```

The generated results are written to `paper/sections/results.tex` and included automatically when compiling the paper.

Compile the paper using the IEEEtran LaTeX template (e.g., Overleaf or a local LaTeX installation).