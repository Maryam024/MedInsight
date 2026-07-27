# MedInsight

**Evaluating Retrieval-Augmented Vision-Language Models for Evidence-Grounded Medical Image Understanding**

MedInsight is a research project that investigates whether retrieval augmentation improves the performance of vision-language models (VLMs) on medical visual question answering (VQA). The system retrieves relevant radiology image-caption pairs from a FAISS index and uses them as supporting evidence during answer generation.

> **Note:** This project is intended for research purposes only and is **not** a clinical decision-support system.

---

## Features

- Retrieval-augmented medical VQA pipeline
- Baseline and RAG model comparison
- CLIP-based image retrieval with FAISS
- Automated evaluation and benchmarking
- FastAPI deployment
- IEEE-format research paper

---

## System Architecture

```text
Medical Image + Question
           │
           ▼
   CLIP Image Encoder
           │
           ▼
      FAISS Retrieval
           │
           ▼
 Retrieved Image-Caption Pairs
           │
           ▼
   Prompt Construction
           │
           ▼
      BLIP-2 Model
           │
           ▼
 Generated Answer + Evidence
```

A detailed description of the pipeline is available in `docs/architecture.md`.

---

## Repository Structure

```text
MedInsight/
├── configs/
├── data/
├── deployment/
├── docker/
├── docs/
├── experiments/
├── notebooks/
├── paper/
├── scripts/
├── src/
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Maryam024/MedInsight.git
cd MedInsight

python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Quick Start

Run the complete pipeline:

```bash
# Download datasets
python scripts/download_data.py --dataset all

# Exploratory data analysis
python scripts/run_eda.py

# Build the retrieval index
python scripts/build_index.py

# Evaluate baseline model
python scripts/run_baseline.py

# Evaluate retrieval-augmented model
python scripts/run_rag_eval.py

# Run additional experiments
python scripts/run_experiments.py --top-k-values 1 3 5 10

# Generate experiment report
python scripts/generate_report.py

# Generate LaTeX tables for the paper
python scripts/generate_paper_results.py
```

---

## Datasets

| Purpose | Dataset |
|---------|---------|
| Retrieval Corpus | ROCOv2 |
| Evaluation Benchmark | VQA-RAD |

See `docs/dataset.md` for dataset details and licensing information.

---

## Evaluation

The models are evaluated using:

- Exact Match Accuracy
- BLEU-4
- ROUGE-L
- Bootstrap Statistical Significance
- Retrieval Quality Analysis
- Error Analysis

Generated reports are saved in:

```text
docs/experiments_findings.md
```

---

## API

Launch the FastAPI server:

```bash
uvicorn deployment.api.main:app --reload
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

## Testing

Run all unit tests:

```bash
pytest tests -v
```

---

## Documentation

Additional documentation is available in the `docs/` directory:

- `architecture.md`
- `dataset.md`
- `eda_findings.md`
- `experiments_findings.md`

---

## Paper

The IEEE-format research paper is available in the `paper/` directory.

To regenerate the experimental results used in the paper:

```bash
python scripts/generate_paper_results.py
```

---

## License

This project is licensed under the MIT License. The datasets used by this project (e.g., ROCOv2 and VQA-RAD) are distributed under their own licenses and are **not** included in this repository.

---

## Author

**Maryam Zaheer**  
BS Computer Science  
University of Engineering and Technology (UET) Lahore