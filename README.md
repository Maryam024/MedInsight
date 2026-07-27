# 🩺 MedInsight

## Evaluating Retrieval-Augmented Vision-Language Models for Evidence-Grounded Medical Image Understanding

<p align="center">

<img src="https://img.shields.io/badge/Research-Medical%20AI-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Task-Medical%20VQA-green?style=for-the-badge">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge">
<img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge">

</p>


<p align="center">
A research framework exploring whether <b>retrieval augmentation</b> improves the factual grounding and reliability of Vision-Language Models for medical image understanding.
</p>


---

# 📌 Overview

MedInsight investigates whether Retrieval-Augmented Generation (RAG) can improve medical Vision-Language Models (VLMs) by providing relevant external evidence during answer generation.

The pipeline retrieves clinically relevant image-caption pairs from a radiology knowledge base using **CLIP embeddings + FAISS**, then provides retrieved evidence to a **BLIP-2 based VLM** for medical visual question answering.

> ⚠️ This project is developed strictly for research purposes.  
> It is **not a clinical decision-support system** and should not be used for medical diagnosis.


---

# ✨ Key Features

✅ Retrieval-Augmented Medical VQA Pipeline  
   
✅ Baseline vs RAG Performance Comparison  

✅ CLIP-based Medical Image Retrieval  

✅ FAISS Vector Database Integration  

✅ Evidence-grounded Answer Generation  

✅ Automated Evaluation Pipeline  

✅ Statistical Significance Testing  

✅ FastAPI Deployment Support  

✅ IEEE-format Research Paper  


---

# 🏗️ System Architecture


```text
                 Medical Image
                      +
                  Question
                      │
                      ▼

             CLIP Image Encoder

                      │

                      ▼

              FAISS Retrieval Index

                      │

                      ▼

       Relevant Radiology Image-Caption Pairs

                      │

                      ▼

             Evidence Prompt Builder

                      │

                      ▼

                  BLIP-2 VLM

                      │

                      ▼

          Answer + Retrieved Evidence
```


Detailed architecture:

```
docs/architecture.md
```


---

# 📂 Repository Structure


```text
MedInsight/
│
├── configs/              # Configuration files
├── data/                 # Dataset utilities
├── deployment/           # FastAPI deployment
├── docker/               # Docker configuration
├── docs/                 # Documentation
├── experiments/          # Experiment logs
├── notebooks/            # Research notebooks
├── paper/                # IEEE paper files
├── scripts/              # Experiment scripts
├── src/                  # Core implementation
├── tests/                # Unit tests
│
├── requirements.txt
├── README.md
└── LICENSE
```


---

# 🚀 Installation


## Clone Repository

```bash
git clone https://github.com/Maryam024/MedInsight.git

cd MedInsight
```


## Create Environment

```bash
python -m venv venv
```


Activate:


### Windows

```bash
venv\Scripts\activate
```


### Linux/macOS

```bash
source venv/bin/activate
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

# ⚡ Quick Start


## 1. Download datasets

```bash
python scripts/download_data.py --dataset all
```


## 2. Run exploratory analysis

```bash
python scripts/run_eda.py
```


## 3. Build retrieval index

```bash
python scripts/build_index.py
```


## 4. Evaluate baseline model

```bash
python scripts/run_baseline.py
```


## 5. Evaluate RAG model

```bash
python scripts/run_rag_eval.py
```


## 6. Run retrieval experiments

```bash
python scripts/run_experiments.py --top-k-values 1 3 5 10
```


## 7. Generate reports

```bash
python scripts/generate_report.py
```


---

# 📊 Datasets


| Purpose | Dataset |
|---|---|
| Retrieval Knowledge Base | ROCOv2 |
| Medical VQA Evaluation | VQA-RAD |


Dataset documentation:

```
docs/dataset.md
```


---

# 📈 Evaluation Metrics


MedInsight evaluates:

| Category | Metrics |
|-|-|
| Answer Quality | Exact Match, BLEU-4, ROUGE-L |
| Retrieval | Retrieval relevance analysis |
| Reliability | Bootstrap significance testing |
| Error Analysis | Failure case investigation |


Experiment results:

```
docs/experiments_findings.md
```


---

# 🌐 API Deployment


Start FastAPI server:


```bash
uvicorn deployment.api.main:app --reload
```


Open interactive documentation:


```
http://localhost:8000/docs
```


---

# 🧪 Testing


Run:


```bash
pytest tests -v
```


---

# 📄 Research Paper


The complete IEEE-format paper is available:


```
paper/MedInsight.pdf
```


Regenerate paper results:


```bash
python scripts/generate_paper_results.py
```


---

# 🔬 Research Contribution


MedInsight explores:

- Whether retrieval augmentation improves medical VLM grounding
- Whether retrieved evidence reduces unsupported answers
- How retrieval quality affects medical reasoning performance


---

# ⚠️ Limitations


Current limitations include:

- Research-scale evaluation datasets
- Dependence on pretrained VLM capabilities
- No clinical validation
- Not intended for real-world diagnosis


---

# 📚 Documentation


Available documentation:


| File | Description |
|-|-|
| architecture.md | System design |
| dataset.md | Dataset details |
| eda_findings.md | Exploratory analysis |
| experiments_findings.md | Experimental results |


---

# 📜 License


MIT License


Datasets such as ROCOv2 and VQA-RAD follow their own licensing terms and are not included.


---

# 👩‍💻 Author


**Maryam Zaheer**

BS Computer Science  
University of Engineering and Technology (UET) Lahore


Research Interests:

- Medical AI
- Vision-Language Models
- Retrieval-Augmented Generation
- Multimodal Learning


---

<p align="center">
⭐ If you find this research useful, consider starring the repository.
</p>
