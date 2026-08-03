# 🩺 MedInsight

### Evaluating Retrieval-Augmented Vision-Language Models for Evidence-Grounded Medical Image Understanding

<p align="center">
  <img src="https://img.shields.io/badge/Research-Medical%20AI-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Task-Medical%20VQA-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

<p align="center">
A research framework testing whether <b>retrieval augmentation</b> improves the accuracy and evidence-grounding of Vision-Language Models on medical visual question answering.
</p>

<p align="center">
  <a href="https://zenodo.org/records/21670123"><b>📄 Paper</b></a> ·
  <a href="#-results">📊 Results</a> ·
  <a href="#-quick-start">🚀 Quick Start</a>
</p>

---

## 📌 Overview

MedInsight builds a **CLIP + FAISS** retrieval index over the ROCOv2 radiology corpus (~90K image-caption pairs) and uses it to supply retrieved evidence to a pretrained **BLIP-2** VLM, then compares that retrieval-augmented condition against the same model answering with no retrieved evidence — same checkpoint, same decoding, same scoring code, differing only in prompt content.

> ⚠️ **Research use only.** Not a clinical decision-support system.

---

## 📊 Results

Evaluated on the official **VQA-RAD** test split , k=5:

| Metric | Baseline | RAG | Δ |
|---|---|---|---|
| Closed-question accuracy | 43.0% | 47.8% | **+4.8 pts** |
| Overall exact match | 26.8% | 29.5% | **+2.7 pts** |
| Open-question BLEU-4 | 3.59% | 3.89% | +0.30 pts |
| Open-question ROUGE-L | 14.22% | 14.28% | +0.06 pts |

Retrieval improved every metric, but a paired bootstrap test (10,000 resamples) found neither delta statistically significant at α = 0.05 (closed exact match: p = 0.1848; open ROUGE-L: p = 0.9642). Reported as a finding, not hidden as a limitation.

A top-k sweep (1, 3, 5, 10) found closed-question accuracy peaks at **k=1 (55.0%)** and declines monotonically to k=10 (46.6%) — excess retrieved context appears to dilute rather than reinforce the model's answer.

Full analysis: [`docs/experiments_findings.md`](docs/experiments_findings.md) · [Paper on Zenodo](https://zenodo.org/records/21670123)

---

## ✨ Key Features

- Retrieval-augmented medical VQA pipeline with controlled baseline-vs-RAG comparison
- CLIP-based image retrieval over a FAISS vector index
- Paired bootstrap significance testing + qualitative error analysis
- FastAPI deployment for interactive inference
- IEEE-format paper documenting full methodology

---

## 🏗️ Architecture

```text
Image + Question → CLIP Encoder → FAISS Retrieval Index
                                          │
                        Retrieved ROCOv2 Captions
                                          │
                          Evidence Prompt Builder
                                          │
                                     BLIP-2 VLM
                                          │
                        Answer + Retrieved Evidence
```

Details: [`docs/architecture.md`](docs/architecture.md)

---

## 🚀 Quick Start

```bash
git clone https://github.com/Maryam024/MedInsight.git
cd MedInsight
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py --dataset all
python scripts/build_index.py
python scripts/run_baseline.py
python scripts/run_rag_eval.py
python scripts/run_experiments.py --top-k-values 1 3 5 10
```

Run the API: `uvicorn deployment.api.main:app --reload` → `localhost:8000/docs`
Run tests: `pytest tests -v`

---

## 📊 Datasets

| Purpose | Dataset | License |
|---|---|---|
| Retrieval corpus | ROCOv2 | CC BY-NC-SA 4.0 |
| Evaluation benchmark | VQA-RAD (official test split) | CC0 |

---

## ⚠️ Limitations

- Non-credentialed datasets only (no MIMIC-CXR); noted as future work
- Test split is small (451 examples) — significance testing is required, not optional
- Retriever embeds only the image, not the question — a likely factor in the k-sweep decline
- No clinician review of answer quality; scored on lexical/exact-match metrics only

---

## 📜 License

MIT. ROCOv2 and VQA-RAD retain their own licenses and are not redistributed here.

---

## 👩‍💻 Author

**Maryam Zaheer** — BS Computer Science, UET Lahore
Medical AI · Vision-Language Models · Retrieval-Augmented Generation

[Portfolio](https://maryam-dev.me) · [LinkedIn](https://linkedin.com/in/maryam-zaheer4) · [GitHub](https://github.com/Maryam024)

<p align="center">⭐ If you find this useful, consider starring the repo.</p>
