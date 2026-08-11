# Dataset Documentation

## Dataset Overview

MedInsight uses two datasets for different purposes:

| Role                 | Dataset     | Size                                     | License         |
| -------------------- | ----------- | ---------------------------------------- | --------------- |
| Retrieval corpus     | **ROCOv2**  | ~90,000 image-caption pairs              | CC BY-NC-SA 4.0 |
| Evaluation benchmark | **VQA-RAD** | 3,515 QA pairs from 315 radiology images | CC0             |

## Dataset Selection

### ROCOv2

ROCOv2 is used as the retrieval corpus because it contains a large collection of radiology image-caption pairs collected from PubMed Central. The dataset covers several imaging modalities, including CT, MRI, X-ray, and ultrasound.

Its size makes it suitable for building a FAISS-based retrieval index while still being practical to process on a single machine.

### VQA-RAD

VQA-RAD is used as the evaluation benchmark because it is a commonly used dataset for medical visual question answering. It contains clinician-authored questions and answers, allowing the baseline and retrieval-augmented approaches to be evaluated under the same benchmark.

### Why Not MIMIC-CXR?

MIMIC-CXR is another widely used dataset for medical imaging research. However, accessing it requires credentialed access through PhysioNet.

To keep the MedInsight pipeline easier to reproduce and avoid additional access requirements, the current implementation uses openly accessible datasets. Support for MIMIC-CXR could be added as future work.

## Preprocessing

The preprocessing pipeline consists of the following steps:

* Resize images to **224 × 224** and normalize them using ImageNet statistics.
* Truncate captions to a maximum of **128 tokens** before indexing.
* Use VQA-RAD's **official 451-example held-out test split** for final evaluation rather than creating a new test split.
* Split the **official training portion (1,793 examples)** into 90% training and 10% validation sets.
* Stratify the internal split by answer type (closed vs. open) when the answer-type label is available.
* Use the complete ROCOv2 dataset as the retrieval corpus.

## Data Licensing

* **ROCOv2:** CC BY-NC-SA 4.0 — research and non-commercial use.
* **VQA-RAD:** CC0 — public domain.

The raw datasets are **not stored in this repository**. They can be downloaded automatically using the provided data preparation scripts.

## Running the Pipeline

Download the datasets and run the exploratory data analysis with:

```bash
python scripts/download_data.py --dataset all
python scripts/run_eda.py
```

The generated dataset statistics and exploratory analysis are saved to:

```text
docs/eda_findings.md
```

This file is generated automatically when the EDA pipeline is executed.
