# Dataset Documentation

## Dataset Overview

MedInsight uses two datasets with distinct roles:

| Role | Dataset | Size | License |
|------|---------|------|---------|
| Retrieval corpus | **ROCOv2** | ~90,000 image-caption pairs | CC BY-NC-SA 4.0 |
| Evaluation benchmark | **VQA-RAD** | 3,515 QA pairs on 315 radiology images | CC0 |

## Dataset Selection

### ROCOv2

ROCOv2 is used as the retrieval corpus because it provides a large collection of radiology image-caption pairs from PubMed Central. It supports multiple imaging modalities (CT, MRI, X-ray, ultrasound) and is large enough to build a meaningful FAISS index while remaining practical to process on a single machine.

### VQA-RAD

VQA-RAD is used for evaluation because it is a widely used benchmark for medical visual question answering. The clinician-authored questions and answers provide a standardized way to compare baseline and retrieval-augmented models.

### Why not MIMIC-CXR?

Although MIMIC-CXR is widely used in medical imaging research, it requires credentialed PhysioNet access. This project uses openly accessible datasets to keep the entire pipeline reproducible without additional approval steps. Extending the retrieval corpus with MIMIC-CXR is left for future work.

## Preprocessing

The preprocessing pipeline includes:

- Resize images to **224×224** and normalize using ImageNet statistics.
- Truncate captions to **128 tokens** before indexing.
- Split VQA-RAD into **80/10/10** train/validation/test sets using stratified sampling by answer type.
- Index the complete ROCOv2 dataset for retrieval.

## Data Licensing

- **ROCOv2:** CC BY-NC-SA 4.0 (research and non-commercial use)
- **VQA-RAD:** CC0 (Public Domain)

Raw datasets are **not included** in this repository. They can be downloaded automatically using the provided scripts.

## Running the Pipeline

```bash
python scripts/download_data.py --dataset all
python scripts/run_eda.py
```

Dataset statistics and exploratory analysis are written to:

```
docs/eda_findings.md
```

This file is generated automatically after running the pipeline.