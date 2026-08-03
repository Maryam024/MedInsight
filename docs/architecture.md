# System Architecture

## Retrieval-Augmented Pipeline

```text
                         ┌─────────────────────┐
                         │ Medical Image       │
                         │ + User Question     │
                         └─────────┬───────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
      ┌──────────────────────┐          ┌──────────────────────┐
      │ CLIP Image Encoder   │          │ Original Image       │
      │ (query embedding)    │          │                      │
      └──────────┬───────────┘          └──────────────────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │ FAISS Index          │
      │ (ROCOv2 embeddings)  │
      └──────────┬───────────┘
                 │
      Top-k retrieved evidence
      (image + caption pairs)
                 │
                 ▼
      ┌─────────────────────────────────────┐
      │ Evidence Prompt Builder             │
      │ (question + retrieved captions)     │
      └──────────┬──────────────────────────┘
                 │
                 ▼
      ┌─────────────────────────────────────┐
      │ BLIP-2 Vision-Language Model        │
      └──────────┬──────────────────────────┘
                 │
                 ▼
      ┌─────────────────────────────────────┐
      │ Generated Answer                    │
      │ + Retrieved Evidence                │
      └─────────────────────────────────────┘
```