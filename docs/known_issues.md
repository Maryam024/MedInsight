# Known Issues & Fixes

This document summarizes notable issues identified during development and how they were resolved.

## Fixed Issues

### Answer Type Detection

**Issue:** The Hugging Face VQA-RAD dataset does not provide an `answer_type` field, causing all examples to be treated as open-ended.

**Fix:** Answer types are inferred from the answer text (`yes`/`no` → `CLOSED`; all others → `OPEN`).

---

### Official VQA-RAD Test Split

**Issue:** The dataset was initially re-split instead of using the official evaluation split.

**Fix:** The pipeline now uses the official Hugging Face `train` and `test` splits. The test set is evaluated without modification.

---

### Script Imports

**Issue:** Running scripts directly (`python scripts/<script>.py`) could not import modules from `src`.

**Fix:** All scripts add the repository root to `sys.path`, allowing them to run directly without additional environment configuration.

## Current Limitations

- Retrieval uses only image embeddings; question text is not included during retrieval.
- BLEU-4 is less informative for very short VQA answers and should be interpreted alongside ROUGE-L.
- Some unit tests use simplified mock objects instead of full Hugging Face processor outputs.