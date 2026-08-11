# Known Issues & Fixes

This document highlights the main issues identified during development, along with the solutions implemented to address them.

## Fixed Issues

### Answer Type Detection

**Issue:** The Hugging Face VQA-RAD dataset does not include an `answer_type` field. As a result, all samples were initially classified as open-ended questions.

**Fix:** The answer type is now inferred directly from the answer text. Answers such as `yes` or `no` are classified as `CLOSED`, while all other answers are classified as `OPEN`.

---

### Official VQA-RAD Test Split

**Issue:** The dataset was initially re-split manually instead of using the official evaluation split provided by the dataset.

**Fix:** The pipeline now uses the official Hugging Face `train` and `test` splits. The test split is evaluated as provided, without further modification.

---

### Script Import Issues

**Issue:** Running scripts directly using `python scripts/<script>.py` could result in import errors because modules inside `src` were not found.

**Fix:** The repository root is now added to `sys.path` by the scripts, allowing them to be executed directly without requiring additional environment configuration.

## Current Limitations

* Retrieval currently relies only on image embeddings; question text is not incorporated into the retrieval process.
* BLEU-4 can be less informative for very short VQA answers, so it should be considered alongside ROUGE-L and other evaluation metrics.
* Some unit tests use simplified mock objects rather than complete Hugging Face processor outputs, so they do not fully reproduce the behavior of the production pipeline.
