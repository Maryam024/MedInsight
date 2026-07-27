from __future__ import annotations

import re
import string
from typing import TypedDict

from src.utils.logger import get_logger

import nltk.translate.bleu_score as _bleu_module
from fractions import Fraction as _StdFraction


class _SafeFraction(_StdFraction):
    def __new__(cls, numerator=0, denominator=None, _normalize=True):
        return _StdFraction.__new__(cls, numerator, denominator)


_bleu_module.Fraction = _SafeFraction
# -------------------------------------------------------------------------

logger = get_logger(__name__)


class MetricsSummary(TypedDict):
    """Aggregate scores over a batch of predictions."""

    num_examples: int
    num_closed: int
    num_open: int
    closed_accuracy: float
    open_bleu: float
    open_rouge_l: float
    overall_exact_match: float


def normalize_answer(text: str) -> str:

    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized prediction equals the normalized reference, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def compute_bleu(prediction: str, reference: str) -> float:
    
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing))


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between a prediction and a single reference answer."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(normalize_answer(reference), normalize_answer(prediction))
    return float(scores["rougeL"].fmeasure)


def compute_metrics_batch(
    predictions: list[str],
    references: list[str],
    answer_types: list[str],
) -> MetricsSummary:
    
    if not (len(predictions) == len(references) == len(answer_types)):
        raise ValueError(
            f"predictions, references, and answer_types must be the same length, "
            f"got {len(predictions)}, {len(references)}, {len(answer_types)}"
        )
    if not predictions:
        raise ValueError("Cannot compute metrics over an empty batch.")

    closed_matches = []
    open_bleu_scores = []
    open_rouge_scores = []
    all_exact_matches = []

    for pred, ref, atype in zip(predictions, references, answer_types):
        em = compute_exact_match(pred, ref)
        all_exact_matches.append(em)
        if atype == "CLOSED":
            closed_matches.append(em)
        else:
            open_bleu_scores.append(compute_bleu(pred, ref))
            open_rouge_scores.append(compute_rouge_l(pred, ref))

    summary: MetricsSummary = {
        "num_examples": len(predictions),
        "num_closed": len(closed_matches),
        "num_open": len(open_bleu_scores),
        "closed_accuracy": (
            sum(closed_matches) / len(closed_matches) if closed_matches else 0.0
        ),
        "open_bleu": (
            sum(open_bleu_scores) / len(open_bleu_scores) if open_bleu_scores else 0.0
        ),
        "open_rouge_l": (
            sum(open_rouge_scores) / len(open_rouge_scores) if open_rouge_scores else 0.0
        ),
        "overall_exact_match": sum(all_exact_matches) / len(all_exact_matches),
    }

    logger.info("Computed metrics over %d examples: %s", len(predictions), summary)
    return summary
