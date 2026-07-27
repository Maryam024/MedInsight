from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from PIL import Image

from src.data.preprocessing import truncate_caption
from src.models.baseline_vlm import BaselineVLM
from src.retrieval.embedder import ImageEmbedder
from src.retrieval.index import RetrievalIndex
from src.utils.config_loader import MedInsightConfig, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievedEvidence(TypedDict):
    """One retrieved corpus item, as surfaced in `RAGVLM.generate`'s output."""

    pair_id: str
    caption: str
    similarity_score: float


class RAGResult(TypedDict):
    """Full output of `RAGVLM.generate`."""

    answer: str
    prompt: str
    retrieved_evidence: list[RetrievedEvidence]


def load_caption_lookup(manifest_path: str | Path) -> dict[str, str]:
    
    lookup: dict[str, str] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            lookup[record["pair_id"]] = record["caption"]
    return lookup


def build_prompt(
    question: str,
    captions: list[str],
    evidence_instruction: str,
    max_caption_words: int,
) -> str:
    
    if not captions:
        return question

    truncated = [truncate_caption(c, max_caption_words) for c in captions]
    evidence_block = "\n".join(f"- {c}" for c in truncated)
    return f"{evidence_instruction.strip()}\n{evidence_block}\n\nQuestion: {question} Answer:"


class RAGVLM:

    def __init__(
        self,
        vlm: BaselineVLM,
        embedder: ImageEmbedder,
        index: RetrievalIndex,
        caption_lookup: dict[str, str],
        top_k: int,
        evidence_instruction: str,
        max_caption_words_in_prompt: int,
    ) -> None:
        self.vlm = vlm
        self.embedder = embedder
        self.index = index
        self.caption_lookup = caption_lookup
        self.top_k = top_k
        self.evidence_instruction = evidence_instruction
        self.max_caption_words_in_prompt = max_caption_words_in_prompt

    @classmethod
    def from_config(
        cls,
        config: MedInsightConfig | None = None,
        index_dir: str | Path = "data/processed/rocov2/index",
        rocov2_manifest_path: str | Path = "data/processed/rocov2/manifest.jsonl",
        top_k_override: int | None = None,
    ) -> "RAGVLM":
        """Assemble a RAGVLM from config plus the Milestone 4 index on disk.

        Loads the same baseline VLM checkpoint as `BaselineVLM.from_config`,
        the CLIP image encoder, the pre-built FAISS index, and a caption
        lookup — everything needed to retrieve and prompt end to end.
        Requires `scripts/build_index.py` to have already been run.

        Parameters
        ----------
        top_k_override : int | None
            If set, overrides `configs/model_config.yaml -> retriever.top_k`
            for this instance only. Used by `scripts/run_experiments.py` to
            sweep top_k without editing the config file per run or re-loading
            the (expensive) model/index for every value tested.
        """
        if config is None:
            config = load_config()

        vlm = BaselineVLM.from_config(config)
        embedder = ImageEmbedder.from_config(config)
        index = RetrievalIndex.load(index_dir)
        caption_lookup = load_caption_lookup(rocov2_manifest_path)

        retriever_cfg = config.model["retriever"]
        rag_cfg = config.model["rag"]
        top_k = top_k_override if top_k_override is not None else retriever_cfg["top_k"]

        return cls(
            vlm=vlm,
            embedder=embedder,
            index=index,
            caption_lookup=caption_lookup,
            top_k=top_k,
            evidence_instruction=rag_cfg["evidence_instruction"],
            max_caption_words_in_prompt=rag_cfg["max_caption_words_in_prompt"],
        )

    def retrieve(self, image: Image.Image) -> list[RetrievedEvidence]:
       
        query_vector = self.embedder.embed_image(image)
        raw_results = self.index.search(query_vector, top_k=self.top_k)

        evidence: list[RetrievedEvidence] = []
        for pair_id, score in raw_results:
            caption = self.caption_lookup.get(pair_id)
            if caption is None:
                logger.warning(
                    "Retrieved pair_id '%s' not found in caption lookup "
                    "(index and manifest may be out of sync); skipping.",
                    pair_id,
                )
                continue
            evidence.append(
                {"pair_id": pair_id, "caption": caption, "similarity_score": score}
            )
        return evidence

    def generate(self, image: Image.Image, question: str) -> RAGResult:
        
        evidence = self.retrieve(image)
        captions = [item["caption"] for item in evidence]
        prompt = build_prompt(
            question,
            captions,
            self.evidence_instruction,
            self.max_caption_words_in_prompt,
        )
        answer = self.vlm.generate_from_prompt(image, prompt)
        return {"answer": answer, "prompt": prompt, "retrieved_evidence": evidence}
