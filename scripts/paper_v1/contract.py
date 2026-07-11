"""Immutable settings and checks for the Search-R1 arXiv v1 target."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class PaperV1Contract:
    git_commit: str
    model_id: str
    algorithm: str
    training_steps: int
    max_turns: int
    topk: int
    target_average_em: float
    dataset_revision: str


PAPER_V1 = PaperV1Contract(
    git_commit="118c6e7361bb68e33c525b50d62f83b63462799e",
    model_id="Qwen/Qwen2.5-3B-Instruct",
    algorithm="ppo",
    training_steps=305,
    max_turns=4,
    topk=3,
    target_average_em=0.327,
    dataset_revision="b7d80abfee334a7a91cb377544f09180d58b34f6",
)
REQUIRED_EVALUATION_DATASETS: Tuple[str, ...] = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
)
PAPER_V1_TARGET_EM: Dict[str, float] = {
    "nq": 0.323,
    "triviaqa": 0.537,
    "popqa": 0.364,
    "hotpotqa": 0.308,
    "2wikimultihopqa": 0.336,
    "musique": 0.105,
    "bamboogle": 0.315,
}
REQUIRED_ASSET_PATHS: Tuple[str, ...] = (
    "data/nq_hotpotqa_train/train.parquet",
    "data/nq_hotpotqa_train/test.parquet",
    "data/wiki18/e5_Flat.index",
    "data/wiki18/wiki-18.jsonl",
    "data/models/Qwen2.5-3B-Instruct/config.json",
    "data/models/e5-base-v2/config.json",
)


def assert_paper_commit(commit: str) -> None:
    """Raise when the checked-out source is not the paper-v1 target."""
    if commit.strip() != PAPER_V1.git_commit:
        raise ValueError(
            "Search-R1 paper v1 requires Git commit "
            f"{PAPER_V1.git_commit} (short: {PAPER_V1.git_commit[:7]}); found {commit.strip() or 'empty'}."
        )


def assess_required_assets(repo_root: Path) -> List[str]:
    """Return the required paper-v1 asset paths that are absent or empty."""
    errors: List[str] = []
    for relative_path in REQUIRED_ASSET_PATHS:
        path = repo_root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing required paper-v1 asset: {relative_path}")
    return errors


def assess_result_metrics(metrics: Mapping[str, float]) -> List[str]:
    """Return missing paper-v1 evaluation datasets from a metrics mapping."""
    missing = [dataset for dataset in REQUIRED_EVALUATION_DATASETS if dataset not in metrics]
    if not missing:
        return []
    return ["Missing evaluation metric(s): " + ", ".join(missing)]


def expected_metric_summary() -> Dict[str, float]:
    """Return the paper table target for the selected primary reproduction row."""
    return {"average_em": PAPER_V1.target_average_em}
