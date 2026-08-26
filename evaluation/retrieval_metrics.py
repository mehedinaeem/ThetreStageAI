"""Standard ranked-retrieval metrics with explicit zero-result behavior."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    retrieved_count: int
    relevant_count: int


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Relevant results in the first K positions divided by K."""
    _validate_k(k)
    hits = sum(source_id in relevant_ids for source_id in retrieved_ids[:k])
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Relevant results recovered in the first K positions."""
    _validate_k(k)
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result, or zero when absent."""
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: set[str] | Mapping[str, float],
    k: int,
) -> float:
    """Normalized discounted cumulative gain using binary or graded relevance."""
    _validate_k(k)
    grades = (
        {source_id: 1.0 for source_id in relevance}
        if isinstance(relevance, set)
        else {source_id: float(grade) for source_id, grade in relevance.items()}
    )
    if any(grade < 0 for grade in grades.values()):
        raise ValueError("Relevance grades cannot be negative")

    dcg = sum(
        (2.0 ** grades.get(source_id, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, source_id in enumerate(retrieved_ids[:k], start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2.0 ** grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    *,
    k: int,
    graded_relevance: Mapping[str, float] | None = None,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_ids, relevant_ids, k),
        recall_at_k=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=reciprocal_rank(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, graded_relevance or relevant_ids, k),
        k=k,
        retrieved_count=len(retrieved_ids),
        relevant_count=len(relevant_ids),
    )


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("K must be a positive integer")
