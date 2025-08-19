import numpy as np
from typing import List, Tuple, Dict, Any


def recall_at_k(retrieved_item_ids: list, correct_item_id: str) -> int:
    """
    Recall@k: Indicates whether the correct item is present in the top-k retrieved items.
    Returns 1 if the correct item is found, otherwise 0.
    """
    return int(correct_item_id in retrieved_item_ids)


def precision_at_k(retrieved_item_ids: list, correct_item_id: str) -> float:
    """
    Precision@k: Measures the proportion of retrieved items in the top-k that are relevant (i.e., match the correct item).
    """
    return np.mean([iid == correct_item_id for iid in retrieved_item_ids])


def average_precision(retrieved_item_ids: list, correct_item_id: str) -> float:
    """
    Average Precision (AP): Computes the average precision score for a single query.
    It averages the precision at each position where a relevant item is retrieved.
    """
    hits = [1 if iid == correct_item_id else 0 for iid in retrieved_item_ids]
    if sum(hits) == 0:
        return 0
    ap = 0
    num_hits = 0
    for i, hit in enumerate(hits):
        if hit:
            num_hits += 1
            ap += num_hits / (i + 1)
    ap /= sum(hits)
    return ap


def reciprocal_rank(retrieved_item_ids: list, correct_item_id: str) -> float:
    """
    Reciprocal Rank (RR): Returns the reciprocal of the rank at which the first relevant item is retrieved.
    If the correct item is not found, returns 0.
    """
    for i, iid in enumerate(retrieved_item_ids):
        if iid == correct_item_id:
            return 1.0 / (i + 1)
    return 0


def compute_metrics(results: List[Tuple[list, str]], k: int) -> Dict[str, Any]:
    """
    Computes aggregate metrics for a set of search results:
    - recall@k: Fraction of queries where the correct item is in the top-k results.
    - precision@k: Average precision for the top-k results across queries.
    - MAP (Mean Average Precision): Mean of average precision scores over all queries.
    - MRR (Mean Reciprocal Rank): Mean of reciprocal ranks over all queries.
    """

    recalls = []
    precisions = []
    average_precisions = []
    reciprocal_ranks = []
    for retrieved_item_ids, correct_item_id in results:
        recalls.append(recall_at_k(retrieved_item_ids, correct_item_id))
        precisions.append(precision_at_k(retrieved_item_ids, correct_item_id))
        average_precisions.append(
            average_precision(retrieved_item_ids, correct_item_id)
        )
        reciprocal_ranks.append(reciprocal_rank(retrieved_item_ids, correct_item_id))
    metrics = {
        f"recall@{k}": np.mean(recalls),
        f"precision@{k}": np.mean(precisions),
        "MAP": np.mean(average_precisions),
        "MRR": np.mean(reciprocal_ranks),
    }
    return metrics
