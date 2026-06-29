"""
algorithms/indepdf.py  —  Method B
-------------------------------------
IndepDF greedy con Jaccard similarity.
Assegna ad ogni foto uno score proporzionale alla sua frequenza
nelle query, pesata per la dimensione inversa del result set.

Riferimento: Rico et al., SIGMOD 2026.
"""

import numpy as np
from typing import Optional


def score_photos(photos: np.ndarray, queries: list) -> np.ndarray:
    """
    score(d) = (1/|Q|) * sum_{q: d in q(D)} 1/|q(D)|

    Foto frequenti in query con pochi risultati ricevono score più alto.
    """
    N = photos.shape[0]
    Q = len(queries)
    scores = np.zeros(N, dtype=np.float64)
    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N:
                scores[idx] += 1.0 / (sz * Q)
    return scores


def run(photos: np.ndarray, queries: list,
        budget: Optional[int] = None) -> dict:
    """
    Parameters
    ----------
    photos : matrice (N, D)
    queries: lista di query
    budget : numero di foto da tenere (default N//2)

    Returns
    -------
    dict con 'kept', 'deleted', 'scores', 'budget'
    """
    N = photos.shape[0]
    if budget is None:
        budget = N // 2

    scores = score_photos(photos, queries)
    ranked = np.argsort(scores)[::-1]

    kept    = sorted(ranked[:budget].tolist())
    deleted = sorted(ranked[budget:].tolist())

    print(f"[B] Budget={budget} | "
          f"score max={scores.max():.4e} min={scores.min():.4e}")

    return {
        "kept":    kept,
        "deleted": deleted,
        "scores":  scores,
        "budget":  budget,
    }
