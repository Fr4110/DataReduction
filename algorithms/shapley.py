"""
algorithms/shapley.py  —  Method C
"""

import numpy as np
from itertools import combinations
from math import factorial
from typing import Optional
from core.data import cosine_matrix, compute_utility


def _harmonic(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def _exact(photos, queries, sim, max_q):
    N = photos.shape[0]
    cache = {}

    def f(subset):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = compute_utility(set(subset), queries, sim, max_q)
        return cache[key]

    phi = np.zeros(N)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        val = 0.0
        for sz in range(len(others) + 1):
            w = factorial(sz) * factorial(N - sz - 1) / factorial(N)
            for S in combinations(others, sz):
                val += w * (f(S + (i,)) - f(S))
        phi[i] = val
        print(f"  phi[{i+1}] = {phi[i]:.6f}")
    return phi


def _closed_form(photos, queries):
    N = photos.shape[0]
    Q = len(queries)
    phi = np.zeros(N, dtype=np.float64)
    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        h = _harmonic(sz)
        w = h / (sz * Q)
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N:
                phi[idx] += w
    return phi


def run(photos: np.ndarray, queries: list,
        budget: int = 3,
        max_q: Optional[int] = None,
        sim: Optional[np.ndarray] = None) -> dict:
    """
    Parameters
    ----------
    photos : matrice (N, D)
    queries: lista di query
    budget : numero di foto da tenere (default 3 come da consegna)
    max_q  : limita il numero di query nel calcolo esatto
    sim    : matrice di similarità precalcolata (opzionale)

    Returns
    -------
    dict con 'kept', 'deleted', 'phi', 'set_value'
    """
    N = photos.shape[0]
    print(f"[C] N={N}, seleziono top-{budget} per valore di Shapley")

    if N <= 15:
        print("[C] Calcolo esatto")
        # Calcola la matrice di similarità se non è stata passata in input
        if sim is None:
            sim = cosine_matrix(photos)
        phi = _exact(photos, queries, sim, max_q)
    else:
        print("[C] Formula chiusa O(N x |Q|)")
        phi = _closed_form(photos, queries)

    # Usa il budget al posto di 'k'
    top = np.argsort(phi)[::-1][:budget].tolist()
    kept    = sorted(top)
    deleted = sorted(set(range(N)) - set(kept))

    print(f"[C] Set selezionato (1-based): {[i+1 for i in kept]}")
    print(f"[C] Somma Shapley values: {phi[kept].sum():.6f}")

    return {
        "kept":      kept,
        "deleted":   deleted,
        "phi":       phi,
        "set_value": float(phi[kept].sum()),
    }