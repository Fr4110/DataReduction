"""
algorithms/brute_force.py  —  Method A

"""

from itertools import combinations
from math import comb
import numpy as np
from core.data import cosine_matrix, compute_utility


def run(photos: np.ndarray, queries: list,
        budget: int = None, max_q: int = None, sim: np.ndarray = None) -> dict:
    """
    Parameters
    ----------
    photos   : matrice (N, D)
    queries  : lista di query (id 1-indexed)
    budget   : numero di foto da mantenere (se None, calcola N - 3)
    max_q    : limita il numero di query usate (None = tutte)
    sim      : matrice di similarità precalcolata (opzionale)

    Returns
    -------
    dict con 'kept', 'deleted', 'utility', 'n_subsets'
    """
    N = photos.shape[0]
    
    # Default: elimina 3 foto se il budget non è specificato (come da consegna)
    if budget is None:
        budget = N - 3
        
    n_delete = N - budget
    n_sub = comb(N, n_delete)
    print(f"[A] N={N}, tengo {budget} foto (elimino {n_delete}) — {n_sub:,} sottoinsiemi")

    # Precalcola la matrice solo se non è stata fornita
    if sim is None:
        sim = cosine_matrix(photos)
        
    all_idx = set(range(N))

    best_u = -1.0
    best_del = None

    for deleted in combinations(range(N), n_delete):
        kept = all_idx - set(deleted)
        u = compute_utility(kept, queries, sim, max_q)
        if u > best_u:
            best_u = u
            best_del = deleted

    best_kept = sorted(all_idx - set(best_del))
    print(f"[A] Utility ottimale: {best_u:.6f}")
    print(f"[A] Foto eliminate (1-based): {[i+1 for i in best_del]}")

    return {
        "kept":     best_kept,
        "deleted":  list(best_del),
        "utility":  best_u,
        "n_subsets": n_sub,
    }