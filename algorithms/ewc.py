"""
algorithms/ewc.py  —  Method D (Entropy-Weighted Coverage, EWC)
----------------------------------------------------------------
Metodo originale proposto in questo progetto.

IDEA: "Entropy-Weighted Coverage"
----------------------------------------------------------------------
IndepDF (Method B) assegna ad ogni foto uno score proporzionale alla
sua frequenza nelle query. Tuttavia, due foto con la stessa frequenza
possono avere caratteristiche molto diverse:

  - Foto A: appare sempre nelle stesse 3 query → specializzata
  - Foto B: appare in 10 query diverse tra loro → generalista

Una foto generalista è più preziosa perché copre porzioni diverse
del log: se viene eliminata, molte query diverse perdono copertura.
Una foto specializzata è più rimpiazzabile perché le suas query
sono già coperte da foto simili.

EWC cattura questo con un termine di entropia:

    entropy(d) = -sum_q  p(q|d) * log(p(q|d))

dove p(q|d) = (1/|q(D)|) / Z_d è la distribuzione normalizzata
delle query in cui d appare, pesata per la scarsità del result set.

Lo score finale è:

    score_EWC(d) = alpha * freq_score(d) + (1-alpha) * norm_entropy(d)

dove freq_score è identico a IndepDF e norm_entropy è l'entropia
normalizzata nell'intervallo [0,1].

alpha ∈ [0,1] è un iperparametro (default 0.5).

VANTAGGI RISPETTO A B e C:
  - vs B: considera la diversità delle query coperte, non solo
          la frequenza; evita di sprecare budget su foto che
          coprono sempre le stesse query
  - vs C: complessità O(N·|Q|) invece di O(N·2^N); direttamente
          interpretabile in termini di teoria dell'informazione
  - vs A: polinomiale, scalabile a N grandi

COMPLESSITÀ: O(N · |Q|) — uguale a IndepDF, molto più veloce di A e C.
"""

import numpy as np
from typing import Optional


def _compute_scores(photos: np.ndarray, queries: list,
                    alpha: float = 0.5) -> np.ndarray:
    N = photos.shape[0]
    Q = len(queries)

    # freq_score: identico a IndepDF
    freq = np.zeros(N, dtype=np.float64)
    # Per l'entropia: accumula p(q|d) * log p(q|d) per ogni foto
    entropy_sum = np.zeros(N, dtype=np.float64)
    # Normalizzatore per p(q|d)
    z = np.zeros(N, dtype=np.float64)

    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        w = 1.0 / sz   # peso scarsità
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N:
                freq[idx] += w / Q
                z[idx] += w

    # Calcola entropia: itera di nuovo per accumulare p*log(p)
    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        w = 1.0 / sz
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N and z[idx] > 0:
                p = w / z[idx]          # p(q|d) normalizzata
                if p > 0:
                    entropy_sum[idx] -= p * np.log(p)

    # Normalizza entropia in [0,1]
    max_entropy = np.log(Q) if Q > 1 else 1.0
    norm_entropy = entropy_sum / max_entropy

    # Normalizza freq in [0,1]
    max_freq = freq.max()
    norm_freq = freq / max_freq if max_freq > 0 else freq

    return alpha * norm_freq + (1 - alpha) * norm_entropy


def run(photos: np.ndarray, queries: list,
        budget: Optional[int] = None,
        alpha: float = 0.5,
        sim=None) -> dict:
    """
    Parameters
    ----------
    photos : matrice (N, D)
    queries: lista di query (id 1-indexed)
    budget : numero di foto da tenere (default N//2)
    alpha  : peso tra freq_score e entropy_score (default 0.5)
    sim    : non usato, mantenuto per compatibilità con run.py

    Returns
    -------
    dict con 'kept', 'deleted', 'scores', 'budget'
    """
    N = photos.shape[0]
    if budget is None:
        budget = N // 2

    print(f"[D - EWC] N={N}, budget={budget}, alpha={alpha}")

    scores = _compute_scores(photos, queries, alpha)

    ranked  = np.argsort(scores)[::-1]
    kept    = sorted(ranked[:budget].tolist())
    deleted = sorted(ranked[budget:].tolist())

    n_active = int((scores > 0).sum())
    print(f"[D] Foto con score > 0: {n_active}")
    print(f"[D] Score max={scores.max():.4f}  mean={scores[scores>0].mean():.4f}")

    return {
        "kept":    kept,
        "deleted": deleted,
        "scores":  scores,
        "budget":  budget,
    }