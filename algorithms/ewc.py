"""
algorithms/ewc.py  —  Method D (Entropy-Weighted Coverage, EWC)
----------------------------------------------------------------
Original method proposed in this project.

IDEA: "Entropy-Weighted Coverage"
----------------------------------------------------------------------
IndepDF (Method B) assigns each photo a score proportional to its
frequency across queries. However, two photos with the same frequency
can have very different characteristics:

  - Photo A: always appears in the same 3 queries → specialised
  - Photo B: appears in 10 different, unrelated queries → generalised

A generalised photo is more valuable because it covers different
portions of the query log: if deleted, many distinct queries lose
coverage. A specialised photo is more replaceable because its queries
are already covered by similar photos.

EWC captures this with an entropy term:

    entropy(d) = -sum_q  p(q|d) * log(p(q|d))

where p(q|d) = (1/|q(D)|) / Z_d is the normalised distribution
over the queries in which d appears, weighted by result-set scarcity.

The final score is:

    score_EWC(d) = alpha * freq_score(d) + (1-alpha) * norm_entropy(d)

where freq_score is identical to IndepDF and norm_entropy is the
entropy normalised to the interval [0,1].

alpha ∈ [0,1] is a hyperparameter (default 0.5).

ADVANTAGES OVER B AND C:
  - vs B: accounts for the diversity of covered queries, not just
          frequency; avoids wasting budget on photos that always
          appear in the same queries
  - vs C: complexity O(N·|Q|) instead of O(N·2^N); directly
          interpretable in terms of information theory
  - vs A: polynomial, scalable to large N

COMPLEXITY: O(N · |Q|) — same as IndepDF, much faster than A and C.
"""

import numpy as np
from typing import Optional


def _compute_scores(photos: np.ndarray, queries: list,
                    alpha: float = 0.5) -> np.ndarray:
    N = photos.shape[0]
    Q = len(queries)

    freq = np.zeros(N, dtype=np.float64)
    entropy_sum = np.zeros(N, dtype=np.float64)
    z = np.zeros(N, dtype=np.float64)

    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        w = 1.0 / sz  
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N:
                freq[idx] += w / Q
                z[idx] += w

    for q in queries:
        sz = len(q)
        if sz == 0:
            continue
        w = 1.0 / sz
        for pid in q:
            idx = pid - 1
            if 0 <= idx < N and z[idx] > 0:
                p = w / z[idx]         
                if p > 0:
                    entropy_sum[idx] -= p * np.log(p)

    max_entropy = np.log(Q) if Q > 1 else 1.0
    norm_entropy = entropy_sum / max_entropy

    max_freq = freq.max()
    norm_freq = freq / max_freq if max_freq > 0 else freq

    return alpha * norm_freq + (1 - alpha) * norm_entropy


def run(photos: np.ndarray, queries: list,
        budget: Optional[int] = None,
        alpha: float = 0.5,
        sim=None) -> dict:
 
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
