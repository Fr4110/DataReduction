import numpy as np
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def read_photo_matrix(filepath=None) -> np.ndarray:
    """
    Legge photos.csv e restituisce una matrice numpy (N, D).
    Indice 0 = foto con id 1.
    """
    path = Path(filepath) if filepath else _DEFAULT_DATA_DIR / "photos.csv"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().strip("[]")
            if line:
                rows.append(np.fromstring(line, sep=",", dtype=np.float32))
    matrix = np.vstack(rows)
    print(f"[data] Foto caricate: {matrix.shape[0]}, dim embedding: {matrix.shape[1]}")
    return matrix


def read_query_log(filepath=None) -> list:
    """
    Legge queries.csv e restituisce una lista di liste di int (id 1-indexed).
    """
    path = Path(filepath) if filepath else _DEFAULT_DATA_DIR / "queries.csv"
    log = []
    with open(path) as f:
        for line in f:
            line = line.strip().strip("[]")
            if line:
                log.append([int(x) for x in line.split(",")])
    print(f"[data] Query caricate: {len(log)}, "
          f"media risultati per query: {sum(len(q) for q in log)/len(log):.1f}")
    return log


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Matrice NxN di cosine similarity calcolata in modo efficiente."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    Xn = X / norms
    return (Xn @ Xn.T).astype(np.float32)


def compute_utility(kept: set, queries: list,
                    sim: np.ndarray,
                    max_q: int = None) -> float:
    """
    Calcola f(D') = media su tutte le query di S(q(D), q(D')).
    kept   : insieme di indici 0-based delle foto mantenute
    queries: lista di query (id 1-indexed)
    sim    : matrice NxN di cosine similarity
    """
    qs = queries if max_q is None else queries[:max_q]
    if not qs:
        return 0.0
    total = 0.0
    for q in qs:
        full = [i - 1 for i in q]
        reduced = [i for i in full if i in kept]
        if not full:
            continue
        if not reduced:
            continue
        total += sum(float(sim[d, reduced].max()) for d in full) / len(full)
    return total / len(qs)
