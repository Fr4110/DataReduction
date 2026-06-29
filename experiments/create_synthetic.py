"""
create_synthetic.py
--------------------
Genera un dataset sintetico di N foto e alcune query strutturate,
da usare per validare tutti e 4 i metodi (incluso A) su scala piccola.

USO:
    python create_synthetic.py

Produce data/photos.csv e data/queries.csv sovrascrivendo quelli reali.
ATTENZIONE: esegui questo PRIMA di run.py se vuoi il dataset sintetico,
poi rimetti i file reali per gli esperimenti su larga scala.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N   = 8     # numero di foto (piccolo abbastanza per Method A esatto)
DIM = 2048  # stessa dimensione del dataset reale
SEED = 42

rng = np.random.default_rng(SEED)

# Foto: vettori casuali quasi-ortogonali
photos = rng.standard_normal((N, DIM)).astype(np.float32)
photos /= np.linalg.norm(photos, axis=1, keepdims=True)

# Query strutturate: sliding window di 3 + query incrociate
queries = []
for i in range(N - 2):
    queries.append([i + 1, i + 2, i + 3])          # finestra scorrevole
queries.append([1, 4, 7])                            # query incrociata
queries.append([2, 5, 8])                            # query incrociata
queries.append([1, 3, 6, 8])                         # query lunga

# Salva photos.csv
with open(DATA_DIR / "photos.csv", "w") as f:
    for vec in photos:
        f.write(",".join(f"{v:.8f}" for v in vec) + "\n")

# Salva queries.csv
with open(DATA_DIR / "queries.csv", "w") as f:
    for q in queries:
        f.write(",".join(map(str, q)) + "\n")

print(f"Dataset sintetico creato:")
print(f"  {N} foto  ({DIM} dimensioni)")
print(f"  {len(queries)} query")
print(f"  File: data/photos.csv, data/queries.csv")
print()
print("Ora puoi lanciare:")
print("  python experiments/run.py")
print()
print("Per tornare al dataset reale, rimetti i file originali in data/")
